import os
import asyncio
import logging
from dotenv import load_dotenv
import discord
from discord.ext import commands
import anthropic

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

AGENT_ID = os.environ["AGENT_ID"]
MEMORY_STORE_ID = os.environ["MEMORY_STORE_ID"]
# Environment ID for the agent's container — fetch once at startup
ENVIRONMENT_ID: str | None = None

anthropic_client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
DISCORD_TOKEN = os.environ["DISCORD_TOKEN"]

# Maps Discord user ID -> Anthropic session ID.
user_sessions: dict[int, str] = {}

# Rotate the session after this many user turns to keep context (and cost) bounded.
# The memory store holds long-term facts; the short-term thread of the conversation
# is carried across rotations by a handoff (summary + last few exchanges) that is
# prepended to the first message of the new session.
SESSION_ROTATION_TURNS = 15
user_turn_counts: dict[int, int] = {}

# (user text, agent reply) pairs for the current session, used to build the handoff.
user_history: dict[int, list[tuple[str, str]]] = {}
# Handoff text waiting to be prepended to the next message sent to a fresh session.
user_handoff: dict[int, str] = {}
# One in-flight request per user: a session can only process one turn at a time.
user_locks: dict[int, asyncio.Lock] = {}

SUMMARY_MODEL = "claude-opus-5-5"
HANDOFF_VERBATIM_EXCHANGES = 3  # most recent exchanges carried over word for word
HANDOFF_EXCHANGE_CHARS = 2000   # per-message cap for the verbatim part

SUMMARY_SYSTEM = """You summarize a conversation between a user and their AI mentor so the mentor can continue it seamlessly in a new session.

Write a compact summary (at most ~300 words) covering:
- What the user shared in this conversation (events, people, feelings, plans), with names and dates.
- What the mentor said, asked or suggested, and how the user responded.
- Open threads: questions the mentor asked that are still unanswered, topics the user is in the middle of.
- Which language each side writes in, and the tone. Describe this as it is; do not suggest changing it.

Write plain prose or short bullets. No preamble."""


def resolve_environment_id() -> str:
    """Find the environment ID by looking at an existing session, or fetching environments."""
    sessions = anthropic_client.beta.sessions.list()
    for s in sessions.data:
        if s.agent.id == AGENT_ID and s.environment_id:
            return s.environment_id
    # Fallback: list environments directly
    envs = anthropic_client.beta.environments.list()
    return envs.data[0].id


def _create_session(discord_user_id: int) -> str:
    session = anthropic_client.beta.sessions.create(
        agent=AGENT_ID,
        environment_id=ENVIRONMENT_ID,
        resources=[{
            "type": "memory_store",
            "memory_store_id": MEMORY_STORE_ID,
            "access": "read_write",
        }],
    )
    user_sessions[discord_user_id] = session.id
    log.info("Created session %s for user %d", session.id, discord_user_id)
    return session.id


def get_or_create_session(discord_user_id: int) -> str:
    if discord_user_id in user_sessions:
        return user_sessions[discord_user_id]
    return _create_session(discord_user_id)


def _format_transcript(history: list[tuple[str, str]], name: str) -> str:
    return "\n\n".join(f"{name}: {u}\n\nMentor: {a}" for u, a in history)


def _summarize_sync(history: list[tuple[str, str]], name: str) -> str | None:
    try:
        response = anthropic_client.messages.create(
            model=SUMMARY_MODEL,
            max_tokens=4000,
            output_config={"effort": "low"},
            system=SUMMARY_SYSTEM,
            messages=[{
                "role": "user",
                "content": f"<conversation>\n{_format_transcript(history, name)}\n</conversation>",
            }],
        )
    except anthropic.APIError:
        log.exception("Summary request failed")
        return None
    if response.stop_reason == "refusal":
        log.warning("Summary request was refused")
        return None
    text = "\n".join(b.text for b in response.content if b.type == "text").strip()
    return text or None


def _truncate(text: str, limit: int) -> str:
    return text if len(text) <= limit else text[:limit] + " […]"


def _build_handoff(history: list[tuple[str, str]], name: str) -> str:
    summary = _summarize_sync(history, name)
    recent = [
        (_truncate(u, HANDOFF_EXCHANGE_CHARS), _truncate(a, HANDOFF_EXCHANGE_CHARS))
        for u, a in history[-HANDOFF_VERBATIM_EXCHANGES:]
    ]
    parts = [
        f"[Note from the chat bridge, not from {name}: to keep context bounded, the "
        "previous session was closed and this is a new one. From the user's side it is "
        "the same, uninterrupted conversation. Do not greet them, do not say it has been "
        "a while, and do not re-introduce topics as if new. Pick up exactly where the "
        "conversation below left off, replying in the same language and style as in those exchanges.]",
    ]
    if summary:
        parts.append(f"<conversation_summary>\n{summary}\n</conversation_summary>")
    parts.append(f"<most_recent_exchanges>\n{_format_transcript(recent, name)}\n</most_recent_exchanges>")
    return "\n\n".join(parts)


def _rotate_sync(discord_user_id: int, name: str) -> None:
    """Close out the current session: build a handoff and pre-create the next session."""
    history = user_history.pop(discord_user_id, [])
    old_session = user_sessions.pop(discord_user_id, None)
    user_turn_counts[discord_user_id] = 0
    if history:
        user_handoff[discord_user_id] = _build_handoff(history, name)
    log.info("Rotated session for user %d (old: %s, %d exchanges)", discord_user_id, old_session, len(history))
    # Create the next session now so the user's next message doesn't wait for it.
    try:
        _create_session(discord_user_id)
    except anthropic.APIError:
        log.exception("Pre-creating session failed; will retry on next message")


def _query_agent_sync(session_id: str, text: str) -> str:
    """Send a user message and collect the agent's response via event stream."""
    parts: list[str] = []
    # Open the stream before sending so no events are missed.
    with anthropic_client.beta.sessions.events.stream(session_id) as stream:
        anthropic_client.beta.sessions.events.send(
            session_id,
            events=[{
                "type": "user.message",
                "content": [{"type": "text", "text": text}],
            }],
        )
        for event in stream:
            if event.type == "agent.message":
                for block in event.content:
                    if hasattr(block, "text"):
                        parts.append(block.text)
            elif event.type == "session.status_terminated":
                break
            elif event.type == "session.status_idle":
                # requires_action idles are transient (e.g. between tool calls); only
                # end_turn / retries_exhausted / budget_reached end the turn.
                if event.stop_reason.type == "requires_action":
                    continue
                if event.stop_reason.type != "end_turn":
                    log.warning("Session %s went idle with %s", session_id, event.stop_reason.type)
                break

    return "\n".join(parts) or "(no response)"


async def _run(func, *args):
    return await asyncio.get_running_loop().run_in_executor(None, func, *args)


intents = discord.Intents.default()
intents.message_content = True
intents.dm_messages = True

bot = commands.Bot(command_prefix="!", intents=intents)


@bot.event
async def on_ready():
    global ENVIRONMENT_ID
    log.info("Logged in as %s (id %d)", bot.user, bot.user.id)
    ENVIRONMENT_ID = await _run(resolve_environment_id)
    log.info("Using environment_id: %s", ENVIRONMENT_ID)


async def _rotate_in_background(uid: int, name: str) -> None:
    async with user_locks[uid]:
        try:
            await _run(_rotate_sync, uid, name)
        except Exception:
            log.exception("Session rotation failed for user %d", uid)


@bot.event
async def on_message(message: discord.Message):
    if message.author.bot:
        return

    is_dm = isinstance(message.channel, discord.DMChannel)
    is_mention = bot.user in message.mentions

    if not (is_dm or is_mention):
        return

    text = message.content
    if is_mention:
        text = text.replace(f"<@{bot.user.id}>", "").replace(f"<@!{bot.user.id}>", "").strip()

    if not text:
        return

    uid = message.author.id
    name = message.author.display_name
    lock = user_locks.setdefault(uid, asyncio.Lock())

    rotate = False
    async with lock, message.channel.typing():
        try:
            session_id = await _run(get_or_create_session, uid)
            handoff = user_handoff.get(uid)
            outgoing = f"{handoff}\n\n{name}'s new message:\n\n{text}" if handoff else text
            reply = await _run(_query_agent_sync, session_id, outgoing)
            user_handoff.pop(uid, None)  # delivered; drop only after success
            user_history.setdefault(uid, []).append((text, reply))
            user_turn_counts[uid] = user_turn_counts.get(uid, 0) + 1
            rotate = user_turn_counts[uid] >= SESSION_ROTATION_TURNS
        except Exception as e:
            log.exception("Error querying agent")
            reply = f"Something went wrong: {e}"

    for chunk in _split(reply, 2000):
        await message.channel.send(chunk)

    if rotate:
        # Summarize and pre-create the next session after replying, off the hot path.
        asyncio.create_task(_rotate_in_background(uid, name))


def _split(text: str, limit: int) -> list[str]:
    """Split into Discord-sized chunks, preferring paragraph, line, then word boundaries."""
    chunks = []
    while len(text) > limit:
        cut = max(text.rfind("\n\n", 0, limit), text.rfind("\n", 0, limit), text.rfind(" ", 0, limit))
        if cut <= 0:
            cut = limit
        chunks.append(text[:cut].rstrip())
        text = text[cut:].lstrip()
    if text:
        chunks.append(text)
    return chunks


if __name__ == "__main__":
    bot.run(DISCORD_TOKEN)
