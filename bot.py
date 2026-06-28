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
# Sessions are long-lived; the agent's attached memory store preserves context across restarts.
user_sessions: dict[int, str] = {}


def resolve_environment_id() -> str:
    """Find the environment ID by looking at an existing session, or fetching environments."""
    sessions = anthropic_client.beta.sessions.list()
    for s in sessions.data:
        if s.agent.id == AGENT_ID and s.environment_id:
            return s.environment_id
    # Fallback: list environments directly
    envs = anthropic_client.beta.environments.list()
    return envs.data[0].id


def get_or_create_session(discord_user_id: int) -> str:
    if discord_user_id not in user_sessions:
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
    return user_sessions[discord_user_id]


def _query_agent_sync(session_id: str, text: str) -> str:
    """Send a user message and collect the agent's response via event stream."""
    anthropic_client.beta.sessions.events.send(
        session_id,
        events=[{
            "type": "user.message",
            "content": [{"type": "text", "text": text}],
        }],
    )

    parts: list[str] = []
    with anthropic_client.beta.sessions.events.stream(session_id) as stream:
        for event in stream:
            if event.type == "agent.message":
                for block in event.content:
                    if hasattr(block, "text"):
                        parts.append(block.text)
            elif event.type == "session.status_idle":
                break

    return "\n".join(parts) or "(no response)"


async def query_agent(session_id: str, text: str) -> str:
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, _query_agent_sync, session_id, text)


intents = discord.Intents.default()
intents.message_content = True
intents.dm_messages = True

bot = commands.Bot(command_prefix="!", intents=intents)


@bot.event
async def on_ready():
    global ENVIRONMENT_ID
    log.info("Logged in as %s (id %d)", bot.user, bot.user.id)
    loop = asyncio.get_event_loop()
    ENVIRONMENT_ID = await loop.run_in_executor(None, resolve_environment_id)
    log.info("Using environment_id: %s", ENVIRONMENT_ID)


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

    async with message.channel.typing():
        try:
            session_id = get_or_create_session(message.author.id)
            reply = await query_agent(session_id, text)
        except Exception as e:
            log.exception("Error querying agent")
            reply = f"Something went wrong: {e}"

    for chunk in _split(reply, 2000):
        await message.channel.send(chunk)


def _split(text: str, limit: int) -> list[str]:
    if len(text) <= limit:
        return [text]
    chunks = []
    while text:
        chunks.append(text[:limit])
        text = text[limit:]
    return chunks


if __name__ == "__main__":
    bot.run(DISCORD_TOKEN)
