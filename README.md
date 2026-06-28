# memory-store-mentor

An experiment in giving a personal AI coach persistent memory.

Built on [Anthropic Managed Agents](https://docs.anthropic.com/en/docs/agents), this project wires together three things:

- **A managed agent** configured as a psychologist and personal mentor, running on Anthropic's infrastructure
- **A memory store** attached to each session, giving the agent access to a persistent, structured record of the user — facts, patterns, goals, reflections accumulated over time
- **A Discord bot** that acts as the chat interface, routing DMs to the agent and back

The agent reads from and writes to the memory store during conversations, so it builds up a picture of who you are across sessions rather than starting fresh each time.

## Architecture

```
Discord DM → bot.py → Anthropic Sessions API → Managed Agent (+ Memory Store)
```

The bot runs as a worker process on [Railway](https://railway.com). Each Discord user gets their own agent session with the memory store mounted read/write.

## Setup

### Prerequisites

- Anthropic API key with access to Managed Agents
- A configured agent and memory store on [platform.claude.com](https://platform.claude.com)
- A Discord bot token (created at [discord.com/developers](https://discord.com/developers/applications))

### Environment variables

```
ANTHROPIC_API_KEY=sk-ant-...
DISCORD_TOKEN=your-discord-bot-token
AGENT_ID=agent_01...
MEMORY_STORE_ID=memstore_01...
```

### Run locally

```bash
uv sync
uv run python bot.py
```

### Deploy

Push to GitHub and connect the repo to a Railway project. Add the environment variables under the service's Variables tab. Railway picks up the `Procfile` and runs the bot as a worker.

## Notes

- Only one instance of the bot should run at a time (local or Railway, not both) — Discord doesn't support multiple concurrent connections with the same token
- Session IDs are held in memory and reset on restart; long-term context is preserved by the memory store, not the session
- The memory store can be pre-populated using `populate_memory.py`
