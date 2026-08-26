# deepagent

LangChain [deepagents](https://github.com/langchain-ai/deepagents) starter (Python, uv).

## Setup

```bash
cp .env.example .env   # add ANTHROPIC_API_KEY
uv sync
uv run main.py "Plan and write a haiku about Arch Linux"
```

Agent ships with built-in todo list, filesystem tools, and subagent delegation.
Add your own tools via `tools=[...]` in `main.py`.
