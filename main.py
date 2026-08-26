"""Minimal LangChain deep agent (deepagents) example."""

import sys

from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from deepagents import create_deep_agent

load_dotenv()

MODEL = ChatAnthropic(
    model="claude-opus-5",
    max_tokens=16000,
    thinking={"type": "adaptive"},
)

SYSTEM_PROMPT = """You are a careful research and coding assistant.
Break large tasks into steps with the todo tool, use the filesystem tools
to persist notes/results, and delegate self-contained subtasks to subagents."""


def build_agent():
    return create_deep_agent(
        model=MODEL,
        tools=[],
        system_prompt=SYSTEM_PROMPT,
        # Persist conversation state across invocations (in-memory checkpointer).
        checkpointer=True,
    )


def main() -> None:
    prompt = " ".join(sys.argv[1:]) or "Introduce yourself and list your tools."
    agent = build_agent()
    config = {"configurable": {"thread_id": "cli"}}
    result = agent.invoke({"messages": [{"role": "user", "content": prompt}]}, config=config)
    print(result["messages"][-1].content)


if __name__ == "__main__":
    main()
