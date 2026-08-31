from __future__ import annotations

import argparse
import asyncio
from pathlib import Path

from langchain_core.messages import AIMessageChunk

from orcha_agent.core.events import (
    Event,
    ModelChunk,
    ToolCallEnd,
    ToolCallStart,
    TurnEnd,
    TurnStart,
)
from orcha_agent.tui.runtime import ApplicationRuntime


class TmuxDriver:
    def __init__(self, state_path: Path) -> None:
        self.state_path = state_path
        self.runtime = ApplicationRuntime(self.submit)

    def signal(self, state: str) -> None:
        with self.state_path.open("a", encoding="utf-8") as stream:
            stream.write(f"{state}\n")

    async def present(self, event: Event) -> None:
        await self.runtime.handle_presentation(event)
        await self.runtime.transcript.handle(event)

    async def stream_turn(self, label: str) -> None:
        prefix = f"TMUX_{label.upper()}"
        start = TurnStart(thread_id="tmux", text=f"turn-{label}")
        await self.present(start)
        await asyncio.sleep(0.2)
        for index in range(1, 31):
            chunk = ModelChunk(
                chunk=AIMessageChunk(content=f"- {prefix}_{index:02d}\n"),
                role="main",
                source_id="main",
            )
            await self.present(chunk)
            if index == 1:
                self.signal(f"turn-{label}-active")
            await asyncio.sleep(0.04)
        self.signal(f"turn-{label}-streamed")
        end = TurnEnd(thread_id="tmux")
        await self.runtime.transcript.handle(end)
        await self.runtime.handle_presentation(end)
        await asyncio.sleep(0.2)
        self.signal(f"turn-{label}-done")

    async def fanout(self) -> None:
        start = TurnStart(thread_id="tmux", text="fanout")
        await self.present(start)
        await asyncio.sleep(0.2)
        call = ToolCallStart(
            name="task",
            id="tmux-fanout",
            args={
                "tasks": [
                    {"name": "alpha", "agent": "task", "task": "return alpha"},
                    {"name": "beta", "agent": "task", "task": "return beta"},
                    {"name": "gamma", "agent": "task", "task": "return gamma"},
                ]
            },
        )
        await self.present(call)
        self.signal("fanout-active")
        for _ in range(30):
            self.runtime.application.invalidate()
            await asyncio.sleep(0.04)
        self.signal("fanout-streamed")
        result = ToolCallEnd(
            name="task",
            id="tmux-fanout",
            result={"jobs": []},
        )
        await self.present(result)
        end = TurnEnd(thread_id="tmux")
        await self.runtime.transcript.handle(end)
        await self.runtime.handle_presentation(end)
        await asyncio.sleep(0.2)
        self.signal("fanout-done")

    async def submit(self, text: str) -> None:
        command = text.strip()
        if command in {"turn-a", "turn-b"}:
            await self.stream_turn(command[-1])
        elif command == "fanout":
            await self.fanout()

    async def run(self) -> None:
        await self.runtime.run()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--state", type=Path, required=True)
    return parser.parse_args()


async def main() -> None:
    args = parse_args()
    for index in range(1, 36):
        print(f"TMUX_STARTUP_{index:02d}", flush=True)
    driver = TmuxDriver(args.state)
    await driver.run()


if __name__ == "__main__":
    asyncio.run(main())
