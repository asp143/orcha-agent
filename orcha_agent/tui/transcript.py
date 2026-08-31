"""Translate kernel events and console output into transcript blocks."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from orcha_agent.core.events import (
    Advisory,
    AgentDelivered,
    AgentFinished,
    AgentSpawned,
    AgentStatus,
    ModelChunk,
    ThreadSwitch,
    ToolCallEnd,
    ToolCallStart,
    TurnEnd,
    TurnStart,
)
from orcha_agent.core.registry import Registry

from .frame import Block, BlockState, Frame, FrameScheduler


def _matches(match: Any, event: object) -> bool:
    if isinstance(match, type):
        return isinstance(event, match)
    if callable(match):
        return bool(match(event))
    return match == type(event).__name__ or match == getattr(event, "name", None)


def _text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, Mapping):
        for key in ("text", "content"):
            if key in value:
                return _text(value[key])
        return ""
    if isinstance(value, (list, tuple)):
        return "".join(_text(item) for item in value)
    return str(value)


def _thinking_text(part: Mapping[str, Any]) -> str:
    if part.get("type") == "reasoning":
        return _text(part.get("summary"))
    if part.get("type") == "thinking":
        return _text(part.get("thinking"))
    return ""


def _reasoning_tokens(chunk: Any) -> int | None:
    usage = getattr(chunk, "usage_metadata", None)
    if not isinstance(usage, Mapping):
        return None
    details = usage.get("output_token_details")
    sources = (details, usage)
    for source in sources:
        if not isinstance(source, Mapping):
            continue
        for key in ("reasoning", "reasoning_tokens"):
            value = source.get(key)
            if isinstance(value, (int, float)):
                return max(0, int(value))
    return None


class Transcript:
    """Event sink that owns source-aware accumulated transcript state."""

    def __init__(
        self,
        frame: Frame | None = None,
        *,
        registry: Registry | Any | None = None,
        scheduler: FrameScheduler | None = None,
    ) -> None:
        self.frame = frame or Frame()
        self.registry = registry
        self.scheduler = scheduler
        self._source_blocks: dict[tuple[str, str], Block] = {}
        self._tools: dict[str, Block] = {}
        self._read_groups: dict[str, Block] = {}
        self._task_blocks: dict[str, Block] = {}
        self._agent_tasks: dict[str, Block] = {}
        self._parent_tasks: dict[str, Block] = {}
        self._deliveries: dict[str, Block] = {}

    def _float_active_tasks(self) -> None:
        active = [
            block
            for block in self.frame.blocks
            if block.kind == "task" and block.state is BlockState.ACTIVE
        ]
        if active:
            active_ids = {id(block) for block in active}
            self.frame.blocks[:] = [
                block for block in self.frame.blocks if id(block) not in active_ids
            ] + active

    def _commit(self, block: Block, *, immediate: bool = False) -> None:
        if block.kind != "task":
            self._float_active_tasks()
        block.settle()
        if self.scheduler is None:
            self.frame.commit_ready()
        elif immediate:
            self.scheduler.commit_now()
        else:
            self.scheduler.request_commit()

    def append_raw(
        self,
        renderable: Any,
        *,
        immediate: bool = False,
        **options: Any,
    ) -> Block:
        block = self.frame.add(
            "raw",
            {"renderable": renderable, "options": options, "level": "raw"},
        )
        self._commit(block, immediate=immediate)
        return block

    def append_banner(
        self,
        message: str,
        *,
        level: str = "error",
        immediate: bool = False,
    ) -> Block:
        lines = message.splitlines()
        if level == "error" and len(lines) > 8:
            lines = [*lines[:7], "…"]
        block = self.frame.add("banner", {"message": "\n".join(lines), "level": level})
        self._commit(block, immediate=immediate)
        return block

    def append_welcome(
        self,
        data: Mapping[str, Any],
        *,
        immediate: bool = True,
    ) -> Block:
        block = self.frame.add("welcome", data)
        self._commit(block, immediate=immediate)
        return block

    def _legacy(self, event: object) -> bool:
        if self.registry is None:
            return False
        for registration in self.registry.renderers:
            if not _matches(registration.match, event):
                continue
            rendered = registration.render(event)
            if rendered is None:
                continue
            self.append_raw(
                rendered,
                immediate=False,
                end="" if isinstance(event, ModelChunk) else "\n",
            )
            return True
        return False

    async def handle(self, event: object) -> None:
        if isinstance(event, TurnStart):
            for prior in self.frame.blocks:
                if prior.state is BlockState.ACTIVE and prior.kind != "task":
                    prior.settle()
            self._source_blocks.clear()
            self._tools = {
                identifier: block
                for identifier, block in self._tools.items()
                if block.kind == "task" and block.state is BlockState.ACTIVE
            }
            self._read_groups.clear()
            if (
                event.text.startswith("<system-notification>")
                and event.text.rstrip().endswith("</system-notification>")
            ):
                return
            if self._legacy(event):
                return
            block = self.frame.add(
                "user",
                {"text": event.text, "thread_id": event.thread_id},
                source_id=str(event.source_id or event.thread_id),
            )
            self._commit(block, immediate=True)
            if self.scheduler is not None:
                self.scheduler.render_now()
            return
        if isinstance(event, Advisory):
            if event.note is None:
                return
            block = self.frame.add(
                "advisory",
                {
                    "note": event.note,
                    "severity": event.severity,
                    "advisor_id": event.advisor_id,
                    "interrupt": event.interrupt,
                },
                source_id=event.advisor_id,
            )
            self._commit(block)
            return
        if self._legacy(event):
            return
        if isinstance(event, AgentSpawned):
            self._agent_spawned(event)
            return
        if isinstance(event, AgentStatus):
            self._agent_status(event)
            return
        if isinstance(event, AgentFinished):
            self._agent_finished(event)
            return
        if isinstance(event, AgentDelivered):
            self._agent_delivered(event)
            return
        if isinstance(event, ModelChunk):
            self._model_chunk(event)
            return
        if isinstance(event, ToolCallStart):
            self._tool_start(event)
            return
        if isinstance(event, ToolCallEnd):
            self._tool_end(event)
            return
        if isinstance(event, ThreadSwitch):
            labels = {
                "compact": "⊟ compacted",
                "clear": "⊠ cleared",
                "branch": f"⎇ branched to {event.new}",
            }
            block = self.frame.add(
                "marker",
                {
                    "text": labels.get(event.reason, event.reason),
                    "old": event.old,
                    "new": event.new,
                    "session_id": event.session_id,
                },
            )
            self._commit(block)
            return
        if isinstance(event, TurnEnd):
            for block in self.frame.blocks:
                if block.state is BlockState.ACTIVE and block.kind != "task":
                    block.settle()
            self._float_active_tasks()
            if self.scheduler is not None:
                self.scheduler.request_commit()
                self.scheduler.request_invalidate()
            return
        if isinstance(event, BaseException):
            self.append_banner(f"{type(event).__name__}: {event}")

    @staticmethod
    def _task_agents(block: Block) -> list[dict[str, Any]]:
        value = block.data.get("agents")
        if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
            return []
        return [dict(item) for item in value if isinstance(item, Mapping)]

    @staticmethod
    def _task_placeholders(event: ToolCallStart) -> list[dict[str, Any]]:
        tasks = event.args.get("tasks")
        if not isinstance(tasks, Sequence) or isinstance(tasks, (str, bytes)):
            return []
        placeholders: list[dict[str, Any]] = []
        for index, value in enumerate(tasks):
            item = dict(value) if isinstance(value, Mapping) else {"task": str(value)}
            placeholders.append(
                {
                    "run_id": f"pending:{event.id}:{index}",
                    "name": item.get("name") or item.get("agent") or f"agent {index + 1}",
                    "agent_type": item.get("agent") or "task",
                    "description": item.get("task") or "",
                    "status": "pending",
                }
            )
        return placeholders

    def _new_task_block(
        self,
        *,
        identifier: str,
        parent_id: str,
        args: Mapping[str, Any] | None = None,
        agents: Sequence[Mapping[str, Any]] = (),
        tool_complete: bool = True,
    ) -> Block:
        block = self.frame.add(
            "task",
            {
                "name": "task",
                "id": identifier,
                "args": dict(args or {}),
                "agents": [dict(agent) for agent in agents],
                "tool_complete": tool_complete,
            },
            source_id=parent_id,
        )
        self._task_blocks[identifier] = block
        return block

    def _task_for_agent(self, parent_id: str, run_id: str, name: str) -> Block:
        known = self._agent_tasks.get(run_id)
        if known is not None and known.state is BlockState.ACTIVE:
            return known

        candidates = [
            block
            for block in self._task_blocks.values()
            if block.state is BlockState.ACTIVE
            and str(block.source_id or "main") == parent_id
        ]
        for block in candidates:
            for agent in self._task_agents(block):
                pending = str(agent.get("run_id", "")).startswith("pending:")
                if pending and name and str(agent.get("name", "")) == name:
                    return block
        for block in candidates:
            if any(
                str(agent.get("run_id", "")).startswith("pending:")
                for agent in self._task_agents(block)
            ):
                return block

        aggregate = self._parent_tasks.get(parent_id)
        if aggregate is None or aggregate.state is not BlockState.ACTIVE:
            aggregate = self._new_task_block(
                identifier=f"agents:{parent_id}",
                parent_id=parent_id,
            )
            self._parent_tasks[parent_id] = aggregate
        return aggregate

    def _update_task_agent(
        self,
        block: Block,
        snapshot: Mapping[str, Any],
        *,
        fallback_status: str | None = None,
    ) -> None:
        if block.state is not BlockState.ACTIVE:
            return
        value = dict(snapshot)
        run_id = str(value.get("run_id") or value.get("id") or "")
        name = str(value.get("name") or "")
        if run_id:
            value["run_id"] = run_id
        if value.get("agent_type") is None and value.get("type") is not None:
            value["agent_type"] = value["type"]

        agents = self._task_agents(block)
        requested_index = value.pop("index", None)
        index = (
            requested_index
            if isinstance(requested_index, int) and 0 <= requested_index < len(agents)
            else None
        )
        if index is not None and run_id:
            duplicate = next(
                (
                    position
                    for position, agent in enumerate(agents)
                    if position != index
                    and str(agent.get("run_id") or agent.get("id") or "") == run_id
                ),
                None,
            )
            if duplicate is not None:
                tasks = block.data.get("args", {}).get("tasks", [])
                if (
                    isinstance(tasks, Sequence)
                    and not isinstance(tasks, (str, bytes))
                    and duplicate < len(tasks)
                ):
                    original = tasks[duplicate]
                    item = (
                        dict(original)
                        if isinstance(original, Mapping)
                        else {"task": str(original)}
                    )
                    restored = {
                        "run_id": f"pending:{block.data.get('id', block.id)}:{duplicate}",
                        "name": item.get("name")
                        or item.get("agent")
                        or f"agent {duplicate + 1}",
                        "agent_type": item.get("agent") or "task",
                        "description": item.get("task") or "",
                        "status": "pending",
                    }
                else:
                    restored = dict(agents[duplicate])
                    restored["run_id"] = (
                        f"pending:{block.data.get('id', block.id)}:{duplicate}"
                    )
                    restored["status"] = "pending"
                    for key in ("result", "delivered", "reason"):
                        restored.pop(key, None)
                agents[duplicate] = restored
        if index is None:
            index = next(
                (
                    position
                    for position, agent in enumerate(agents)
                    if str(agent.get("run_id", "")).startswith("pending:")
                    and str(agent.get("name") or "") == name
                ),
                None,
            )
        if index is None:
            index = next(
                (
                    position
                    for position, agent in enumerate(agents)
                    if str(agent.get("run_id", "")).startswith("pending:")
                ),
                None,
            )
        if index is None:
            if fallback_status and not value.get("status"):
                value["status"] = fallback_status
            agents.append(value)
        else:
            current = agents[index]
            if fallback_status and not value.get("status") and not current.get("status"):
                value["status"] = fallback_status
            agents[index] = {**current, **value}
        block.update(agents=agents)
        if run_id:
            self._agent_tasks[run_id] = block

    def _merge_task_result(self, block: Block, result: Any) -> None:
        if not isinstance(result, Mapping):
            return
        errors = result.get("errors")
        error_indexes = {
            error.get("index")
            for error in errors
            if isinstance(error, Mapping) and isinstance(error.get("index"), int)
        } if isinstance(errors, Sequence) and not isinstance(errors, (str, bytes)) else set()
        success_indexes = iter(
            index for index in range(len(self._task_agents(block))) if index not in error_indexes
        )
        for key in ("spawned", "agents", "tasks", "jobs", "results"):
            snapshots = result.get(key)
            if not isinstance(snapshots, Sequence) or isinstance(snapshots, (str, bytes)):
                continue
            for snapshot in snapshots:
                if isinstance(snapshot, Mapping):
                    value = dict(snapshot)
                    if key == "spawned":
                        value["index"] = next(success_indexes, None)
                    self._update_task_agent(block, value)

        if isinstance(errors, Sequence) and not isinstance(errors, (str, bytes)):
            agents = self._task_agents(block)
            changed = False
            for error in errors:
                if not isinstance(error, Mapping):
                    continue
                index = error.get("index")
                if not isinstance(index, int) or not 0 <= index < len(agents):
                    continue
                agents[index] = {
                    **agents[index],
                    "status": "failed",
                    "result": {"error": error.get("error") or "Agent failed to start"},
                }
                changed = True
            if changed:
                block.update(agents=agents)

    def _settle_task_if_complete(self, block: Block) -> None:
        if block.state is not BlockState.ACTIVE or not block.data.get("tool_complete"):
            return
        agents = self._task_agents(block)
        terminal = {
            "aborted",
            "cancelled",
            "canceled",
            "done",
            "error",
            "failed",
            "success",
            "succeeded",
        }
        if not agents:
            if "result" not in block.data:
                return
        elif not all(
            str(agent.get("status", "")).casefold() in terminal
            and "result" in agent
            and (
                bool(agent.get("delivered"))
                or str(agent.get("run_id", "")).startswith("pending:")
            )
            for agent in agents
        ):
            return
        block.settle()
        if self.scheduler is not None:
            self.scheduler.request_commit()

    def _task_changed(self, block: Block) -> None:
        self._settle_task_if_complete(block)
        if self.scheduler is not None:
            if block.state is BlockState.ACTIVE:
                self.scheduler.start_spinner()
            self.scheduler.request_invalidate()

    def _agent_spawned(self, event: AgentSpawned) -> None:
        block = self._task_for_agent(event.parent_id, event.run_id, event.name)
        self._update_task_agent(
            block,
            {
                "run_id": event.run_id,
                "name": event.name,
                "agent_type": event.agent_type,
                "status": "pending",
            },
        )
        self._task_changed(block)

    def _agent_status(self, event: AgentStatus) -> None:
        block = self._task_for_agent(event.parent_id, event.run_id, event.name)
        self._update_task_agent(
            block,
            {
                "run_id": event.run_id,
                "name": event.name,
                "agent_type": event.agent_type,
                "status": event.status,
                "reason": event.reason,
            },
        )
        self._task_changed(block)

    def _agent_finished(self, event: AgentFinished) -> None:
        block = self._task_for_agent(event.parent_id, event.run_id, event.name)
        self._update_task_agent(
            block,
            {
                "run_id": event.run_id,
                "name": event.name,
                "agent_type": event.agent_type,
                "result": event.result,
            },
            fallback_status="done",
        )
        self._task_changed(block)

    def _agent_delivered(self, event: AgentDelivered) -> None:
        for index, value in enumerate(event.jobs):
            if not isinstance(value, Mapping):
                continue
            job = dict(value)
            run_id = str(job.get("run_id") or job.get("id") or "")
            task = self._agent_tasks.get(run_id)
            if task is not None and task.state is BlockState.ACTIVE:
                self._update_task_agent(task, job)
                self._settle_task_if_complete(task)
            if run_id and run_id in self._deliveries:
                continue
            block = self.frame.add(
                "delivery",
                {"job": job},
                source_id=event.parent_id,
                block_id=f"delivery:{run_id}" if run_id else None,
            )
            if run_id:
                self._deliveries[run_id] = block
            self._commit(block, immediate=index == len(event.jobs) - 1)
        if self.scheduler is not None:
            self.scheduler.request_invalidate()

    def _tool_start(self, event: ToolCallStart) -> None:
        source = str(event.source_id or "main")
        if event.name == "task":
            block = self._new_task_block(
                identifier=event.id,
                parent_id=source,
                args=event.args,
                agents=self._task_placeholders(event),
                tool_complete=False,
            )
            self._tools[event.id] = block
            self._task_changed(block)
            return

        block = self._read_groups.get(source) if event.name == "read_file" else None
        can_group = (
            block is not None
            and block.state is BlockState.ACTIVE
            and bool(self.frame.blocks)
            and self.frame.blocks[-1] is block
        )
        if can_group:
            calls = block.data.get("calls")
            if not isinstance(calls, list):
                calls = [
                    {
                        "id": block.data["id"],
                        "args": block.data.get("args", {}),
                    }
                ]
            block.update(
                calls=[
                    *calls,
                    {"id": event.id, "args": event.args},
                ]
            )
        else:
            block = self.frame.add(
                "tool",
                {"name": event.name, "args": event.args, "id": event.id},
                source_id=event.source_id,
            )
            if event.name == "read_file":
                self._read_groups[source] = block
        self._tools[event.id] = block
        if self.scheduler is not None:
            self.scheduler.start_spinner()
            self.scheduler.request_invalidate()

    def _tool_end(self, event: ToolCallEnd) -> None:
        block = self._tools.get(event.id)
        if block is None:
            if event.name == "task":
                block = self._new_task_block(
                    identifier=event.id,
                    parent_id="main",
                )
            else:
                block = self.frame.add(
                    "tool",
                    {"name": event.name, "args": {}, "id": event.id},
                )
            self._tools[event.id] = block
        if block.kind == "task":
            block.update(result=event.result, tool_complete=True)
            self._merge_task_result(block, event.result)
            self._task_changed(block)
            return

        calls = block.data.get("calls")
        if isinstance(calls, list):
            updated = [
                {**call, "result": event.result}
                if call.get("id") == event.id
                else call
                for call in calls
            ]
            block.update(calls=updated)
            complete = all("result" in call for call in updated)
        else:
            block.update(result=event.result)
            complete = True
        if complete:
            block.settle()
            self._float_active_tasks()
            if self.scheduler is not None:
                self.scheduler.request_commit()
        if self.scheduler is not None:
            self.scheduler.request_invalidate()

    def _source_block(self, source_id: str, kind: str, role: str) -> Block:
        key = (source_id, kind)
        block = self._source_blocks.get(key)
        if block is not None:
            return block
        data: dict[str, Any] = {"text": "", "role": role}
        if kind == "assistant":
            data.update(role=role, subagent=role == "subagent" or role.startswith("subagent:"))
        block = self.frame.add(kind, data, source_id=source_id)
        if kind == "thinking":
            assistant = self._source_blocks.get((source_id, "assistant"))
            if assistant is not None:
                self.frame.blocks.remove(block)
                self.frame.blocks.insert(self.frame.blocks.index(assistant), block)
        self._source_blocks[key] = block
        return block

    def _model_chunk(self, event: ModelChunk) -> None:
        source_id = str(event.source_id or event.role)
        value = getattr(event.chunk, "content", event.chunk)
        parts = value if isinstance(value, (list, tuple)) else (value,)
        thinking_seen = False
        usage_tokens = _reasoning_tokens(event.chunk)
        for part in parts:
            if isinstance(part, Mapping) and part.get("type") in {"reasoning", "thinking"}:
                content = _thinking_text(part)
                kind = "thinking"
            else:
                content = _text(part)
                kind = "assistant"
            if not content:
                continue
            block = self._source_block(source_id, kind, event.role)
            accumulated = f"{block.data['text']}{content}"
            changes: dict[str, Any] = {"text": accumulated}
            if kind == "thinking":
                thinking_seen = True
                changes["reasoning_tokens"] = (
                    usage_tokens
                    if usage_tokens is not None
                    else max(1, (len(accumulated) + 3) // 4)
                )
            block.update(changes)
        if usage_tokens is not None:
            thinking = self._source_blocks.get((source_id, "thinking"))
            if (
                thinking is not None
                and thinking.data.get("reasoning_tokens") != usage_tokens
            ):
                thinking.update(reasoning_tokens=usage_tokens)
            thinking_seen = thinking is not None
        if self.scheduler is not None:
            if thinking_seen:
                self.scheduler.start_spinner()
            self.scheduler.request_invalidate()

    def print(self, *objects: Any, **kwargs: Any) -> Block:
        block = self.frame.add(
            "raw",
            {
                "objects": tuple(objects),
                "options": dict(kwargs),
                "level": "raw",
            },
        )
        self._commit(block)
        return block


    def release_committed(self, blocks: list[Block]) -> None:
        """Release accumulator references after scrollback owns the blocks."""

        committed = {
            block.id
            for block in blocks
            if block.state is BlockState.COMMITTED
        }
        if not committed:
            return
        self._source_blocks = {
            key: block
            for key, block in self._source_blocks.items()
            if block.id not in committed
        }
        self._tools = {
            key: block
            for key, block in self._tools.items()
            if block.id not in committed
        }
        self._read_groups = {
            key: block
            for key, block in self._read_groups.items()
            if block.id not in committed
        }
        self._task_blocks = {
            key: block
            for key, block in self._task_blocks.items()
            if block.id not in committed
        }
        self._agent_tasks = {
            key: block
            for key, block in self._agent_tasks.items()
            if block.id not in committed
        }
        self._parent_tasks = {
            key: block
            for key, block in self._parent_tasks.items()
            if block.id not in committed
        }
        self._deliveries = {
            key: block
            for key, block in self._deliveries.items()
            if block.id not in committed
        }
    def clear(self) -> None:
        self.frame.blocks.clear()
        self._source_blocks.clear()
        self._tools.clear()
        self._read_groups.clear()
        self._task_blocks.clear()
        self._agent_tasks.clear()
        self._parent_tasks.clear()
        self._deliveries.clear()


__all__ = ["Transcript"]
