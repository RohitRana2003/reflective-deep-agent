"""Plan Once, Execute Once — Worker tool factory.

Provides ``build_worker_tool``, which creates a LangChain tool that spins up
an independent sub-agent (worker) to execute a single focused task.  This
mirrors the ``SubAgentMiddleware`` / ``task`` pattern in Deep Agents but is
designed to be used inside a non-looping DAG.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Sequence
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.tools import BaseTool, StructuredTool

from plan_once.token_tracker import get_tracker

_agent_logger = logging.getLogger("plan_once.workers")

# Default system prompt given to every worker unless overridden.
_DEFAULT_WORKER_PROMPT = (
    "You are a capable Worker Agent. You receive a concrete task and must "
    "complete it using any tools available to you. You can:\n"
    "- Read, write, edit, and create files using filesystem tools\n"
    "- Analyze data, generate content, write code\n"
    "- Plan multi-step operations\n"
    "- Produce clear, well-structured output\n\n"
    "Always complete the task fully and return a clear summary of what "
    "you did and any results."
)


def build_worker_tool(
    model: BaseChatModel,
    *,
    worker_tools: Sequence[BaseTool | Callable | dict[str, Any]] | None = None,
    worker_prompt: str = _DEFAULT_WORKER_PROMPT,
    name: str = "delegate_to_worker",
    middleware: Sequence[Any] | None = None,
    recursion_limit: int = 50,
) -> BaseTool:
    """Create a tool that delegates work to an isolated worker sub-agent.

    Each invocation spins up a fresh ``create_agent`` graph with its own
    context window, executes the task, and returns only the final text output.
    This keeps the orchestrator's context lean.

    Args:
        model: The LLM backing each worker instance.
        worker_tools: Tools the worker agent can use (filesystem, search, etc.).
            If ``None``, the worker runs tool-less (useful for pure reasoning).
        worker_prompt: System prompt injected into every worker invocation.
        name: Name of the returned tool visible to the orchestrator.
        middleware: Optional list of ``AgentMiddleware`` instances (from Deep
            Agents) to attach to each worker.  Use
            ``build_worker_middleware()`` from ``plan_once.deep_features`` to
            construct the stack.
        recursion_limit: Maximum number of ReAct loop iterations for each
            worker sub-agent.  Defaults to ``50``, which is enough for
            multi-step file operations (find → read → edit → verify).
            Direct-mode workers don't use this at all.

    Returns:
        A ``StructuredTool`` that the orchestrator can call (potentially in
        parallel) to delegate individual steps of its plan.
    """
    resolved_tools: list[BaseTool | Callable | dict[str, Any]] = list(worker_tools or [])
    resolved_middleware: list[Any] = list(middleware or [])

    model_name = getattr(model, "model", None) or getattr(model, "model_name", "unknown")

    def _delegate(task_description: str, mode: str = "auto") -> str:  # noqa: D401
        """Execute a single, focused step of a larger plan.

        Use this tool to delegate work to an independent worker agent.
        Each worker has its own isolated context window.

        Args:
            task_description: Detailed task for the worker. Include file
                contents directly when available so the worker doesn't
                need to read files itself.
            mode: Execution mode.
                "direct" — single LLM call, no tools, fastest. Best when
                file contents are already in task_description.
                "agent" — full ReAct agent with filesystem tools. Best
                for large files the worker needs to read itself.
                "auto" — decide automatically based on task length.
        """
        import time as _time  # noqa: PLC0415

        effective_mode = mode
        if effective_mode == "auto":
            # Only use direct when file contents are auto-injected (large task).
            # Short task = worker must use tools to find/read files itself.
            effective_mode = "direct" if len(task_description) > 500 else "agent"

        # Safety override: if the orchestrator said 'direct' but no file contents
        # are embedded in the task, the worker will have nothing to work with.
        # Force 'agent' so it can use filesystem tools (grep, read_file, ls).
        mode_was_overridden = False
        if effective_mode == "direct" and "FILE CONTENTS" not in task_description:
            effective_mode = "agent"
            mode_was_overridden = True

        if effective_mode == "direct":
            # Direct LLM call — no tools, no ReAct loop, no middleware.
            # Single call, maximum efficiency.
            #
            # CRITICAL: prepend a direct-mode clarification so the LLM
            # doesn't output tool-call syntax as plain text (it has no
            # tools in this mode — it must analyze the embedded data).
            direct_prefix = (
                "DIRECT MODE — you have NO tools available.\n"
                "The file contents are ALREADY EMBEDDED in the task below.\n"
                "Analyze them directly and write your answer as text.\n"
                "Do NOT output grep/read_file/ls commands — they will NOT "
                "execute. Work with the data you already have.\n\n"
            )
            input_msgs = [
                ("system", direct_prefix + worker_prompt),
                ("user", task_description),
            ]
            t0 = _time.time()
            response = model.invoke(input_msgs)
            duration = _time.time() - t0
            get_tracker().record(
                role="worker",
                response=response,
                model_name=str(model_name),
                duration_seconds=duration,
                worker_task=task_description,
                input_messages=input_msgs,
            )
            # Register a trace (no tool calls) for the reflection loop
            _register_direct_trace(
                task_description=task_description,
                result_text=str(response.content) if response.content else "",
                mode_was_overridden=mode_was_overridden,
            )
            return str(response.content) if response.content else ""

        # Agent mode — full ReAct loop with tools and optional middleware.
        from langchain.agents import create_agent  # noqa: PLC0415

        # When file contents are already embedded in the task description,
        # tell the worker NOT to re-read the same file — save tool calls
        # for write/edit operations only.
        effective_prompt = worker_prompt
        if "FILE CONTENTS (auto-injected)" in task_description:
            agent_embedded_prefix = (
                "EMBEDDED FILE DATA: The source file contents are ALREADY "
                "included below in your task under 'FILE CONTENTS (auto-injected)'.\n"
                "Do NOT call read_file or grep on the embedded file — the data "
                "is already in your context.\n"
                "Reserve your tool calls for write_file / edit_file operations ONLY.\n\n"
            )
            effective_prompt = agent_embedded_prefix + worker_prompt

        kwargs: dict[str, Any] = {"tools": resolved_tools}
        if resolved_middleware:
            kwargs["middleware"] = resolved_middleware
        worker = create_agent(model, **kwargs)
        t0 = _time.time()
        try:
            result = worker.invoke(
                {
                    "messages": [
                        ("system", effective_prompt),
                        ("user", task_description),
                    ],
                },
                config={"recursion_limit": recursion_limit},
            )
        except Exception as agent_exc:  # noqa: BLE001
            duration = _time.time() - t0
            # Try to extract partial results from the exception context
            _agent_logger.warning(
                "Worker agent hit error after %.1fs: %s",
                duration, type(agent_exc).__name__,
            )
            # Return a useful error message instead of crashing
            return (
                f"Worker completed partial work but hit a limit after "
                f"{duration:.0f}s: {type(agent_exc).__name__}. "
                f"The task may need to be broken into smaller steps or "
                f"the recursion_limit increased."
            )
        duration = _time.time() - t0

        # Log every LLM turn inside the worker
        _record_worker_calls(
            result, model_name=str(model_name),
            task_description=task_description, total_duration=duration,
            mode_was_overridden=mode_was_overridden,
        )

        final_message = result["messages"][-1]
        return str(final_message.content) if final_message.content else ""

    async def _adelegate(task_description: str, mode: str = "auto") -> str:  # noqa: D401
        """Async variant: execute a single, focused step of a larger plan."""
        import time as _time  # noqa: PLC0415

        effective_mode = mode
        if effective_mode == "auto":
            effective_mode = "direct" if len(task_description) > 500 else "agent"

        mode_was_overridden = False
        if effective_mode == "direct" and "FILE CONTENTS" not in task_description:
            effective_mode = "agent"
            mode_was_overridden = True

        if effective_mode == "direct":
            direct_prefix = (
                "DIRECT MODE — you have NO tools available.\n"
                "The file contents are ALREADY EMBEDDED in the task below.\n"
                "Analyze them directly and write your answer as text.\n"
                "Do NOT output grep/read_file/ls commands — they will NOT "
                "execute. Work with the data you already have.\n\n"
            )
            input_msgs = [
                ("system", direct_prefix + worker_prompt),
                ("user", task_description),
            ]
            t0 = _time.time()
            response = await model.ainvoke(input_msgs)
            duration = _time.time() - t0
            get_tracker().record(
                role="worker",
                response=response,
                model_name=str(model_name),
                duration_seconds=duration,
                worker_task=task_description,
                input_messages=input_msgs,
            )
            _register_direct_trace(
                task_description=task_description,
                result_text=str(response.content) if response.content else "",
                mode_was_overridden=mode_was_overridden,
            )
            return str(response.content) if response.content else ""

        from langchain.agents import create_agent  # noqa: PLC0415

        # Same embedded-file preamble as sync path
        effective_prompt = worker_prompt
        if "FILE CONTENTS (auto-injected)" in task_description:
            agent_embedded_prefix = (
                "EMBEDDED FILE DATA: The source file contents are ALREADY "
                "included below in your task under 'FILE CONTENTS (auto-injected)'.\n"
                "Do NOT call read_file or grep on the embedded file — the data "
                "is already in your context.\n"
                "Reserve your tool calls for write_file / edit_file operations ONLY.\n\n"
            )
            effective_prompt = agent_embedded_prefix + worker_prompt

        kwargs: dict[str, Any] = {"tools": resolved_tools}
        if resolved_middleware:
            kwargs["middleware"] = resolved_middleware
        worker = create_agent(model, **kwargs)
        t0 = _time.time()
        try:
            result = await worker.ainvoke(
                {
                    "messages": [
                        ("system", effective_prompt),
                        ("user", task_description),
                    ],
                },
                config={"recursion_limit": recursion_limit},
            )
        except Exception as agent_exc:  # noqa: BLE001
            duration = _time.time() - t0
            _agent_logger.warning(
                "Worker agent hit error after %.1fs: %s",
                duration, type(agent_exc).__name__,
            )
            return (
                f"Worker completed partial work but hit a limit after "
                f"{duration:.0f}s: {type(agent_exc).__name__}. "
                f"The task may need to be broken into smaller steps or "
                f"the recursion_limit increased."
            )
        duration = _time.time() - t0

        _record_worker_calls(
            result, model_name=str(model_name),
            task_description=task_description, total_duration=duration,
            mode_was_overridden=mode_was_overridden,
        )

        final_message = result["messages"][-1]
        return str(final_message.content) if final_message.content else ""

    def _extract_tool_call_trace(
        messages: list[Any],
    ) -> list[Any]:   # list[ToolCallRecord]
        """Extract ordered ToolCallRecord list from a ReAct message sequence."""
        from langchain_core.messages import AIMessage, ToolMessage  # noqa: PLC0415
        from plan_once.reflector import ToolCallRecord  # noqa: PLC0415

        # Map tool_call_id → ToolMessage content
        tool_results: dict[str, str] = {}
        for msg in messages:
            if isinstance(msg, ToolMessage):
                tool_results[str(msg.tool_call_id)] = str(msg.content)[:400]

        records: list[ToolCallRecord] = []
        for msg in messages:
            if isinstance(msg, AIMessage):
                for tc in (getattr(msg, "tool_calls", []) or []):
                    tc_id   = tc.get("id", "") if isinstance(tc, dict) else getattr(tc, "id", "")
                    tc_name = tc.get("name", "?") if isinstance(tc, dict) else getattr(tc, "name", "?")
                    tc_args = tc.get("args", {}) if isinstance(tc, dict) else getattr(tc, "args", {})
                    records.append(ToolCallRecord(
                        name=str(tc_name),
                        args=dict(tc_args) if tc_args else {},
                        result=tool_results.get(str(tc_id), ""),
                    ))
        return records

    def _register_direct_trace(
        *,
        task_description: str,
        result_text: str,
        mode_was_overridden: bool,
    ) -> None:
        """Register a WorkerTrace for a direct-mode (no tool calls) worker."""
        from plan_once.reflector import WorkerTrace  # noqa: PLC0415

        trace = WorkerTrace(
            task_description=task_description,
            mode="direct",
            tool_calls=[],
            total_tokens=0,
            result=result_text[:400],
            mode_was_overridden=mode_was_overridden,
        )
        get_tracker().add_worker_trace(trace)

    def _record_worker_calls(
        result: dict[str, Any],
        *,
        model_name: str,
        task_description: str,
        total_duration: float,
        mode_was_overridden: bool = False,
    ) -> None:
        """Record every AIMessage in the worker result as a separate LLM call.

        Also builds a WorkerTrace (tool call sequence + token totals) and
        registers it with the tracker for the auto-reflection loop.
        """
        from langchain_core.messages import AIMessage  # noqa: PLC0415
        from plan_once.reflector import WorkerTrace  # noqa: PLC0415

        tracker = get_tracker()
        ai_msgs = [
            m for m in result.get("messages", [])
            if isinstance(m, AIMessage)
        ]
        if not ai_msgs:
            return
        # Distribute total wall-clock time proportionally across turns
        per_turn = total_duration / len(ai_msgs)
        total_tok = 0
        for msg in ai_msgs:
            rec = tracker.record(
                role="worker",
                response=msg,
                model_name=model_name,
                duration_seconds=per_turn,
                worker_task=task_description,
            )
            total_tok += rec.total_tokens

        # Build tool call trace and register for reflection loop
        all_messages = result.get("messages", [])
        tool_calls = _extract_tool_call_trace(all_messages)
        final_msg = ai_msgs[-1]
        result_text = str(getattr(final_msg, "content", "") or "")[:400]
        trace = WorkerTrace(
            task_description=task_description,
            mode="agent",
            tool_calls=tool_calls,
            total_tokens=total_tok,
            result=result_text,
            mode_was_overridden=mode_was_overridden,
        )
        tracker.add_worker_trace(trace)

    return StructuredTool.from_function(
        name=name,
        func=_delegate,
        coroutine=_adelegate,
        description=(
            "Delegate a task to an independent worker agent. Workers can "
            "read, write, edit, and create files, analyze data, generate "
            "content, and more.\n"
            "- mode='direct': Single LLM call, no tools (fastest — for "
            "  analysis when file contents are in the task_description).\n"
            "- mode='agent': Full agent with filesystem tools (read_file, "
            "  write_file, edit_file, ls, grep, glob). Use for ANY task "
            "  that involves creating, modifying, or browsing files.\n"
            "- mode='auto': Decides automatically based on task length."
        ),
    )
