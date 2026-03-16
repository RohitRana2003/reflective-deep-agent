"""Deep Agents middleware integration for Plan Once workers.

Bridges the ``deepagents`` SDK middleware stack into worker sub-agents so
they gain filesystem tools, todo planning, summarization, memory, and
skill loading — without hand-rolling ``@tool`` functions.

Usage::

    from plan_once.deep_features import build_worker_middleware

    mw = build_worker_middleware(model, enable_filesystem=True)
    worker = create_agent(model, tools=[], middleware=mw)
"""

from __future__ import annotations

from typing import Any

from langchain_core.language_models import BaseChatModel


def build_worker_middleware(
    model: BaseChatModel,
    *,
    backend: Any | None = None,
    enable_filesystem: bool = True,
    enable_todos: bool = True,
    enable_summarization: bool = True,
    enable_subagents: bool = False,
    enable_memory: bool = False,
    memory_sources: list[str] | None = None,
    enable_skills: bool = False,
    skill_sources: list[str] | None = None,
) -> list[Any]:
    """Build a Deep Agents middleware stack for worker sub-agents.

    Each enabled middleware is appended in the recommended order (the same
    order used by ``create_deep_agent`` internally).

    Args:
        model: LLM used by the worker (needed by summarization middleware).
        backend: A ``BackendProtocol`` instance or factory.  Defaults to
            ``StateBackend`` (ephemeral, in-memory per worker).
        enable_filesystem: Give workers ``ls``, ``read_file``, ``write_file``,
            ``edit_file``, ``glob``, ``grep`` tools automatically.
        enable_todos: Give workers a ``write_todos`` tool so they can plan
            their own sub-tasks internally.
        enable_summarization: Auto-summarise long worker conversations to
            prevent context overflow.
        enable_subagents: Give workers a ``task`` tool so they can spawn
            their own sub-sub-agents.  Disabled by default to avoid deep
            nesting.
        enable_memory: Load ``AGENTS.md`` files into the worker system
            prompt for project-specific context.
        memory_sources: Paths to memory files (e.g.
            ``["/memory/AGENTS.md"]``).  Required when *enable_memory* is
            ``True``.
        enable_skills: Load skill definitions into the worker system prompt
            for specialised behaviour.
        skill_sources: Paths to skill directories (e.g.
            ``["/skills/user/"]``).  Required when *enable_skills* is
            ``True``.

    Returns:
        A list of ``AgentMiddleware`` instances ready to pass to
        ``create_agent(…, middleware=…)``.
    """
    # Lazy imports so the module is importable even when deepagents is not
    # installed (the graph/workers modules still work without it).
    from langchain.agents.middleware import TodoListMiddleware  # type: ignore[import-untyped]

    from deepagents.backends import StateBackend
    from deepagents.middleware.filesystem import FilesystemMiddleware
    from deepagents.middleware.summarization import create_summarization_middleware

    if backend is None:
        backend = StateBackend  # pass the *class* — acts as a factory

    mw: list[Any] = []

    # --- 1. TodoListMiddleware (planning) ---
    if enable_todos:
        mw.append(TodoListMiddleware())

    # --- 2. MemoryMiddleware (AGENTS.md) ---
    if enable_memory:
        from deepagents.middleware.memory import MemoryMiddleware

        sources = memory_sources or []
        if sources:
            mw.append(MemoryMiddleware(backend=backend, sources=sources))

    # --- 3. SkillsMiddleware ---
    if enable_skills:
        from deepagents.middleware.skills import SkillsMiddleware

        sources = skill_sources or []
        if sources:
            mw.append(SkillsMiddleware(backend=backend, sources=sources))

    # --- 4. FilesystemMiddleware (ls, read_file, write_file, …) ---
    if enable_filesystem:
        mw.append(FilesystemMiddleware(backend=backend))

    # --- 5. SubAgentMiddleware (task tool) ---
    if enable_subagents:
        from deepagents.middleware.subagents import (
            GENERAL_PURPOSE_SUBAGENT,
            SubAgentMiddleware,
        )

        # Give the worker a general-purpose sub-sub-agent by default.
        gp_spec: dict[str, Any] = {
            **GENERAL_PURPOSE_SUBAGENT,
            "model": model,
            "tools": [],
            "middleware": [
                TodoListMiddleware(),
                FilesystemMiddleware(backend=backend),
            ],
        }
        mw.append(
            SubAgentMiddleware(
                backend=backend,
                subagents=[gp_spec],
            )
        )

    # --- 6. SummarizationMiddleware ---
    if enable_summarization:
        mw.append(create_summarization_middleware(model, backend))

    # --- 7. PatchToolCallsMiddleware (fixes malformed tool calls) ---
    try:
        from deepagents.middleware.patch_tool_calls import PatchToolCallsMiddleware

        mw.append(PatchToolCallsMiddleware())
    except ImportError:
        pass

    return mw
