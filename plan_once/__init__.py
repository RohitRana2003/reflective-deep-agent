"""Plan Once, Execute Once — General-Purpose Deep Agent Architecture.

A DAG-based orchestrator-worker pattern that eliminates recursion entirely:

1. **Triage Node** — Zero-cost Python: extracts file paths, detects intent
   (read/write/mixed), reads files, classifies complexity, recommends
   mode and features.
2. **Orchestrator Node** — Single LLM turn.  Reads triage assessment and
   delegates tasks via ``delegate_to_worker`` tool calls.
3. **Smart Worker Node** — Auto-injects file contents from triage into
   worker task descriptions.  ``mode='direct'`` = single LLM call (fast);
   ``mode='agent'`` = full ReAct loop with filesystem tools (write/edit).
4. **Synthesizer Node** — Only runs when multiple workers produced outputs.
   Skipped entirely for single-worker tasks.
5. **Reflect Node** *(optional)* — Runs after every execution when
   ``enable_reflection=True``.  Detects failure patterns in worker traces
   (over-exploration, duplicate calls, path-not-found errors) and writes
   new contrastive entries to ``reflections.yaml`` via one LLM call per
   failure.  Zero cost when no failures are detected.
6. **END** — The graph finishes.  No recursion possible.

Workers get the full Deep Agents middleware stack when enabled:
  - FilesystemMiddleware (ls, read_file, write_file, edit_file, glob, grep)
  - TodoListMiddleware (write_todos planning)
  - SummarizationMiddleware (auto-summarize long conversations)
  - MemoryMiddleware (AGENTS.md context loading)
  - SkillsMiddleware (SKILL.md specialized behaviours)
  - SubAgentMiddleware (workers spawn sub-sub-agents)
  - PatchToolCallsMiddleware (fix malformed tool calls)
"""

from plan_once.graph import create_plan_once_agent
from plan_once.reflector import FailurePattern, WorkerTrace, run_reflection_pass
from plan_once.state import PlanOnceState
from plan_once.token_tracker import TokenTracker, get_tracker
from plan_once.workers import build_worker_tool

# Deep Agents integration (lazy — works even when deepagents is not installed)
try:
    from plan_once.deep_features import build_worker_middleware
except ImportError:  # pragma: no cover
    build_worker_middleware = None  # type: ignore[assignment,misc]

__all__ = [
    "FailurePattern",
    "PlanOnceState",
    "TokenTracker",
    "WorkerTrace",
    "build_worker_middleware",
    "build_worker_tool",
    "create_plan_once_agent",
    "get_tracker",
    "run_reflection_pass",
]
