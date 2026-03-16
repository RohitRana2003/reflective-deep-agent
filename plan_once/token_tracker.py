"""Plan Once, Execute Once — LLM call logging and token tracking.

Provides a global ``TokenTracker`` that records every LLM invocation across
the orchestrator, workers, and synthesizer.  After a run completes, call
``tracker.summary()`` to get a detailed breakdown of calls and tokens used.

Usage::

    from plan_once.token_tracker import get_tracker

    tracker = get_tracker()
    tracker.reset()

    # ... run the agent ...

    print(tracker.summary())
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger("plan_once")


@dataclass
class LLMCallRecord:
    """A single recorded LLM invocation."""

    call_id: int
    role: str  # "orchestrator", "worker", "synthesizer"
    worker_task: str | None  # task description (workers only)
    model_name: str
    input_tokens: int
    output_tokens: int
    total_tokens: int
    tool_calls_emitted: int  # number of tool calls in the response
    duration_seconds: float
    timestamp: float = field(default_factory=time.time)
    input_preview: str = ""   # truncated view of what went IN
    output_preview: str = ""  # truncated view of what came OUT


class TokenTracker:
    """Thread-safe tracker for all LLM calls in a Plan Once run.

    Collects ``LLMCallRecord`` entries from orchestrator, worker, and
    synthesizer nodes.  Provides ``summary()`` for a human-readable
    breakdown and ``totals()`` for programmatic access.

    Also stores ``WorkerTrace`` objects (captured by workers.py) that
    the auto-reflection loop reads to detect failure patterns.
    """

    def __init__(self) -> None:
        self._records: list[LLMCallRecord] = []
        self._lock = threading.Lock()
        self._call_counter = 0
        self._worker_traces: list[Any] = []   # list[WorkerTrace] (avoid circular import)

    # ------------------------------------------------------------------
    # Recording
    # ------------------------------------------------------------------
    def record(
        self,
        *,
        role: str,
        response: Any,
        model_name: str = "unknown",
        duration_seconds: float = 0.0,
        worker_task: str | None = None,
        input_messages: list[Any] | None = None,
    ) -> LLMCallRecord:
        """Record an LLM call and extract token usage from the response.

        Args:
            role: One of ``"orchestrator"``, ``"worker"``, ``"synthesizer"``.
            response: The ``AIMessage`` returned by the LLM.
            model_name: Model identifier string.
            duration_seconds: Wall-clock time for the call.
            worker_task: Task description (only for worker calls).
            input_messages: Optional list of input messages for verbose
                logging.  When provided, a preview of the input content
                is stored alongside the token counts.

        Returns:
            The created ``LLMCallRecord``.
        """
        # Extract token usage from response_metadata (works with ChatOllama,
        # ChatAnthropic, ChatOpenAI, etc.)
        meta = getattr(response, "response_metadata", {}) or {}
        usage = meta.get("usage", meta.get("token_usage", {})) or {}

        # ChatOllama puts tokens at top-level of response_metadata
        input_tokens = (
            usage.get("prompt_tokens", 0)
            or usage.get("input_tokens", 0)
            or meta.get("prompt_eval_count", 0)
        )
        output_tokens = (
            usage.get("completion_tokens", 0)
            or usage.get("output_tokens", 0)
            or meta.get("eval_count", 0)
        )
        total_tokens = (
            usage.get("total_tokens", 0)
            or (input_tokens + output_tokens)
        )

        # Count tool calls
        tool_calls = getattr(response, "tool_calls", []) or []
        num_tool_calls = len(tool_calls)

        # Build I/O previews for verbose logging
        input_preview = ""
        if input_messages:
            parts: list[str] = []
            for m in input_messages:
                if isinstance(m, tuple) and len(m) == 2:
                    role_label, text = m
                    parts.append(f"[{role_label}] {str(text)[:200]}")
                elif isinstance(m, dict):
                    parts.append(f"[{m.get('role', '?')}] {str(m.get('content', ''))[:200]}")
                elif hasattr(m, "content"):
                    role_label = getattr(m, "type", getattr(m, "role", "?"))
                    parts.append(f"[{role_label}] {str(m.content)[:200]}")
            input_preview = " | ".join(parts)
            if len(input_preview) > 500:
                input_preview = input_preview[:500] + "..."

        output_content = str(getattr(response, "content", "") or "")
        output_preview = output_content[:500]
        if len(output_content) > 500:
            output_preview += "..."

        # Add tool call info to output preview
        if num_tool_calls > 0:
            tc_names = [tc.get("name", "?") if isinstance(tc, dict) else getattr(tc, "name", "?") for tc in tool_calls]
            output_preview = f"[TOOL CALLS: {', '.join(tc_names)}] {output_preview}"

        with self._lock:
            self._call_counter += 1
            rec = LLMCallRecord(
                call_id=self._call_counter,
                role=role,
                worker_task=worker_task,
                model_name=model_name,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                total_tokens=total_tokens,
                tool_calls_emitted=num_tool_calls,
                duration_seconds=round(duration_seconds, 3),
                input_preview=input_preview,
                output_preview=output_preview,
            )
            self._records.append(rec)

        # Log immediately
        task_info = f" | task: {worker_task[:60]}..." if worker_task and len(worker_task) > 60 else (f" | task: {worker_task}" if worker_task else "")
        logger.info(
            "[LLM Call #%d] %-13s | model=%-30s | in=%5d out=%5d total=%5d | tools=%d | %.2fs%s",
            rec.call_id,
            rec.role.upper(),
            rec.model_name,
            rec.input_tokens,
            rec.output_tokens,
            rec.total_tokens,
            rec.tool_calls_emitted,
            rec.duration_seconds,
            task_info,
        )

        return rec

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------
    @property
    def records(self) -> list[LLMCallRecord]:
        """Return a copy of all recorded calls."""
        with self._lock:
            return list(self._records)

    def totals(self) -> dict[str, Any]:
        """Return aggregate token counts grouped by role.

        Returns:
            Dict with keys ``"by_role"`` (per-role breakdown),
            ``"total_calls"``, ``"total_input_tokens"``,
            ``"total_output_tokens"``, ``"total_tokens"``,
            ``"total_duration_seconds"``.
        """
        with self._lock:
            records = list(self._records)

        by_role: dict[str, dict[str, int | float]] = {}
        total_in = total_out = total_tok = 0
        total_dur = 0.0
        for r in records:
            role_stats = by_role.setdefault(r.role, {
                "calls": 0, "input_tokens": 0, "output_tokens": 0,
                "total_tokens": 0, "duration_seconds": 0.0,
            })
            role_stats["calls"] += 1  # type: ignore[operator]
            role_stats["input_tokens"] += r.input_tokens  # type: ignore[operator]
            role_stats["output_tokens"] += r.output_tokens  # type: ignore[operator]
            role_stats["total_tokens"] += r.total_tokens  # type: ignore[operator]
            role_stats["duration_seconds"] += r.duration_seconds  # type: ignore[operator]
            total_in += r.input_tokens
            total_out += r.output_tokens
            total_tok += r.total_tokens
            total_dur += r.duration_seconds

        return {
            "by_role": by_role,
            "total_calls": len(records),
            "total_input_tokens": total_in,
            "total_output_tokens": total_out,
            "total_tokens": total_tok,
            "total_duration_seconds": round(total_dur, 3),
        }

    def summary(self) -> str:
        """Return a human-readable summary of all LLM calls and tokens.

        Example output::

            ═══════════════════════════════════════════════════════════
            LLM CALL LOG  (7 calls)
            ═══════════════════════════════════════════════════════════
            #1  ORCHESTRATOR   | qwen3-coder:480b  | in=  320 out=  185 total=  505 | tools=5 |  2.41s
            #2  WORKER         | qwen3-coder:480b  | in=  410 out=   92 total=  502 | tools=0 |  1.83s | task: Count entries by log level...
            ...
            ───────────────────────────────────────────────────────────
            TOKEN USAGE SUMMARY
            ───────────────────────────────────────────────────────────
              orchestrator :   1 calls |   320 in |   185 out |   505 total |  2.41s
              worker       :   5 calls |  2050 in |   460 out |  2510 total |  9.15s
              synthesizer  :   1 calls |  3000 in |   500 out |  3500 total |  3.20s
            ───────────────────────────────────────────────────────────
              TOTAL        :   7 calls |  5370 in |  1145 out |  6515 total | 14.76s
            ═══════════════════════════════════════════════════════════
        """
        records = self.records
        totals = self.totals()

        if not records:
            return "No LLM calls recorded."

        lines: list[str] = []
        sep_thick = "=" * 72
        sep_thin = "-" * 72

        lines.append(sep_thick)
        lines.append(f"LLM CALL LOG  ({len(records)} call{'s' if len(records) != 1 else ''})")
        lines.append(sep_thick)

        for r in records:
            task_part = ""
            if r.worker_task:
                task_text = r.worker_task[:50] + "..." if len(r.worker_task) > 50 else r.worker_task
                task_part = f" | task: {task_text}"
            lines.append(
                f"  #{r.call_id:<3d} {r.role.upper():<14s} | {r.model_name:<28s} "
                f"| in={r.input_tokens:>5d} out={r.output_tokens:>5d} total={r.total_tokens:>5d} "
                f"| tools={r.tool_calls_emitted} | {r.duration_seconds:>6.2f}s"
                f"{task_part}"
            )

        lines.append(sep_thin)
        lines.append("TOKEN USAGE SUMMARY")
        lines.append(sep_thin)

        by_role = totals["by_role"]
        for role_name in ["orchestrator", "worker", "synthesizer"]:
            if role_name in by_role:
                s = by_role[role_name]
                lines.append(
                    f"  {role_name:<14s} : {s['calls']:>3d} calls "
                    f"| {s['input_tokens']:>6d} in "
                    f"| {s['output_tokens']:>6d} out "
                    f"| {s['total_tokens']:>6d} total "
                    f"| {s['duration_seconds']:>7.2f}s"
                )
        # Any roles not in the standard set
        for role_name, s in by_role.items():
            if role_name not in ("orchestrator", "worker", "synthesizer"):
                lines.append(
                    f"  {role_name:<14s} : {s['calls']:>3d} calls "
                    f"| {s['input_tokens']:>6d} in "
                    f"| {s['output_tokens']:>6d} out "
                    f"| {s['total_tokens']:>6d} total "
                    f"| {s['duration_seconds']:>7.2f}s"
                )

        lines.append(sep_thin)
        lines.append(
            f"  {'TOTAL':<14s} : {totals['total_calls']:>3d} calls "
            f"| {totals['total_input_tokens']:>6d} in "
            f"| {totals['total_output_tokens']:>6d} out "
            f"| {totals['total_tokens']:>6d} total "
            f"| {totals['total_duration_seconds']:>7.2f}s"
        )
        lines.append(sep_thick)

        return "\n".join(lines)

    def verbose_log(self) -> str:
        """Return a detailed log showing input/output content for each call.

        Unlike ``summary()`` which only shows token counts, this method
        reveals *what* went into each LLM call and *what* came out,
        making it easy to debug and understand the pipeline.
        """
        records = self.records
        if not records:
            return "No LLM calls recorded."

        lines: list[str] = []
        sep = "=" * 80

        lines.append(sep)
        lines.append(f"VERBOSE I/O LOG  ({len(records)} call{'s' if len(records) != 1 else ''})")
        lines.append(sep)

        for r in records:
            lines.append("")
            task_part = ""
            if r.worker_task:
                task_text = r.worker_task[:80] + "..." if len(r.worker_task) > 80 else r.worker_task
                task_part = f"\n  Task: {task_text}"
            lines.append(
                f"── Call #{r.call_id} | {r.role.upper()} | "
                f"in={r.input_tokens} out={r.output_tokens} total={r.total_tokens} | "
                f"{r.duration_seconds:.2f}s{task_part}"
            )
            lines.append(f"  INPUT:  {r.input_preview[:300] if r.input_preview else '(not captured)'}")
            lines.append(f"  OUTPUT: {r.output_preview[:300] if r.output_preview else '(empty)'}")

        lines.append("")
        lines.append(sep)
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Worker traces (for auto-reflection loop)
    # ------------------------------------------------------------------
    def add_worker_trace(self, trace: Any) -> None:  # trace: WorkerTrace
        """Register a WorkerTrace produced by a worker invocation."""
        with self._lock:
            self._worker_traces.append(trace)

    @property
    def worker_traces(self) -> list[Any]:
        """Return a copy of all WorkerTrace objects recorded so far."""
        with self._lock:
            return list(self._worker_traces)

    def reset(self) -> None:
        """Clear all recorded calls and worker traces."""
        with self._lock:
            self._records.clear()
            self._call_counter = 0
            self._worker_traces.clear()


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------
_global_tracker = TokenTracker()


def get_tracker() -> TokenTracker:
    """Return the global ``TokenTracker`` singleton."""
    return _global_tracker
