"""Plan Once — Auto-Reflection Loop (ExACT-inspired).

After every agent run this module:
1. Analyses worker execution traces for failure patterns — ZERO LLM cost.
2. When failures are confirmed, generates a contrastive YAML entry with
   ONE LLM call (the same model used by the orchestrator).
3. Appends the entry to ``reflections.yaml`` so every future run avoids
   repeating the same mistake.

Failure patterns detected automatically (pure Python):
  over_exploration   — > threshold tool calls for a task that should be simple.
  duplicate_calls    — same tool + same args called ≥ 2 times.
  path_not_found     — a tool returned a file-not-found error.
  empty_search       — grep/search returned no results (wrong pattern or path).
  direct_mode_escape — mode was forced from 'direct' to 'agent' at runtime
                       because the orchestrator omitted file contents.

Each detected pattern triggers ONE LLM call to write the reflection text.
If no failure is detected the reflect node is a no-op (zero cost).

YAML schema (matches existing reflections.yaml):
    - task_type:  <category>
      expected:   "<what the worker assumed — 1 sentence>"
      actual:     "<what actually happened — 1 sentence>"
      reflection: >
        <corrected strategy — 2-4 prescriptive lines>
"""

from __future__ import annotations

import logging
import os
import re
import time
from dataclasses import dataclass, field
from typing import Any

import yaml

logger = logging.getLogger("plan_once.reflector")

# ──────────────────────────────────────────────────────────────────────────────
# Tool-call thresholds — more calls than this signals over-exploration
# ──────────────────────────────────────────────────────────────────────────────
_DIRECT_TOOL_THRESHOLD = 0   # direct workers should make ZERO tool calls
_AGENT_TOOL_THRESHOLD  = 5   # agent workers should stay ≤ 5


# ──────────────────────────────────────────────────────────────────────────────
# Data classes
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class ToolCallRecord:
    """One tool call made by a worker during its ReAct loop."""
    name: str
    args: dict
    result: str   # truncated tool output


@dataclass
class WorkerTrace:
    """Complete execution trace for a single worker invocation."""
    task_description: str
    mode: str                                       # 'direct' | 'agent'
    tool_calls: list[ToolCallRecord] = field(default_factory=list)
    total_tokens: int = 0
    result: str = ""
    mode_was_overridden: bool = False               # direct→agent safety flip


@dataclass
class FailurePattern:
    """A confirmed failure pattern extracted from a WorkerTrace."""
    pattern_type: str   # over_exploration | duplicate_calls | path_not_found | …
    task_type: str      # log-analysis | file-editing | code-generation | …
    expected: str       # what the worker assumed
    actual: str         # what actually happened
    evidence: str       # concrete snippet from the trace
    task_description: str = ""


# ──────────────────────────────────────────────────────────────────────────────
# Task-type inference
# ──────────────────────────────────────────────────────────────────────────────

def _infer_task_type(task_description: str, tool_names: list[str]) -> str:
    td = task_description.lower()
    if any(w in td for w in ("log", ".log", "error", "warn", "grep", "root cause")):
        return "log-analysis"
    if any(w in td for w in ("edit", "write", "append", "create", "modify", "patch")):
        return "file-editing"
    if any(w in td for w in ("python", "code", "script", "function", "class", "def ")):
        return "code-generation"
    if any(w in td for w in ("report", "markdown", "summary", "analyse", "analysis")):
        return "report-writing"
    if "grep" in tool_names or "read_file" in tool_names:
        return "log-analysis"
    return "general"


# ──────────────────────────────────────────────────────────────────────────────
# Pattern detection — pure Python, zero LLM cost
# ──────────────────────────────────────────────────────────────────────────────

def detect_failure_patterns(traces: list[WorkerTrace]) -> list[FailurePattern]:
    """Scan worker traces and return every detected failure pattern.

    No LLM calls are made here — this is a fast heuristic pass.
    Only patterns that are clearly problematic (not borderline) are reported
    so we don't flood ``reflections.yaml`` with noise.
    """
    patterns: list[FailurePattern] = []

    for trace in traces:
        tool_names = [tc.name for tc in trace.tool_calls]
        n_tools    = len(trace.tool_calls)
        task_type  = _infer_task_type(trace.task_description, tool_names)

        # ── 1. OVER-EXPLORATION ────────────────────────────────────────────
        threshold = _DIRECT_TOOL_THRESHOLD if trace.mode == "direct" else _AGENT_TOOL_THRESHOLD
        if n_tools > threshold:
            seq = " → ".join(tool_names)
            patterns.append(FailurePattern(
                pattern_type="over_exploration",
                task_type=task_type,
                expected=(
                    f"Task should complete in ≤{threshold} tool call(s) "
                    f"(was mode='{trace.mode}')"
                ),
                actual=(
                    f"Worker made {n_tools} tool calls: {seq[:200]}"
                ),
                evidence=f"Full tool sequence: {seq}",
                task_description=trace.task_description,
            ))

        # ── 2. DUPLICATE TOOL CALLS ────────────────────────────────────────
        seen: dict[str, int] = {}
        for tc in trace.tool_calls:
            key = f"{tc.name}::{tuple(sorted(tc.args.items()))}"
            seen[key] = seen.get(key, 0) + 1
        dupes = {k: v for k, v in seen.items() if v > 1}
        if dupes:
            dupe_summary = "; ".join(
                f"{k.split('::')[0]} × {v}" for k, v in dupes.items()
            )
            patterns.append(FailurePattern(
                pattern_type="duplicate_calls",
                task_type=task_type,
                expected="Each tool + argument combination is called at most once",
                actual=f"Duplicate calls detected: {dupe_summary}",
                evidence=dupe_summary,
                task_description=trace.task_description,
            ))

        # ── 3. PATH NOT FOUND ──────────────────────────────────────────────
        _not_found_phrases = (
            "not found", "no such file", "filenotfounderror",
            "no matches found", "does not exist", "path does not",
            "cannot find", "unable to find",
        )
        bad_calls = [
            tc for tc in trace.tool_calls
            if any(ph in tc.result.lower() for ph in _not_found_phrases)
        ]
        if bad_calls:
            ex = bad_calls[0]
            patterns.append(FailurePattern(
                pattern_type="path_not_found",
                task_type=task_type,
                expected=f"'{ex.name}' would succeed with the provided path",
                actual=f"'{ex.name}' returned file-not-found: {ex.result[:150]}",
                evidence=f"tool={ex.name} args={ex.args}",
                task_description=trace.task_description,
            ))

        # ── 4. EMPTY SEARCH RESULTS ────────────────────────────────────────
        _empty_results = ("", "[]", "no matches", "0 results", "none found")
        empty_searches = [
            tc for tc in trace.tool_calls
            if tc.name in ("grep", "search")
            and tc.result.strip().lower() in _empty_results
        ]
        if empty_searches:
            eg = empty_searches[0]
            patterns.append(FailurePattern(
                pattern_type="empty_search",
                task_type=task_type,
                expected=(
                    f"grep pattern '{eg.args.get('pattern', '?')}' "
                    f"would return matching lines"
                ),
                actual=f"grep returned empty results for args: {eg.args}",
                evidence=f"grep args={eg.args}, result='{eg.result[:80]}'",
                task_description=trace.task_description,
            ))

        # ── 5. DIRECT-MODE SAFETY OVERRIDE ────────────────────────────────
        if trace.mode_was_overridden:
            patterns.append(FailurePattern(
                pattern_type="direct_mode_escape",
                task_type=task_type,
                expected=(
                    "Orchestrator used mode='direct' expecting file contents "
                    "to be embedded in the task"
                ),
                actual=(
                    "No 'FILE CONTENTS' block was present — mode was forced "
                    "to 'agent' so the worker could use filesystem tools"
                ),
                evidence=(
                    f"task prefix: {trace.task_description[:200]}"
                ),
                task_description=trace.task_description,
            ))

    return patterns


# ──────────────────────────────────────────────────────────────────────────────
# LLM-based reflection generation
# ──────────────────────────────────────────────────────────────────────────────

_REFLECTION_PROMPT = """\
You are writing an entry for a YAML contrastive-reflection memory file.
A worker agent made a mistake. Study the failure and produce an entry in
EXACTLY the YAML format shown below. Output ONLY the raw YAML — no markdown
fences, no extra text.

Required format:
  - task_type: {task_type}
    expected: "<one sentence — what the worker assumed would work>"
    actual: "<one sentence — what actually happened / the mistake>"
    reflection: >
      <2-4 lines: prescriptive corrected strategy, starts with an action verb>

Failure details
───────────────
Pattern  : {pattern_type}
Task desc: {task_description}
Expected : {expected}
Actual   : {actual}
Evidence : {evidence}

Rules
─────
• task_type MUST be one of: log-analysis, file-editing, code-generation,
  report-writing, general
• expected / actual: single sentence each, no line breaks
• reflection: prescriptive ("Always X before Y", "Use grep -c instead of …")
• Do NOT include YAML document separators (---)
• Output ONLY the YAML block
"""


def generate_reflection_entry(
    model: Any,
    pattern: FailurePattern,
) -> str | None:
    """Call the LLM once to produce a YAML reflection entry.

    Returns the raw YAML string ready to be appended to ``reflections.yaml``,
    or ``None`` if the call fails or produces unparseable output.
    """
    prompt = _REFLECTION_PROMPT.format(
        task_type=pattern.task_type,
        pattern_type=pattern.pattern_type,
        task_description=pattern.task_description[:300],
        expected=pattern.expected,
        actual=pattern.actual,
        evidence=pattern.evidence,
    )
    try:
        response = model.invoke([("user", prompt)])
        raw = str(response.content).strip()
        # Strip any accidental markdown fences
        raw = re.sub(r"^```(?:yaml)?[\r\n]+", "", raw, flags=re.MULTILINE)
        raw = re.sub(r"^```\s*$", "", raw, flags=re.MULTILINE)
        raw = raw.strip()
        # Quick sanity check — must contain task_type
        if "task_type" not in raw:
            logger.warning("REFLECTOR ▸ LLM output missing 'task_type', skipping.")
            return None
        return raw
    except Exception as exc:  # noqa: BLE001
        logger.warning("REFLECTOR ▸ reflection generation failed: %s", exc)
        return None


# ──────────────────────────────────────────────────────────────────────────────
# YAML append
# ──────────────────────────────────────────────────────────────────────────────

def append_to_reflections(
    yaml_path: str,
    new_entries_yaml: list[str],
) -> int:
    """Append validated reflection entries to ``reflections.yaml``.

    Validates that each entry parses cleanly as YAML before writing.
    Returns the count of entries successfully appended.
    """
    if not new_entries_yaml:
        return 0

    valid: list[str] = []
    for raw in new_entries_yaml:
        try:
            parsed = yaml.safe_load(raw)
            if isinstance(parsed, list) and parsed:
                valid.append(raw)
            elif isinstance(parsed, dict) and "task_type" in parsed:
                valid.append(raw)
            else:
                logger.warning("REFLECTOR ▸ Skipping entry — unexpected YAML shape.")
        except yaml.YAMLError as exc:
            logger.warning("REFLECTOR ▸ Skipping unparseable YAML: %s", exc)

    if not valid:
        logger.info("REFLECTOR ▸ No valid entries to append.")
        return 0

    try:
        with open(yaml_path, encoding="utf-8") as fh:
            existing = fh.read()
    except OSError:
        existing = "reflections:\n"

    datestamp = time.strftime("%Y-%m-%d %H:%M")
    block = f"\n  # ── Auto-generated ({datestamp}) ─────────────────────────────────\n"
    for entry in valid:
        # Indent every non-blank line by 2 spaces so the list item
        # lives under the top-level ``reflections:`` mapping.
        lines = entry.splitlines()
        indented_lines: list[str] = []
        for line in lines:
            indented_lines.append(("  " + line) if line.strip() else line)
        block += "\n".join(indented_lines) + "\n"

    new_content = existing.rstrip() + "\n" + block
    try:
        with open(yaml_path, "w", encoding="utf-8") as fh:
            fh.write(new_content)
        logger.info(
            "REFLECTOR ▸ Appended %d reflection(s) to %s",
            len(valid), yaml_path,
        )
        return len(valid)
    except OSError as exc:
        logger.warning("REFLECTOR ▸ Could not write %s: %s", yaml_path, exc)
        return 0


# ──────────────────────────────────────────────────────────────────────────────
# High-level orchestrator called by the graph's reflect node
# ──────────────────────────────────────────────────────────────────────────────

def run_reflection_pass(
    model: Any,
    traces: list[WorkerTrace],
    reflections_path: str = "reflections.yaml",
    *,
    deduplicate: bool = True,
) -> int:
    """Full reflection pipeline: detect → generate → append.

    Args:
        model:            LLM to use for reflection entry generation.
        traces:           WorkerTrace objects captured during the run.
        reflections_path: Path to the YAML file to append to.
        deduplicate:      Skip generating entries for pattern types that
                          already appear in the existing YAML (avoids
                          accumulating near-identical entries).

    Returns:
        Number of new reflection entries appended.
    """
    patterns = detect_failure_patterns(traces)
    if not patterns:
        logger.info("REFLECTOR ▸ No failure patterns detected — no reflection needed.")
        return 0

    logger.info(
        "REFLECTOR ▸ Detected %d failure pattern(s): %s",
        len(patterns),
        ", ".join(p.pattern_type for p in patterns),
    )

    # Optional deduplication: load existing YAML and skip pattern types
    # that already have an entry with the same task_type + pattern_type.
    existing_keys: set[str] = set()
    if deduplicate and os.path.isfile(reflections_path):
        try:
            with open(reflections_path, encoding="utf-8") as fh:
                doc = yaml.safe_load(fh)
            for entry in (doc or {}).get("reflections", []):
                if isinstance(entry, dict):
                    existing_keys.add(
                        f"{entry.get('task_type', '')}::{entry.get('pattern_type', '')}"
                    )
        except Exception:  # noqa: BLE001
            pass

    new_entries: list[str] = []
    for pattern in patterns:
        dedup_key = f"{pattern.task_type}::{pattern.pattern_type}"
        if deduplicate and dedup_key in existing_keys:
            logger.info(
                "REFLECTOR ▸ Skipping duplicate: %s (already in YAML).",
                dedup_key,
            )
            continue
        entry_yaml = generate_reflection_entry(model, pattern)
        if entry_yaml:
            new_entries.append(entry_yaml)
            existing_keys.add(dedup_key)   # prevent within-run dupes

    return append_to_reflections(reflections_path, new_entries)
