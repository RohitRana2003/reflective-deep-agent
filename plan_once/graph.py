"""Plan Once, Execute Once — DAG graph builder.

Constructs a directed acyclic graph (DAG) that is *architecturally incapable*
of looping.  The flow is strictly::

    START ─► triage ─► orchestrator ─► execute_workers ─► synthesizer ─► END
                                    └──────────────────► synthesizer ─► END

The **triage** node is a zero-cost Python function (no LLM call) that:
  - Extracts file paths from the query
  - Detects intent (READ, WRITE, MIXED, GENERAL)
  - Reads files and counts lines
  - Classifies complexity (SIMPLE/MODERATE/COMPLEX)
  - Recommends mode ('direct' vs 'agent') and features

The **smart worker node** auto-injects file contents from triage into each
worker's task description, so the orchestrator only needs to describe *what*
to do — not re-paste file contents.

Workers in 'agent' mode get the full Deep Agents middleware stack:
filesystem tools, todos, summarization, memory, skills, subagents.
"""

from __future__ import annotations

import logging
import os
import re
from collections.abc import Callable, Sequence
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_ollama import ChatOllama
from langchain_core.messages import ToolMessage as LCToolMessage
from langchain_core.tools import BaseTool
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from plan_once.state import PlanOnceState
from plan_once.token_tracker import get_tracker
from plan_once.workers import build_worker_tool

_worker_logger = logging.getLogger("plan_once.workers")

# ---------------------------------------------------------------------------
# Default prompts
# ---------------------------------------------------------------------------
_DEFAULT_ORCHESTRATOR_PROMPT = (
    "You are a general-purpose Orchestrator. You have exactly ONE TURN.\n\n"
    "Read the CONTEXT ASSESSMENT below, then delegate ALL work using "
    "`delegate_to_worker` tool calls based on your intelligent analysis.\n\n"
    "The `delegate_to_worker` tool takes:\n"
    "  - task_description: describe the task for the worker\n"
    "  - mode: 'direct' (single LLM call, fastest when files are embedded)\n"
    "          'agent'  (full agent with filesystem tools for complex ops)\n\n"
    "INTELLIGENT DECISION MAKING:\n"
    "- When files are embedded below: consider mode='direct' for analysis tasks\n"
    "- When files need to be created/modified: use mode='agent' for filesystem tools\n" 
    "- When files are too large: use mode='agent' and let workers read selectively\n"
    "- For complex multi-step tasks: use multiple specialized workers\n"
    "- For simple tasks: use a single worker\n\n"
    "SKILLS: If skills are available in the context, pick the most relevant one "
    "per worker and include it in task_description.\n"
    "Example: 'Use the log-analysis skill to count ERROR lines in app.txt'\n\n"
    "You can handle ANY task — analysis, file editing, code generation, "
    "content creation, research, planning, etc. Make intelligent decisions "
    "based on the full context provided."
)

_DEFAULT_SYNTHESIZER_PROMPT = (
    "You are a Synthesizer. Review the worker execution results in the \n"
    "conversation history and produce a cohesive, well-structured \n"
    "response. Directly address the user's original request.\n\n"
    "If workers performed file operations (write, edit, create), \n"
    "confirm what was done and where. If they analyzed data, provide \n"
    "a clear summary of findings with actionable insights."
)


# ---------------------------------------------------------------------------
# Triage — zero-cost file pre-read and complexity assessment
# ---------------------------------------------------------------------------
_FILE_EXTENSIONS = (
    r"\.(?:log|txt|csv|json|xml|yaml|yml|md|py|js|ts|jsx|tsx|html|css"
    r"|cfg|conf|ini|properties|out|err|sh|bat|ps1|sql|toml|lock|env)"
)
_FILE_PATH_RE = re.compile(
    rf"(?:[\w./$\\-]+/)*[\w.$-]+{_FILE_EXTENSIONS}", re.IGNORECASE,
)

# Intent detection patterns - REMOVED
# Let the orchestrator LLM make intelligent decisions based on context
# instead of following hardcoded regex rules

_triage_logger = logging.getLogger("plan_once.triage")


def _load_available_skills(skills_root: str = "skills") -> dict[str, tuple[str, str]]:
    """Dynamically scan the skills/ directory.

    Returns {name: (description, full_body)} so callers can inject the
    complete SKILL.md content directly — eliminating the worker's
    read_file round-trip for the skill instructions.
    """
    available: dict[str, tuple[str, str]] = {}
    if not os.path.isdir(skills_root):
        return available
    for entry in sorted(os.listdir(skills_root)):
        skill_dir = os.path.join(skills_root, entry)
        skill_md = os.path.join(skill_dir, "SKILL.md")
        if not os.path.isfile(skill_md):
            continue
        try:
            with open(skill_md, encoding="utf-8", errors="replace") as f:
                text = f.read()
            name, desc = entry, ""
            if text.startswith("---"):
                end = text.find("\n---", 3)
                if end != -1:
                    import yaml as _yaml  # noqa: PLC0415
                    fm = _yaml.safe_load(text[3:end])
                    name = fm.get("name", entry)
                    desc = fm.get("description", "")
            available[name] = (desc, text)
        except Exception:  # noqa: BLE001
            available[entry] = ("", "")
    return available





def _extract_file_paths(text: str) -> list[str]:
    """Extract plausible file paths from free-form text."""
    matches = _FILE_PATH_RE.findall(text)
    seen: set[str] = set()
    result: list[str] = []
    for m in matches:
        normalized = m.replace("\\", "/")
        if normalized not in seen:
            seen.add(normalized)
            result.append(m)
    return result


def _build_triage_node() -> Callable[[PlanOnceState], dict[str, Any]]:
    """Build a zero-cost triage node (pure Python, no LLM call).

    Extracts file paths from the user's query, reads them, and provides
    full context to the orchestrator LLM. The orchestrator decides worker
    modes and task distribution intelligently based on the complete picture.
    """

    def triage(state: PlanOnceState) -> dict[str, Any]:
        # Find the latest user query
        user_query = ""
        for msg in reversed(state["messages"]):
            if hasattr(msg, "type") and getattr(msg, "type", None) == "human":
                user_query = str(msg.content)
                break

        file_paths = _extract_file_paths(user_query)

        _triage_logger.info("─" * 60)
        # _triage_logger.info("TRIAGE ▸ INTENT: %s | QUERY: %s", intent.upper(), user_query[:100])
        _triage_logger.info("─" * 60)

        # ── Skill discovery (pure Python, zero LLM cost) ─────────────
        available_skills = _load_available_skills()
        skills_block = ""
        if available_skills:
            # Inject full SKILL.md body — worker never needs to read_file it
            skill_sections = []
            for sk_name, (sk_desc, sk_body) in available_skills.items():
                skill_sections.append(
                    f"\n--- SKILL: {sk_name} ---\n"
                    f"Description: {sk_desc}\n"
                    f"{sk_body}\n"
                    f"--- END SKILL: {sk_name} ---"
                )
            skills_block = (
                "\nAVAILABLE SKILLS (full content pre-loaded — do NOT read_file them):\n"
                + "\n".join(f"  - {n}: {d}" for n, (d, _) in available_skills.items())
                + "\nInclude the skill name in the worker's task_description.\n"
                + "\n".join(skill_sections)
                + "\n"
            )
            _triage_logger.info(
                "TRIAGE ▸ SKILLS: %s (full content injected)",
                ", ".join(available_skills.keys()),
            )

        # Read explicit file paths first
        file_contents: dict[str, str] = {}
        total_lines = 0
        import glob as _glob  # noqa: PLC0415
        
        for path in file_paths:
            if os.path.isfile(path):
                candidates = [path]
            else:
                # Fuzzy matching: user wrote app.log but file is app.txt
                stem = os.path.splitext(os.path.basename(path))[0]
                parent = os.path.dirname(path) or "."
                candidates = _glob.glob(os.path.join(parent, f"{stem}.*"))
                if candidates:
                    _triage_logger.info(
                        "TRIAGE ▸ FILES: '%s' not found, fuzzy-matched → %s",
                        path, [os.path.basename(c) for c in candidates],
                    )
            
            for candidate in candidates:
                if os.path.isfile(candidate) and candidate not in file_contents:
                    try:
                        with open(candidate, encoding="utf-8", errors="replace") as f:
                            content = f.read()
                        file_contents[candidate] = content
                        total_lines += content.count("\n") + 1
                    except OSError:
                        pass

        # Auto-discover files if no explicit paths found
        if not file_contents:
            discovered: list[str] = []
            for pattern in ("**/*.log", "**/*.txt", "**/*.csv", "**/*.json"):
                discovered.extend(_glob.glob(pattern, recursive=True)[:10])
                if len(discovered) >= 15:
                    break
            discovered = list(dict.fromkeys(p.replace("\\", "/") for p in discovered))[:15]

            # Match discovered files against filename words in query
            query_words = set(w.lower() for w in re.split(r'[\s\'\"]+', user_query))
            query_stems = {os.path.splitext(w)[0] for w in query_words if "." in w and len(w) > 3}
            
            for p in discovered:
                basename = os.path.basename(p).lower()
                file_stem = os.path.splitext(basename)[0]
                if (
                    basename in query_words
                    or file_stem in query_stems
                    or any(basename.startswith(w) for w in query_words if len(w) > 3)
                ):
                    try:
                        with open(p, encoding="utf-8", errors="replace") as f:
                            content = f.read()
                        file_contents[p] = content
                        total_lines += content.count("\n") + 1
                    except OSError:
                        pass

            _triage_logger.info(
                "TRIAGE ▸ FILES: Auto-discovered %d, matched+read %d by filename.",
                len(discovered), len(file_contents),
            )

        # Classify complexity based on total content
        if total_lines <= 200:
            complexity = "SIMPLE"
        elif total_lines <= 2000:
            complexity = "MODERATE"
        else:
            complexity = "COMPLEX"

        # Always embed files for SIMPLE/MODERATE complexity
        file_block = ""
        if file_contents and complexity in ("SIMPLE", "MODERATE"):
            file_parts: list[str] = []
            for path, content in file_contents.items():
                line_count = content.count("\n") + 1
                file_parts.append(f"--- FILE: {path} ({line_count} lines) ---")
                file_parts.append(content)
                file_parts.append("--- END FILE ---")
                file_parts.append("")
            file_block = "\n".join(file_parts)

        # Build neutral context for orchestrator
        parts: list[str] = [
            "CONTEXT ASSESSMENT:",
            f"  Files found: {len(file_contents)}",
            f"  Total lines: {total_lines}",
            f"  Complexity:  {complexity}",
        ]

        if file_contents:
            parts.append("  Available files:")
            for path, content in file_contents.items():
                line_count = content.count("\n") + 1
                parts.append(f"    - {path}: {line_count} lines")

        if complexity in ("SIMPLE", "MODERATE") and file_contents:
            parts.extend([
                "",
                "FILE CONTENTS: Available below (auto-injected into workers)",
                "You can use mode='direct' for analysis tasks or mode='agent' for file operations.",
            ])
        elif complexity == "COMPLEX":
            parts.extend([
                "",
                "FILES TOO LARGE: Use mode='agent' and let workers read files as needed.",
            ])
        else:
            # No files found
            parts.extend([
                "",
                "NO FILES: Use mode='agent' if workers need to find/create files,", 
                "or mode='direct' for pure reasoning tasks.",
            ])

        triage_ctx = "\n".join(parts)
        _triage_logger.info(
            "TRIAGE ▸ RESULT: %d file(s), %d lines → %s | skills: %s",
            len(file_contents), total_lines, complexity,
            ", ".join(available_skills.keys()) if available_skills else "none",
        )
        
        return {
            "triage_context": triage_ctx + skills_block,
            "file_contents_for_workers": file_block,
        }

    return triage


# ---------------------------------------------------------------------------
# Node builders
# ---------------------------------------------------------------------------
def _build_orchestrator_node(
    model: BaseChatModel,
    worker_tool: BaseTool,
    *,
    system_prompt: str,
) -> Callable[[PlanOnceState], dict[str, Any]]:
    """Build the orchestrator node function.

    The orchestrator gets a *single* LLM turn.  It must analyze the user's
    request and emit all necessary ``delegate_to_worker`` tool calls at once.
    """
    model_with_tools = model.bind_tools([worker_tool])
    model_name = getattr(model, "model", None) or getattr(model, "model_name", "unknown")

    def orchestrator(state: PlanOnceState) -> dict[str, Any]:
        # Incorporate triage context into the system prompt so the
        # orchestrator knows file sizes and has contents pre-loaded.
        triage_ctx = state.get("triage_context", "")
        prompt = f"{system_prompt}\n\n{triage_ctx}" if triage_ctx else system_prompt
        messages = [{"role": "system", "content": prompt}, *state["messages"]]
        t0 = __import__("time").time()
        response = model_with_tools.invoke(messages)
        duration = __import__("time").time() - t0
        get_tracker().record(
            role="orchestrator",
            response=response,
            model_name=str(model_name),
            duration_seconds=duration,
            input_messages=messages,
        )
        return {"messages": [response]}

    return orchestrator


def _build_synthesizer_node(
    model: BaseChatModel,
    *,
    system_prompt: str,
) -> Callable[[PlanOnceState], dict[str, Any]]:
    """Build the synthesizer node function.

    Runs *without* tools so it can only produce a natural-language response.
    """

    model_name = getattr(model, "model", None) or getattr(model, "model_name", "unknown")

    def synthesizer(state: PlanOnceState) -> dict[str, Any]:
        messages = [{"role": "system", "content": system_prompt}, *state["messages"]]
        t0 = __import__("time").time()
        response = model.invoke(messages)
        duration = __import__("time").time() - t0
        get_tracker().record(
            role="synthesizer",
            response=response,
            model_name=str(model_name),
            duration_seconds=duration,
            input_messages=messages,
        )
        return {"messages": [response]}

    return synthesizer


# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------
def _make_route_after_workers(enable_reflection: bool) -> Callable[[PlanOnceState], str]:
    """Return a routing function that respects the reflection flag.

    When *enable_reflection* is ``True``, single-worker tasks route to the
    ``reflect`` node instead of directly to ``END``, so the reflection pass
    runs regardless of whether a synthesizer was needed.
    """
    single_worker_target = "reflect" if enable_reflection else END

    def _route_after_workers(state: PlanOnceState) -> str:
        """Route after workers: skip synthesizer if there was only 1 worker.

        When SIMPLE triage produced a single worker, the worker's output IS
        the final report.  No need to waste another LLM call re-summarizing it.
        """
        from langchain_core.messages import ToolMessage  # noqa: PLC0415

        tool_msgs = [m for m in state["messages"] if isinstance(m, ToolMessage)]
        if len(tool_msgs) <= 1:
            return single_worker_target
        return "synthesizer"

    return _route_after_workers


def _build_reflect_node(
    model: BaseChatModel,
    reflections_path: str = "reflections.yaml",
) -> Callable[[PlanOnceState], dict[str, Any]]:
    """Build the auto-reflection node.

    This node runs at the END of every graph execution.  It:

    1. Reads all ``WorkerTrace`` objects accumulated during the run.
    2. Runs a zero-cost Python heuristic pass to detect failure patterns
       (over-exploration, duplicate calls, path-not-found, empty search,
       direct-mode escape).
    3. For each detected pattern, fires ONE LLM call to generate a
       contrastive YAML entry (ExACT-style: expected / actual / reflection).
    4. Appends valid new entries to ``reflections.yaml`` so future runs
       benefit from the lesson learned.

    If no failures are detected the node is a no-op (zero LLM cost).
    """
    _reflect_logger = logging.getLogger("plan_once.reflect")

    def reflect(state: PlanOnceState) -> dict[str, Any]:
        from plan_once.reflector import run_reflection_pass  # noqa: PLC0415

        traces = get_tracker().worker_traces
        if not traces:
            _reflect_logger.info("REFLECT ▸ No worker traces — skipping.")
            return {}

        _reflect_logger.info(
            "REFLECT ▸ Analysing %d worker trace(s) for failure patterns...",
            len(traces),
        )
        count = run_reflection_pass(
            model,
            traces,
            reflections_path=reflections_path,
        )
        if count:
            _reflect_logger.info(
                "REFLECT ▸ %d new reflection(s) written to %s.",
                count, reflections_path,
            )
        else:
            _reflect_logger.info("REFLECT ▸ No new reflections needed.")
        return {}

    return reflect


def _route_orchestrator(state: PlanOnceState) -> str:
    """Route to skill_selector (if tool calls exist) or synthesizer."""
    last_message = state["messages"][-1]
    if hasattr(last_message, "tool_calls") and len(last_message.tool_calls) > 0:
        return "execute_workers"
    return "synthesizer"


def _build_smart_worker_node(
    worker_tool: BaseTool,
) -> Callable[[PlanOnceState], dict[str, Any]]:
    """Build a worker execution node that auto-injects file contents.

    Instead of using LangGraph's generic ``ToolNode``, this custom node
    reads ``file_contents_for_workers`` from state and prepends it to each
    worker's ``task_description``.  This means the orchestrator only needs
    to describe *what* to analyze (short output) — the file data is injected
    automatically, eliminating the biggest source of wasted tokens.
    """

    def execute_workers(state: PlanOnceState) -> dict[str, Any]:
        # Find the orchestrator's AI message with tool calls
        last_ai = state["messages"][-1]
        tool_calls = getattr(last_ai, "tool_calls", []) or []
        if not tool_calls:
            return {"messages": []}

        file_data = state.get("file_contents_for_workers", "")

        results: list[LCToolMessage] = []
        for i, tc in enumerate(tool_calls):
            # Extract args
            args = tc.get("args", tc) if isinstance(tc, dict) else tc
            if isinstance(args, dict) and "args" in args:
                inner_args = dict(args["args"])
            elif isinstance(args, dict):
                inner_args = dict(args)
            else:
                inner_args = {"task_description": str(args)}

            task_desc = inner_args.get("task_description", "")

            # Auto-inject file contents if available and not already present
            if file_data and file_data.strip() not in task_desc:
                task_desc = (
                    f"{task_desc}\n\n"
                    f"FILE CONTENTS (auto-injected):\n{file_data}"
                )
                inner_args["task_description"] = task_desc

            # Call the worker tool
            try:
                result_text = worker_tool.invoke(inner_args)
            except Exception as exc:  # noqa: BLE001
                result_text = f"Worker error: {exc}"

            tc_id = (
                tc.get("id", "") if isinstance(tc, dict)
                else getattr(tc, "id", "")
            )
            results.append(
                LCToolMessage(
                    content=str(result_text),
                    tool_call_id=tc_id,
                )
            )
            _worker_logger.info(
                "Worker returned %d chars for task: %.60s...",
                len(str(result_text)),
                task_desc[:60],
            )

        return {"messages": results}

    return execute_workers


# ---------------------------------------------------------------------------
# Public factory
# ---------------------------------------------------------------------------
def create_plan_once_agent(
    model: BaseChatModel | None = None,
    *,
    worker_tools: Sequence[BaseTool | Callable | dict[str, Any]] | None = None,
    worker_model: BaseChatModel | None = None,
    worker_prompt: str | None = None,
    orchestrator_prompt: str | None = None,
    synthesizer_prompt: str | None = None,
    worker_recursion_limit: int = 50,
    # Deep Agents integration -------------------------------------------------
    enable_deep_features: bool = False,
    backend: Any | None = None,
    enable_filesystem: bool = True,
    enable_todos: bool = True,
    enable_summarization: bool = True,
    enable_subagents: bool = False,
    enable_memory: bool = False,
    memory_sources: list[str] | None = None,
    enable_skills: bool = False,
    skill_sources: list[str] | None = None,
    # Auto-reflection loop ----------------------------------------------------
    enable_reflection: bool = False,
    reflections_path: str = "reflections.yaml",
) -> CompiledStateGraph:
    """Create a Plan Once, Execute Once agent graph.

    Builds a DAG that physically prevents recursion:

    1. **Orchestrator** — plans and emits parallel ``delegate_to_worker``
       tool calls in a *single* LLM turn.
    2. **Workers** — each tool call spins up an independent sub-agent that
       executes its task in isolation.
    3. **Synthesizer** — merges all worker outputs into a single response.

    The critical distinction from a standard ReAct agent is that the worker
    node routes to the synthesizer, *never* back to the orchestrator.

    Args:
        model: LLM for orchestrator and synthesizer.  Defaults to
            ``ChatOllama(model="qwen3-coder:480b-cloud")``.
        worker_tools: Tools available to each worker sub-agent.
        worker_model: LLM for workers.  Defaults to *model*.
        worker_prompt: System prompt for workers.
        orchestrator_prompt: System prompt for the orchestrator.
        synthesizer_prompt: System prompt for the synthesizer.
        enable_deep_features: When ``True``, wire Deep Agents middleware into
            every worker sub-agent.  Workers automatically gain filesystem
            tools, todo planning, summarization, and more — without
            hand-rolling ``@tool`` functions.  Defaults to ``False`` for
            backward compatibility.
        backend: A ``BackendProtocol`` instance or factory for the Deep Agents
            middleware stack.  Defaults to ``StateBackend`` (ephemeral).  Pass
            a ``FilesystemBackend`` for shared disk access.
        enable_filesystem: Give workers ``ls``, ``read_file``, ``write_file``,
            ``edit_file``, ``glob``, ``grep``.  Only used when
            *enable_deep_features* is ``True``.
        enable_todos: Give workers a ``write_todos`` planning tool.
        enable_summarization: Auto-summarise long worker conversations.
        enable_subagents: Give workers a ``task`` tool for sub-sub-agents.
        enable_memory: Load ``AGENTS.md`` files into worker system prompts.
        memory_sources: Paths to ``AGENTS.md`` files.
        enable_skills: Load skill definitions into worker system prompts.
        skill_sources: Paths to skill directories.
        enable_reflection: When ``True``, add an auto-reflection node that
            runs after every execution.  It analyses worker tool-call traces
            for failure patterns (over-exploration, duplicate calls,
            path-not-found, empty search) and writes new contrastive entries
            to ``reflections.yaml`` using one LLM call per detected pattern.
            Zero cost when no failures are detected.  Defaults to ``False``.
        reflections_path: Path to the YAML file that stores reflections.
            Defaults to ``"reflections.yaml"`` (project root).  Only used
            when *enable_reflection* is ``True``.

    Returns:
        A compiled ``StateGraph`` ready for ``.invoke()`` / ``.ainvoke()``.

    Example::

        from langchain_ollama import ChatOllama
        from plan_once import create_plan_once_agent

        agent = create_plan_once_agent(
            ChatOllama(model="qwen3-coder:480b-cloud"),
        )
        result = agent.invoke({
            "messages": [HumanMessage(content="Analyze these logs for errors.")]
        })
        print(result["messages"][-1].content)

    Example with Deep Agents features::

        agent = create_plan_once_agent(
            enable_deep_features=True,   # workers get filesystem, todos, etc.
            enable_memory=True,
            memory_sources=["/memory/AGENTS.md"],
        )
    """
    if model is None:
        model = ChatOllama(
            model="qwen3-coder:480b-cloud",
        )

    effective_worker_model = worker_model or model
    effective_orchestrator_prompt = orchestrator_prompt or _DEFAULT_ORCHESTRATOR_PROMPT
    effective_synthesizer_prompt = synthesizer_prompt or _DEFAULT_SYNTHESIZER_PROMPT

    # Build the worker delegation tool
    worker_kwargs: dict[str, Any] = {}
    if worker_prompt is not None:
        worker_kwargs["worker_prompt"] = worker_prompt

    # Deep Agents middleware for workers
    if enable_deep_features:
        from plan_once.deep_features import build_worker_middleware  # noqa: PLC0415

        worker_middleware = build_worker_middleware(
            effective_worker_model,
            backend=backend,
            enable_filesystem=enable_filesystem,
            enable_todos=enable_todos,
            enable_summarization=enable_summarization,
            enable_subagents=enable_subagents,
            enable_memory=enable_memory,
            memory_sources=memory_sources,
            enable_skills=enable_skills,
            skill_sources=skill_sources,
        )
        worker_kwargs["middleware"] = worker_middleware

    worker_tool = build_worker_tool(
        effective_worker_model,
        worker_tools=worker_tools,
        recursion_limit=worker_recursion_limit,
        **worker_kwargs,
    )

    # Build node functions
    orchestrator_fn = _build_orchestrator_node(
        model,
        worker_tool,
        system_prompt=effective_orchestrator_prompt,
    )
    synthesizer_fn = _build_synthesizer_node(
        model,
        system_prompt=effective_synthesizer_prompt,
    )

    # ----- Assemble the DAG -----
    workflow = StateGraph(PlanOnceState)

    workflow.add_node("triage", _build_triage_node())
    workflow.add_node("orchestrator", orchestrator_fn)
    workflow.add_node(
        "execute_workers",
        _build_smart_worker_node(worker_tool),
    )
    workflow.add_node("synthesizer", synthesizer_fn)

    # Auto-reflection node (optional)
    if enable_reflection:
        reflect_fn = _build_reflect_node(model, reflections_path=reflections_path)
        workflow.add_node("reflect", reflect_fn)

    # START ─► triage ─► orchestrator
    workflow.add_edge(START, "triage")
    workflow.add_edge("triage", "orchestrator")

    # orchestrator ─► execute_workers | synthesizer
    workflow.add_conditional_edges("orchestrator", _route_orchestrator)

    # workers ─► synthesizer (multiple workers) | reflect/END (single worker)
    route_after_workers = _make_route_after_workers(enable_reflection)
    workflow.add_conditional_edges("execute_workers", route_after_workers)

    if enable_reflection:
        # synthesizer ─► reflect ─► END
        workflow.add_edge("synthesizer", "reflect")
        workflow.add_edge("reflect", END)
    else:
        # synthesizer ─► END  (original behaviour)
        workflow.add_edge("synthesizer", END)

    return workflow.compile()
