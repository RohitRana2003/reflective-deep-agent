"""Plan Once — General-Purpose Deep Agent POC.

Workers get the **full Deep Agents middleware stack** automatically:

- ``FilesystemMiddleware``  → ``ls``, ``read_file``, ``write_file``,
  ``edit_file``, ``glob``, ``grep`` tools — no hand-rolled @tool functions
- ``TodoListMiddleware``    → ``write_todos`` so workers can plan sub-tasks
- ``SummarizationMiddleware`` → auto-summarise long worker conversations
- ``PatchToolCallsMiddleware`` → fix malformed tool calls
- ``MemoryMiddleware``      → load AGENTS.md context (project knowledge)
- ``SkillsMiddleware``      → load skill definitions (specialized behaviours)
- ``SubAgentMiddleware``    → workers can spawn sub-sub-agents for complex jobs

Usage:
    cd plan_once_langgraph

    # File analysis (auto-detects read intent, uses mode='direct'):
    python examples/run_deep.py "find all ERROR lines in examples/sample_logs/app.log"

    # File editing (auto-detects write intent, uses mode='agent'):
    python examples/run_deep.py "edit examples/sample_logs/app.log and add 10 more log entries"

    # General tasks:
    python examples/run_deep.py "create a Python hello world script at output/hello.py"
"""

import logging
import sys

from deepagents.backends import FilesystemBackend
from langchain_core.messages import HumanMessage
from langchain_ollama import ChatOllama

from plan_once import create_plan_once_agent, get_tracker

DEFAULT_QUERY = (
    "Perform a comprehensive analysis of the log file at "
    "'examples/sample_logs/app.log'. Specifically:\n"
    "1. Count entries by log level (INFO/WARN/ERROR).\n"
    "2. Extract all ERROR lines and identify root causes.\n"
    "3. Search for slow-query warnings.\n"
    "4. Check for security-related events.\n"
    "5. Write the final report to 'output/log_report.md'."
)


def main() -> None:
    """Run the general-purpose Deep Agent."""
    # Configure logging so LLM call records print in real-time
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(message)s",
        datefmt="%H:%M:%S",
    )
    # Suppress httpx request noise (Ollama API calls clutter the output)
    logging.getLogger("httpx").setLevel(logging.WARNING)

    query = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_QUERY

    # Reset the global tracker so each run starts fresh
    tracker = get_tracker()
    tracker.reset()

    model = ChatOllama(model="qwen3-coder:480b-cloud")

    # ------------------------------------------------------------------
    # Full Deep Agents feature stack:
    #
    # backend=FilesystemBackend(".", virtual_mode=True):
    #   Workers see a virtual filesystem rooted at the current directory.
    #   The middleware normalizes all paths to start with "/" (e.g.,
    #   "/examples/sample_logs/app.log"), and virtual_mode=True strips
    #   the leading "/" and joins to root_dir, so it resolves to the
    #   REAL file at ./examples/sample_logs/app.log.
    #
    #   virtual_mode=False is INCOMPATIBLE with the middleware layer
    #   because the middleware always prepends "/" to paths, and the
    #   backend treats "/..." as absolute OS paths (C:\examples\...).
    #
    # The agent automatically detects intent from the query:
    #   - READ/ANALYZE tasks → mode='direct' (fast, no tools)
    #   - WRITE/EDIT/CREATE tasks → mode='agent' (filesystem tools)
    #   - COMPLEX tasks → mode='agent' (full ReAct loop)
    # ------------------------------------------------------------------
    agent = create_plan_once_agent(
        model,
        enable_deep_features=True,
        backend=FilesystemBackend(root_dir=".", virtual_mode=True),
        # Deep Agents feature flags:
        enable_filesystem=True,      # ls, read_file, write_file, edit_file, glob, grep
        enable_todos=False,          # OFF — adds wasted ReAct turns for simple tasks
        enable_summarization=False,  # OFF for small tasks; enable for large contexts
        enable_subagents=False,      # task tool — workers spawn sub-sub-agents
        # ---- AGENTS.md (Memory) ----
        # Injects project knowledge into every worker's system prompt.
        # Workers see the project structure, directory layout, output
        # standards, and efficiency principles automatically.
        enable_memory=True,
        memory_sources=["/AGENTS.md", "/reflections.yaml"],  # project knowledge + past failure reflections
        # ---- Skills (SKILL.md) ----
        # Injects skill NAME + DESCRIPTION into worker system prompt.
        # Workers use read_file to load the full SKILL.md on demand
        # (progressive disclosure — only loads what's relevant).
        # Available skills: log-analysis, file-editing, code-generation,
        # report-writing
        enable_skills=True,
        skill_sources=["/skills/"],
        # ---- Auto-Reflection Loop (ExACT-inspired) ----
        # After every run the reflect node analyses worker tool-call traces
        # for failure patterns (over-exploration, duplicate calls,
        # path-not-found, empty search results).  When a failure is found
        # one LLM call generates a contrastive YAML entry:
        #   expected / actual / reflection
        # and appends it to reflections.yaml.  On the NEXT run that entry
        # is injected via MemoryMiddleware, closing the learning loop.
        # Zero cost when no failures are detected.
        enable_reflection=True,
        reflections_path="reflections.yaml",
        # Prompts: general-purpose (not log-specific)
        # ExACT-inspired: backtrack + explore on failure (Exploratory Learning)
        worker_prompt=(
            "You are a capable Worker Agent with full filesystem access. "
            "You can read, write, edit, and create files. You can analyze "
            "data, generate content, write code, and more.\\n\\n"
            "STRICT SCOPE & STOP RULES:\\n"
            "- SCOPE: Answer ONLY what the task explicitly asks. "
            "  Do NOT add unrequested extra counts, sections, or analysis.\\n"
            "- STOP: The moment you have the specific answer requested, "
            "  write your final response. Do NOT make any more tool calls.\\n"
            "- HARD LIMIT: Maximum 5 tool calls for ANY task. "
            "  If you reach 5 calls, write your answer with the data you have.\\n"
            "- SIMPLE LOOKUPS: 1-2 tool calls (grep or read_file — not both).\\n"
            "- ANALYSIS TASKS: max 4 tool calls — plan upfront what you need.\\n"
            "- NO NARRATION: No 'Let me first...' or 'I will now...' commentary.\\n\\n"
            "EFFICIENT TOOL STRATEGY:\\n"
            "- Log counts: ONE grep per level (grep pattern='ERROR', grep pattern='WARN').\\n"
            "- Root cause: grep for ERROR+WARN in ONE call using pattern='ERROR|WARN', "
            "  then ONE read_file for context if needed. Stop after 3 tool calls max.\\n"
            "- File not found: ONE ls/glob to locate it, then proceed. Do not repeat.\\n\\n"
            "EXPLORE & BACKTRACK (only if first attempt fails):\\n"
            "- If a tool call returns empty/unexpected results, "
            "  try ONE alternative. State: 'Expected X, got Y, trying Z'.\\n"
            "- Never retry the same tool call with the same arguments.\\n\\n"
            "This IS the final output — make it complete and professional."
        ),
        # orchestrator_prompt and synthesizer_prompt use the general-purpose
        # defaults from graph.py — no need to override unless specializing.
    )

    result = agent.invoke(
        {"messages": [HumanMessage(content=query)]}
    )

    print("\n" + "=" * 72)
    print("RESULT")
    print("=" * 72)
    print(result["messages"][-1].content)

    # Print verbose I/O log (shows what went IN and OUT of each call)
    print()
    print(tracker.verbose_log())

    # Print the compact token usage summary
    print()
    print(tracker.summary())


if __name__ == "__main__":
    main()
