# Plan Once, Execute Once — Log Analyzer — End-to-End Technical Documentation

## Table of Contents

1. [What Is This?](#1-what-is-this)
2. [The Problem It Solves](#2-the-problem-it-solves)
3. [Architecture Overview](#3-architecture-overview)
4. [Folder Structure](#4-folder-structure)
5. [File-by-File Breakdown](#5-file-by-file-breakdown)
   - [pyproject.toml](#pyprojecttoml)
   - [plan_once/\_\_init\_\_.py](#plan_once__init__py)
   - [plan_once/state.py](#plan_oncestatepy)
   - [plan_once/workers.py](#plan_onceworkerspy)
   - [plan_once/graph.py](#plan_oncegraphpy)
   - [plan_once/deep_features.py](#plan_oncedeep_featurespy)
   - [tests/\_\_init\_\_.py](#tests__init__py)
   - [tests/chat_model.py](#testschat_modelpy)
   - [tests/test_graph.py](#teststest_graphpy)
   - [tests/test_workers.py](#teststest_workerspy)
   - [examples/run_deep.py](#examplesrun_deeppy)
6. [Execution Flow — Step by Step](#6-execution-flow--step-by-step)
7. [Deep Agents Middleware Integration](#7-deep-agents-middleware-integration)
8. [Dependencies](#8-dependencies)
9. [How to Run](#9-how-to-run)
10. [What's Left to Explore](#10-whats-left-to-explore)

---

## 1. What Is This?

`plan_once_langgraph/` is a **standalone, self-contained** implementation of the "Plan Once, Execute Once" architecture described in `plan_once_architecture.md`, themed as a **log analyzer** and powered by **Ollama** (`qwen3-coder:480b-cloud`).

It builds a **pure LangGraph agent** that decomposes a log-analysis request into parallel sub-tasks, executes them all at once, and synthesizes the results into a cohesive report — all in a single forward pass with **zero possibility of infinite recursion**.

### Deep Agents Integration

When `enable_deep_features=True` is set, every worker sub-agent is automatically wrapped with the **Deep Agents middleware stack**:

| Middleware | What workers gain |
|------------|------------------|
| `FilesystemMiddleware` | `ls`, `read_file`, `write_file`, `edit_file`, `glob`, `grep` — no `@tool` functions needed |
| `TodoListMiddleware` | `write_todos` — workers plan their own sub-tasks internally |
| `SummarizationMiddleware` | Auto-summarise long conversations to prevent context overflow |
| `PatchToolCallsMiddleware` | Fix malformed tool calls from the LLM |
| `MemoryMiddleware` *(opt-in)* | Load `AGENTS.md` files into worker system prompt |
| `SkillsMiddleware` *(opt-in)* | Load skill definitions for specialized behavior |
| `SubAgentMiddleware` *(opt-in)* | Workers spawn their own sub-sub-agents |

The integration is **optional**: the core library (`plan_once/`) works without `deepagents` installed. Deep Agents is declared as an optional dependency under `pip install -e ".[deep]"`.

---

## 2. The Problem It Solves

### The Classic ReAct Loop Problem

A standard ReAct agent has this flow:

```
Agent → Tool → Agent → Tool → Agent → Tool → ... → Agent → END
```

The `Tool → Agent` edge creates a **cycle**. The agent can get stuck in infinite loops, burn tokens indefinitely, or hit recursion limits. Deep Agents' `create_deep_agent` uses `recursion_limit: 1000` to mitigate this, but the loop is still architecturally present.

### The Plan Once Solution

This implementation replaces the cycle with a **directed acyclic graph (DAG)**:

```
START → Orchestrator → Workers (parallel) → Synthesizer → END
```

There is no edge from Workers back to the Orchestrator. The graph **physically cannot loop**. It runs exactly once: plan, execute, synthesize, done.

---

## 3. Architecture Overview

```
                          ┌─────────────────┐
                          │      START      │
                          └────────┬────────┘
                                   │
                                   ▼
                          ┌─────────────────┐
                          │  ORCHESTRATOR   │
                          │                 │
                          │  Single LLM     │
                          │  turn. Emits    │
                          │  N tool calls.  │
                          └────────┬────────┘
                                   │
                      ┌────────────┼────────────┐
                      │  tool_calls?            │  no tool_calls
                      ▼                         ▼
             ┌─────────────────┐       ┌─────────────────┐
             │ EXECUTE_WORKERS │       │                 │
             │                 │       │                 │
             │ ToolNode runs   │       │                 │
             │ all calls in    │       │                 │
             │ parallel. Each  │       │                 │
             │ spins up a      │       │                 │
             │ fresh sub-agent │       │                 │
             │ (with optional  │       │                 │
             │  Deep Agents    │       │                 │
             │  middleware)    │       │                 │
             └────────┬────────┘       │                 │
                      │                │                 │
                      └───────┬────────┘                 │
                              ▼                          │
                     ┌─────────────────┐                 │
                     │   SYNTHESIZER   │◄────────────────┘
                     │                 │
                     │  No tools.      │
                     │  Reads all      │
                     │  worker results │
                     │  and produces   │
                     │  final answer.  │
                     └────────┬────────┘
                              │
                              ▼
                     ┌─────────────────┐
                     │      END        │
                     └─────────────────┘
```

### The Three Nodes

| Node | LLM Turn | Tools Bound? | Purpose |
|------|----------|-------------|---------|
| **Orchestrator** | 1 (exactly one) | Yes — `delegate_to_worker` | Reads the user's log-analysis request. Decomposes it into independent sub-tasks (error extraction, pattern search, level counting, etc.). Emits N parallel `delegate_to_worker` tool calls in a single response. |
| **Execute Workers** | 0 (LangGraph `ToolNode`) | N/A — it IS the tool executor | LangGraph's built-in `ToolNode` receives the tool calls from the orchestrator and executes them. Each `delegate_to_worker` call spins up an independent `create_agent` sub-agent with its own context window. When Deep Agents is enabled, each worker gets the middleware stack (filesystem, todos, summarization). |
| **Synthesizer** | 1 (exactly one) | No — plain LLM | Sees the full message history: user request → orchestrator plan → all worker results. Produces a structured log analysis report (executive summary, errors, patterns, recommendations). |

### The Critical Edge

```python
workflow.add_edge("execute_workers", "synthesizer")  # NOT back to orchestrator
```

In a standard ReAct agent, tools route **back to the agent** (`Tool → Agent`). Here, tools route **forward to the synthesizer** (`Workers → Synthesizer → END`). This single design choice is what makes recursion impossible.

---

## 4. Folder Structure

```
plan_once_langgraph/
│
├── pyproject.toml                 # Project config, deps, optional [deep] extra, pytest
├── README.md                      # Quick-start README
├── WALKTHROUGH.md                 # THIS FILE — full end-to-end documentation
│
├── plan_once/                     # Core library (5 files)
│   ├── __init__.py                # Public API exports (4 symbols)
│   ├── state.py                   # PlanOnceState TypedDict
│   ├── workers.py                 # build_worker_tool() factory (supports middleware)
│   ├── graph.py                   # create_plan_once_agent() DAG builder + Deep flags
│   └── deep_features.py           # Deep Agents middleware bridge
│
├── tests/                         # Unit tests (20 tests)
│   ├── __init__.py                # Package marker
│   ├── chat_model.py              # GenericFakeChatModel for deterministic testing
│   ├── test_graph.py              # 13 tests: routing, nodes, end-to-end graph
│   └── test_workers.py            # 7 tests: worker tool factory + state
│
└── examples/                      # Runnable log-analysis POC
    ├── run_deep.py                # Deep Agents middleware-powered log analyzer
    └── sample_logs/
        └── app.log                # Sample application log (42 lines)
```

**Total: 13 files** (+ this walkthrough)

---

## 5. File-by-File Breakdown

---

### `pyproject.toml`

**What it does**: Defines this as a standalone Python project with its own dependencies, build system, and test configuration.

**Key sections**:

| Section | Purpose |
|---------|---------|
| `[project]` | Name: `plan-once-langgraph`, version `0.1.0`, requires Python ≥ 3.11 |
| `dependencies` | `langchain-core`, `langchain`, `langchain-ollama`, `langgraph` — the four core packages |
| `[project.optional-dependencies] deep` | `deepagents>=0.4.7,<1.0.0` — install with `pip install -e ".[deep]"` to enable middleware |
| `[dependency-groups] test` | Just `pytest` — no other test dependencies |
| `[tool.setuptools.packages.find]` | Only packages the `plan_once` directory (not `tests/` or `examples/`) |
| `[tool.pytest.ini_options]` | Sets `pythonpath = ["."]` so `from plan_once import ...` works in tests without installing |

**Why `deepagents` is optional**: The core DAG architecture works without it. The middleware just enhances workers with automatic tools, planning, and context management.

---

### `plan_once/__init__.py`

**What it does**: Package entry point. Exports 4 public symbols.

```python
from plan_once.graph import create_plan_once_agent
from plan_once.state import PlanOnceState
from plan_once.workers import build_worker_tool

# Deep Agents integration (lazy — works even when deepagents is not installed)
try:
    from plan_once.deep_features import build_worker_middleware
except ImportError:
    build_worker_middleware = None
```

**Public API**:

| Symbol | Type | What it is |
|--------|------|-----------|
| `create_plan_once_agent` | Function | Main factory — builds and compiles the DAG |
| `PlanOnceState` | TypedDict | The graph state schema |
| `build_worker_tool` | Function | Creates the `delegate_to_worker` tool |
| `build_worker_middleware` | Function / `None` | Builds the Deep Agents middleware stack. Gracefully `None` if `deepagents` is not installed |

**Usage**: `from plan_once import create_plan_once_agent, build_worker_middleware`

---

### `plan_once/state.py`

**What it does**: Defines the single state object shared by all nodes in the graph.

```python
class PlanOnceState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]
```

**How it works**:
- `messages` is a list of LangChain `BaseMessage` objects (HumanMessage, AIMessage, ToolMessage, etc.)
- The `Annotated[..., operator.add]` tells LangGraph to **append** new messages from each node rather than overwriting the list
- When the orchestrator returns `{"messages": [response]}`, LangGraph appends `response` to the existing list
- When the ToolNode returns worker results, those get appended too
- By the time the synthesizer runs, it sees the full history: user message → orchestrator message → all worker results

**Why just messages?**: This is the minimal state needed for the DAG pattern. The orchestrator plans via its system prompt (not via a separate `todos` field). Worker results flow through messages. The synthesizer reads from messages. No extra state fields are required.

---

### `plan_once/workers.py`

**What it does**: Factory function that creates a LangChain `StructuredTool` called `delegate_to_worker`. When the orchestrator emits this tool call, it spins up a fresh sub-agent to execute the task.

**The function**: `build_worker_tool(model, *, worker_tools, worker_prompt, name, middleware) → BaseTool`

**Parameters**:

| Param | Default | Purpose |
|-------|---------|---------|
| `model` | (required) | The LLM each worker uses |
| `worker_tools` | `None` | Tools each worker can access (e.g., `read_file`, `search`) |
| `worker_prompt` | `"You are a Log Analysis Worker..."` | System instructions for workers |
| `name` | `"delegate_to_worker"` | Tool name visible to the orchestrator |
| `middleware` | `None` | List of `AgentMiddleware` instances (from Deep Agents). When provided, passed directly to `create_agent(…, middleware=…)` |

**What happens when the tool is called**:

1. A brand new `create_agent(model, tools=resolved_tools, middleware=resolved_middleware)` graph is created — completely isolated, fresh context window
2. The `task_description` string (provided by the orchestrator) is sent as a `("user", ...)` message to this worker
3. The worker agent runs to completion with `config={"recursion_limit": 200}` (it CAN loop internally — it's a regular ReAct agent, and Deep Agents middleware adds extra turns)
4. Only the **final message text** is extracted and returned as a string to the parent graph
5. The worker is discarded — its full message history, intermediate tool calls, etc. are garbage collected

**Why `recursion_limit=200`**: When Deep Agents middleware is active, each worker turn involves extra middleware steps (filesystem tool calls, todo planning, summarization checks). The default LangGraph limit of 25 is too low and causes `GraphRecursionError`. Setting it to 200 gives workers enough room to complete multi-step tasks.

**Why this matters**:
- **Context isolation**: Each worker only sees its own task description, not the full conversation. If the user asked "analyze 10 repositories," each worker only loads one repo into its context.
- **Token savings**: The orchestrator never sees the intermediate steps of each worker. It only gets a concise summary.
- **Independence**: Workers can't interfere with each other. They each get their own `create_agent` instance.

**Sync and async**: Both `_delegate` (sync) and `_adelegate` (async) implementations are provided, wrapped via `StructuredTool.from_function(func=..., coroutine=...)`.

---

### `plan_once/graph.py`

**What it does**: The main module. Builds and compiles the 3-node DAG. This is where the orchestrator prompt, synthesizer prompt, and the Deep Agents integration flags all come together.

**Private functions**:

#### `_build_orchestrator_node(model, worker_tool, *, system_prompt)`

Creates a closure that:
1. Calls `model.bind_tools([worker_tool])` — gives the LLM the ability to emit `delegate_to_worker` calls
2. Prepends the system prompt to the state messages
3. Invokes the LLM once
4. Returns the LLM's response (which may contain 0 or more tool calls)

The orchestrator's system prompt is critical — it tells the LLM:
```
You are a Log Analysis Orchestrator. You have exactly ONE TURN.
1. Identify all independent, parallel analysis steps.
2. Use delegate_to_worker for EVERY step concurrently.
DO NOT attempt to do the work yourself.
```

This prompt engineering is what forces the LLM to emit multiple parallel tool calls in a single response.

#### `_build_synthesizer_node(model, *, system_prompt)`

Creates a closure that:
1. Does **not** bind any tools — the synthesizer can only produce text
2. Prepends the synthesizer system prompt
3. Invokes the LLM once
4. Returns the final natural-language answer

Because the synthesizer has no tools, it cannot trigger more work. It can only summarize what the workers produced.

#### `_route_orchestrator(state) → str`

Routing function that checks the orchestrator's output:
- If the last message has `tool_calls` with length > 0 → route to `"execute_workers"`
- Otherwise → route directly to `"synthesizer"` (the orchestrator answered the question itself)

This handles the edge case where the request is so simple the orchestrator doesn't need workers at all (e.g., "What is 2+2?").

**Public function**:

#### `create_plan_once_agent(model, *, worker_tools, worker_model, worker_prompt, orchestrator_prompt, synthesizer_prompt, enable_deep_features, backend, enable_filesystem, enable_todos, enable_summarization, enable_subagents, enable_memory, memory_sources, enable_skills, skill_sources) → CompiledStateGraph`

The main factory. Assembles everything:

```python
workflow = StateGraph(PlanOnceState)

workflow.add_node("orchestrator", orchestrator_fn)
workflow.add_node("execute_workers", ToolNode([worker_tool]))
workflow.add_node("synthesizer", synthesizer_fn)

workflow.add_edge(START, "orchestrator")
workflow.add_conditional_edges("orchestrator", _route_orchestrator)
workflow.add_edge("execute_workers", "synthesizer")   # ← THE CRITICAL LINE
workflow.add_edge("synthesizer", END)

return workflow.compile()
```

**Core parameters**:

| Param | Default | Purpose |
|-------|---------|---------|
| `model` | `ChatOllama(model="qwen3-coder:480b-cloud")` | LLM for orchestrator + synthesizer |
| `worker_tools` | `None` | Tools passed to each worker |
| `worker_model` | Same as `model` | Separate LLM for workers (e.g., smaller model) |
| `worker_prompt` | Default log-analysis worker prompt | System instructions for workers |
| `orchestrator_prompt` | Default "plan in one turn" prompt | System instructions for orchestrator |
| `synthesizer_prompt` | Default "produce log analysis report" prompt | System instructions for synthesizer |

**Deep Agents integration parameters** (only used when `enable_deep_features=True`):

| Param | Default | Purpose |
|-------|---------|---------|
| `enable_deep_features` | `False` | Master switch — when `True`, builds middleware stack and passes it to every worker |
| `backend` | `None` → `StateBackend` (ephemeral) | A `BackendProtocol` instance. Pass `FilesystemBackend(root_dir=".", virtual_mode=False)` for real disk access |
| `enable_filesystem` | `True` | Workers get `ls`, `read_file`, `write_file`, `edit_file`, `glob`, `grep` |
| `enable_todos` | `True` | Workers get `write_todos` for internal planning |
| `enable_summarization` | `True` | Auto-summarise long worker conversations |
| `enable_subagents` | `False` | Workers get a `task` tool to spawn sub-sub-agents |
| `enable_memory` | `False` | Load `AGENTS.md` files into worker system prompt |
| `memory_sources` | `None` | Paths to memory files |
| `enable_skills` | `False` | Load skill definitions into worker system prompt |
| `skill_sources` | `None` | Paths to skill directories |

**What happens when `enable_deep_features=True`**:

1. `graph.py` imports `build_worker_middleware` from `plan_once.deep_features`
2. Calls it with the effective worker model and all feature flags
3. Gets back a list of middleware instances
4. Passes `middleware=worker_middleware` to `build_worker_tool()`
5. Workers are now created via `create_agent(model, tools=..., middleware=...)` — they get Deep Agents tools automatically

---

### `plan_once/deep_features.py`

**What it does**: Bridges the Deep Agents SDK middleware stack into Plan Once workers. This is the **only file** that directly imports from `deepagents`.

**The function**: `build_worker_middleware(model, *, backend, enable_filesystem, enable_todos, ...) → list[Any]`

**How it builds the middleware stack** (in order):

| Order | Middleware | Added when | What it provides |
|-------|-----------|------------|-----------------|
| 1 | `TodoListMiddleware` | `enable_todos=True` | `write_todos` tool — workers can plan sub-tasks |
| 2 | `MemoryMiddleware` | `enable_memory=True` | Loads `AGENTS.md` files into system prompt |
| 3 | `SkillsMiddleware` | `enable_skills=True` | Loads skill definitions into system prompt |
| 4 | `FilesystemMiddleware` | `enable_filesystem=True` | `ls`, `read_file`, `write_file`, `edit_file`, `glob`, `grep` tools |
| 5 | `SubAgentMiddleware` | `enable_subagents=True` | `task` tool — workers spawn sub-sub-agents |
| 6 | `SummarizationMiddleware` | `enable_summarization=True` | Auto-summarise conversations that exceed context limits |
| 7 | `PatchToolCallsMiddleware` | Always (if importable) | Fix malformed tool calls from the LLM |

**Backend handling**: If no `backend` is passed, defaults to `StateBackend` (the *class*, acting as a factory). For workers to read/write real files, pass `FilesystemBackend(root_dir=".", virtual_mode=False)`.

**Why `virtual_mode=False`**: `FilesystemBackend` defaults to `virtual_mode=True`, which layers a virtual overlay on top of the real filesystem. Setting it to `False` gives workers direct disk access and suppresses a deprecation warning.

**Lazy imports**: All `deepagents.*` imports are inside the function body so the module is importable even when `deepagents` is not installed. The `__init__.py` wraps the import in a `try/except` and falls back to `build_worker_middleware = None`.

---

### `tests/__init__.py`

**What it does**: Empty file. Marks `tests/` as a Python package so `from tests.chat_model import ...` works.

---

### `tests/chat_model.py`

**What it does**: Provides `GenericFakeChatModel` — a deterministic fake LLM for testing.

**How it works**:
- Takes an `Iterator[AIMessage]` of pre-programmed responses
- Each call to `_generate()` pops the next message from the iterator
- `bind_tools()` returns `self` (no-op) — so the fake model works when the orchestrator binds tools to it
- Tracks all calls in `call_history` for inspection in tests

**Why we need it**: Tests must be deterministic, fast, and not require a running Ollama instance or Deep Agents middleware. By pre-programming the exact sequence of LLM responses, we can verify the graph wiring without hitting the model.

**Example usage in tests**:
```python
model = GenericFakeChatModel(messages=iter([
    AIMessage(content="", tool_calls=[...]),   # orchestrator response
    AIMessage(content="worker result"),         # worker response
    AIMessage(content="final summary"),         # synthesizer response
]))
agent = create_plan_once_agent(model)
result = agent.invoke({"messages": [HumanMessage(content="...")]})
```

---

### `tests/test_graph.py`

**What it does**: 13 tests covering the graph builder, routing logic, and end-to-end execution.

**Test classes and what they verify**:

#### `TestRouteOrchestrator` (3 tests)
| Test | What it checks |
|------|---------------|
| `test_routes_to_workers_when_tool_calls_present` | AIMessage with tool_calls → returns `"execute_workers"` |
| `test_routes_to_synthesizer_when_no_tool_calls` | AIMessage with no tool_calls → returns `"synthesizer"` |
| `test_routes_to_synthesizer_when_empty_tool_calls` | AIMessage with `tool_calls=[]` → returns `"synthesizer"` |

#### `TestBuildOrchestratorNode` (2 tests)
| Test | What it checks |
|------|---------------|
| `test_orchestrator_invokes_model_with_system_prompt` | Model is called exactly once, system prompt is prepended |
| `test_orchestrator_returns_model_response` | Tool calls in the model response are preserved in the output |

#### `TestBuildSynthesizerNode` (1 test)
| Test | What it checks |
|------|---------------|
| `test_synthesizer_produces_final_response` | Given a full message history with worker results, synthesizer produces a final text response |

#### `TestCreatePlanOnceAgent` (7 tests)
| Test | What it checks |
|------|---------------|
| `test_graph_compiles_without_errors` | `create_plan_once_agent` returns a non-null compiled graph |
| `test_graph_has_expected_nodes` | Graph contains `orchestrator`, `execute_workers`, `synthesizer` nodes |
| `test_direct_answer_path` | Full invocation: orchestrator answers directly → synthesizer → correct final output |
| `test_delegation_path` | Full invocation: orchestrator delegates → workers execute → synthesizer merges → correct final output |
| `test_custom_prompts_accepted` | Custom orchestrator/synthesizer/worker prompts are accepted without errors |
| `test_custom_worker_tools_accepted` | Worker tools are accepted without errors |
| `test_no_recursion_in_graph_structure` | Verifies no edge exists from `execute_workers` or `synthesizer` back to `orchestrator` |

---

### `tests/test_workers.py`

**What it does**: 7 tests covering the worker tool factory and state definition.

#### `TestBuildWorkerTool` (6 tests)
| Test | What it checks |
|------|---------------|
| `test_returns_structured_tool` | Returns a StructuredTool with name `"delegate_to_worker"` |
| `test_custom_name` | Custom tool name is applied correctly |
| `test_tool_has_description` | Tool has a meaningful description containing "delegate" or "worker" |
| `test_tool_invocation_returns_string` | Invoking the tool returns a string containing the worker's response |
| `test_tool_with_custom_prompt` | Custom worker prompt works (result still comes through) |
| `test_tool_with_worker_tools` | Passing tools to the worker is accepted without errors |

#### `TestPlanOnceState` (1 test)
| Test | What it checks |
|------|---------------|
| `test_state_accepts_messages` | PlanOnceState TypedDict accepts a messages field with BaseMessage objects |

---

### `examples/run_deep.py`

**What it does**: The primary (and only) runnable example. Runs a log analysis using the **full Deep Agents middleware stack** on every worker.

**What workers get automatically** (no `@tool` functions needed):
- `ls`, `read_file`, `write_file`, `edit_file`, `glob`, `grep` — via `FilesystemMiddleware`
- `write_todos` — via `TodoListMiddleware`
- Auto-summarization — via `SummarizationMiddleware`
- Malformed tool call fixing — via `PatchToolCallsMiddleware`

**Key configuration**:
```python
agent = create_plan_once_agent(
    model,
    enable_deep_features=True,
    backend=FilesystemBackend(root_dir=".", virtual_mode=False),  # real disk access
    enable_filesystem=True,
    enable_todos=True,
    enable_summarization=True,
    enable_subagents=False,
    enable_memory=False,
    enable_skills=False,
)
```

**Why `FilesystemBackend(root_dir=".", virtual_mode=False)`**: The default `StateBackend` is ephemeral — workers would get filesystem tools but they'd operate on an empty in-memory filesystem and couldn't see any real files. `FilesystemBackend` with `root_dir="."` roots all file operations at the current working directory, and `virtual_mode=False` gives direct disk access.

**CLI support**: Accepts an optional query via `sys.argv[1]`:
```bash
python examples/run_deep.py "find all ERROR lines in examples/sample_logs/app.log"
```

If no argument is provided, uses a default comprehensive log-analysis query that:
1. Counts entries by log level
2. Extracts all ERROR lines
3. Searches for slow-query warnings
4. Checks for security-related events
5. Writes a final report to `output/log_report.md`

**Requirements**: Ollama running locally with `qwen3-coder:480b-cloud` loaded, plus `pip install -e ".[deep]"`.

---

## 6. Execution Flow — Step by Step

Here's exactly what happens when you call `agent.invoke({"messages": [HumanMessage(...)]})` with Deep Agents middleware enabled:

### Step 1: START → Orchestrator

LangGraph enters the `orchestrator` node.

**Input state**:
```python
{"messages": [HumanMessage(content="Analyze the log file at examples/sample_logs/app.log...")]}
```

**What happens**:
- System prompt is prepended: `"You are a Log Analysis Orchestrator. Workers have access to filesystem tools (ls, read_file, write_file, edit_file, glob, grep) and a todo list. You have exactly ONE TURN..."`
- `model.bind_tools([delegate_to_worker]).invoke(messages)` is called
- The LLM analyzes the request and produces an `AIMessage` with N tool calls:

```python
AIMessage(
    content="I'll delegate these analysis tasks...",
    tool_calls=[
        {"name": "delegate_to_worker", "args": {"task_description": "Use read_file to load examples/sample_logs/app.log, then count entries by level..."}, "id": "call_1"},
        {"name": "delegate_to_worker", "args": {"task_description": "Use grep to find all ERROR lines and identify root causes..."}, "id": "call_2"},
        {"name": "delegate_to_worker", "args": {"task_description": "Search for slow-query warnings and measure response times..."}, "id": "call_3"},
        {"name": "delegate_to_worker", "args": {"task_description": "Check for security events: failed logins, unauthorized access..."}, "id": "call_4"},
        {"name": "delegate_to_worker", "args": {"task_description": "Write the final report to output/log_report.md..."}, "id": "call_5"},
    ]
)
```

**Output state** (appended):
```python
{"messages": [HumanMessage(...), AIMessage(tool_calls=[...])]}
```

### Step 2: Routing Decision

`_route_orchestrator` checks the last message:
- `tool_calls` has 5 entries → returns `"execute_workers"`

### Step 3: Orchestrator → Execute Workers

LangGraph enters the `execute_workers` node (a `ToolNode`).

**What happens for each tool call**:
1. `ToolNode` reads all 5 tool calls from the AIMessage
2. For each tool call, it invokes `delegate_to_worker(task_description=...)`
3. Inside each invocation:
   - A fresh `create_agent(model, tools=resolved_tools, middleware=resolved_middleware)` is created
   - The Deep Agents middleware stack injects filesystem tools + todos + summarization automatically
   - The task description is sent as a user message
   - The worker agent runs its internal ReAct loop (may do multiple steps: `read_file → analyze → write_todos → summarize`) with `recursion_limit=200`
   - Only the final message text is extracted and returned
4. Results come back as 5 `ToolMessage` objects

**Output state** (appended):
```python
{"messages": [
    HumanMessage(...),
    AIMessage(tool_calls=[...]),
    ToolMessage(content="Level counts: INFO=25, WARN=8, ERROR=7...", tool_call_id="call_1"),
    ToolMessage(content="ERROR analysis: OutOfMemoryError in PaymentService...", tool_call_id="call_2"),
    ToolMessage(content="Slow queries: 3 warnings, avg 2.3s response time...", tool_call_id="call_3"),
    ToolMessage(content="Security: 47 failed login attempts from 192.168.1.105...", tool_call_id="call_4"),
    ToolMessage(content="Report written to output/log_report.md...", tool_call_id="call_5"),
]}
```

### Step 4: Execute Workers → Synthesizer

The fixed edge `workflow.add_edge("execute_workers", "synthesizer")` fires. **No routing decision**. **No chance to loop back.**

### Step 5: Synthesizer → END

LangGraph enters the `synthesizer` node.

**What happens**:
- System prompt: `"Review the worker execution results and produce a cohesive log analysis report..."`
- The LLM sees all messages: user request, orchestrator plan, all 5 worker results
- Produces a single `AIMessage` with the final merged log analysis report
- No tools are bound, so the synthesizer cannot trigger more work

**Final state**:
```python
{"messages": [
    HumanMessage(...),
    AIMessage(tool_calls=[...]),
    ToolMessage(...), ToolMessage(...), ToolMessage(...), ToolMessage(...), ToolMessage(...),
    AIMessage(content="## Log Analysis Report\n\nHere is your complete analysis: ...")   # ← this is returned
]}
```

### Step 6: Done

`result["messages"][-1].content` contains the final answer.

---

## 7. Deep Agents Middleware Integration

This section explains how the Deep Agents SDK middleware is integrated into the Plan Once architecture.

### The Bridge: `deep_features.py`

`plan_once/deep_features.py` is the **only file** that imports from `deepagents`. It provides a single function: `build_worker_middleware()`.

This function constructs the middleware stack in the same order used by `create_deep_agent` internally:

```
TodoListMiddleware → MemoryMiddleware → SkillsMiddleware →
FilesystemMiddleware → SubAgentMiddleware → SummarizationMiddleware →
PatchToolCallsMiddleware
```

Each middleware is conditionally added based on boolean flags. The stack is returned as a flat `list[Any]` ready to pass to `create_agent(…, middleware=…)`.

### How Middleware Flows into Workers

```
create_plan_once_agent(enable_deep_features=True, backend=...)
    │
    ├── graph.py imports build_worker_middleware from deep_features.py
    ├── Calls build_worker_middleware(model, backend=..., enable_*=...)
    ├── Gets back list of middleware instances
    ├── Passes middleware= to build_worker_tool()
    │
    └── workers.py stores the middleware list
        │
        └── Each delegate_to_worker() call:
            └── create_agent(model, tools=..., middleware=resolved_middleware)
                └── Worker sub-agent has all middleware-provided tools + behaviors
```

### Backend: StateBackend vs FilesystemBackend

| Backend | Workers see | Use case |
|---------|------------|----------|
| `StateBackend` (default) | Empty in-memory filesystem | Unit testing, pure reasoning tasks |
| `FilesystemBackend(root_dir=".")` | Real files on disk | Log file analysis, code review, any task needing real file access |
| `FilesystemBackend(root_dir=".", virtual_mode=False)` | Real files on disk (no overlay) | Recommended — direct access, no deprecation warning |

### What Each Middleware Does to Workers

**`FilesystemMiddleware(backend=...)`**:
- Injects tools: `ls`, `read_file`, `write_file`, `edit_file`, `glob`, `grep`
- These tools operate on the `backend` — if `FilesystemBackend`, they read/write actual files
- Workers can autonomously navigate the filesystem, read logs, and write reports

**`TodoListMiddleware()`**:
- Injects tool: `write_todos`
- Workers can plan their own sub-tasks before executing them
- Useful for complex analysis tasks that require multiple steps

**`SummarizationMiddleware`** (via `create_summarization_middleware(model, backend)`):
- Monitors conversation length
- If the worker's message history gets too long, automatically summarizes older messages
- Prevents context overflow in long-running worker tasks

**`PatchToolCallsMiddleware()`**:
- Intercepts malformed tool calls from the LLM
- Fixes common issues like missing required fields or incorrect argument types
- Acts as a safety net for model output quality

**`MemoryMiddleware(backend=..., sources=[...])`** *(opt-in)*:
- Loads `AGENTS.md` files from the specified paths
- Injects their content into the worker's system prompt
- Provides project-specific context to workers

**`SkillsMiddleware(backend=..., sources=[...])`** *(opt-in)*:
- Loads skill definition files from the specified directories
- Injects them into the worker's system prompt
- Enables specialized behavior patterns

**`SubAgentMiddleware(backend=..., subagents=[...])`** *(opt-in)*:
- Injects a `task` tool
- Workers can spawn their own sub-sub-agents (a general-purpose agent by default)
- Disabled by default to avoid deeply nested agent hierarchies

---

## 8. Dependencies

### Core dependencies (always required)

| Package | Version | What it provides |
|---------|---------|-----------------|
| `langchain-core` | ≥ 1.2.7, < 2 | `BaseMessage`, `BaseChatModel`, `BaseTool`, `StructuredTool` |
| `langchain` | ≥ 1.2.7, < 2 | `create_agent` (used inside workers to create ReAct sub-agents) |
| `langchain-ollama` | ≥ 1.0.0, < 2 | `ChatOllama` (connects to local Ollama instance) |
| `langgraph` | ≥ 0.4.0 | `StateGraph`, `ToolNode`, `START`, `END`, `CompiledStateGraph` |

### Optional dependency (Deep Agents middleware)

| Package | Version | Install command | What it provides |
|---------|---------|----------------|-----------------|
| `deepagents` | ≥ 0.4.7, < 1 | `pip install -e ".[deep]"` | `FilesystemMiddleware`, `TodoListMiddleware`, `SummarizationMiddleware`, `MemoryMiddleware`, `SkillsMiddleware`, `SubAgentMiddleware`, `PatchToolCallsMiddleware`, `StateBackend`, `FilesystemBackend` |

### Test dependency

| Package | Install | What it provides |
|---------|---------|-----------------|
| `pytest` | `pip install -e ".[test]"` or `pip install pytest` | Test runner |

### Model

| Model | Provider | How to load |
|-------|----------|-------------|
| `qwen3-coder:480b-cloud` | Ollama (local) | `ollama pull qwen3-coder:480b-cloud` |

No API keys required — Ollama runs locally.

---

## 9. How to Run

### Install

```bash
cd plan_once_langgraph

# Core only (no Deep Agents)
pip install -e .

# With Deep Agents middleware support
pip install -e ".[deep]"
```

### Run tests (no Ollama needed, no Deep Agents needed)

```bash
pytest -v
```

Expected output: `20 passed`

Tests use `GenericFakeChatModel` — they are deterministic, fast, and don't require any running services.

### Run the log analyzer (needs Ollama + Deep Agents)

```bash
# Make sure Ollama is running and the model is loaded
ollama pull qwen3-coder:480b-cloud

# Default query — comprehensive log analysis
python examples/run_deep.py

# Custom query
python examples/run_deep.py "find all ERROR lines in examples/sample_logs/app.log"
python examples/run_deep.py "count how many WARN vs ERROR entries are in examples/sample_logs/app.log"
python examples/run_deep.py "list all security-related events and recommend countermeasures"
```

### What to expect

The output will look like:

```
========================================================================
LOG ANALYSIS REPORT  (Deep Agents workers)
========================================================================
## Log Analysis Report

### Executive Summary
Analysis of examples/sample_logs/app.log reveals 7 ERROR entries...

### Errors by Component
- PaymentService: OutOfMemoryError, NullPointerException
- DatabasePool: Connection timeout...

### Slow Queries
3 slow-query warnings detected...

### Security Events
47 failed login attempts from 192.168.1.105...

### Recommendations
1. Increase JVM heap allocation for PaymentService
2. Review database connection pool settings
...
```

---

## 10. What's Left to Explore

The core architecture and Deep Agents integration are complete. Here are remaining possibilities:

| Feature | Status | What it would add |
|---------|--------|------------------|
| `enable_memory=True` | Wired, untested | Workers load `AGENTS.md` for project context |
| `enable_skills=True` | Wired, untested | Workers load skill definitions for specialized behavior |
| `enable_subagents=True` | Wired, untested | Workers can spawn sub-sub-agents via `task` tool |
| `HumanInTheLoopMiddleware` | Not wired | Pause for human approval before dangerous operations |
| `AnthropicPromptCachingMiddleware` | Not applicable | Only relevant for Anthropic models (we use Ollama) |
| Shared backend across workers | Available via `FilesystemBackend` | Changes by Worker A are visible to Worker B if they share the same backend instance |
| Worker-specific models | Supported via `worker_model=` | Use a smaller/faster model for workers, larger for orchestrator |
| Streaming output | Not implemented | Stream tokens from the synthesizer as they're generated |
| Parallel execution | Partially — `ToolNode` handles it | True async parallel worker execution (currently sequential in sync mode) |
