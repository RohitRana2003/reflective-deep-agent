# Plan Once, Execute Once — Log Analyzer

A **standalone** implementation of the "Plan Once, Execute Once" architecture described in
[`plan_once_architecture.md`](../plan_once_architecture.md), themed as a **log analyzer**
and powered by **Ollama** (`qwen3-coder:480b-cloud`).

Workers can optionally run with the full **Deep Agents middleware stack** — gaining filesystem
tools, todo planning, context summarization, memory, and skills — without hand-rolling
`@tool` functions.


---

## Architecture

```
START ─► Orchestrator ─► Workers (parallel) ─► Synthesizer ─► END
```

Workers route to the **synthesizer**, never back to the orchestrator.
This makes the graph a **directed acyclic graph (DAG)** — infinite loops are architecturally impossible.

### Nodes

| Node | Role |
|------|------|
| **Orchestrator** | Single LLM turn. Analyzes the log-analysis request, decomposes it into independent sub-tasks, emits parallel `delegate_to_worker` tool calls. |
| **Workers** | `ToolNode` executes all tool calls in parallel. Each spins up an independent `create_agent` sub-agent with its own context window. When `enable_deep_features=True`, workers get the full Deep Agents middleware stack (filesystem, todos, summarization, etc.). |
| **Synthesizer** | Sees the original request + all worker results. Produces one cohesive log analysis report. No tools bound — can only produce text. |

### How it maps to Deep Agents

| Deep Agents concept | Plan Once equivalent |
|---------------------|---------------------|
| `SubAgentMiddleware` / `task` tool | `build_worker_tool` / `delegate_to_worker` |
| Agent ReAct loop | **Eliminated** — DAG enforces single pass |
| `create_deep_agent` | `create_plan_once_agent` |
| `FilesystemMiddleware` + `StateBackend` | `enable_deep_features=True` + `FilesystemBackend` |
| `TodoListMiddleware` | `enable_todos=True` (on by default with deep features) |
| `SummarizationMiddleware` | `enable_summarization=True` (on by default) |
| `MemoryMiddleware` / `SkillsMiddleware` | `enable_memory=True` / `enable_skills=True` (opt-in) |

---

## Folder structure

```
plan_once_langgraph/
├── pyproject.toml                 # standalone project config + optional [deep] extra
├── README.md                      # this file
├── WALKTHROUGH.md                 # full end-to-end technical documentation
│
├── plan_once/                     # core library (5 files)
│   ├── __init__.py                # public API exports
│   ├── state.py                   # PlanOnceState TypedDict
│   ├── workers.py                 # build_worker_tool factory (supports middleware)
│   ├── graph.py                   # create_plan_once_agent DAG builder
│   └── deep_features.py           # Deep Agents middleware bridge
│
├── tests/                         # unit tests (20 tests)
│   ├── __init__.py
│   ├── chat_model.py              # GenericFakeChatModel for deterministic testing
│   ├── test_graph.py              # 13 tests: routing, nodes, end-to-end graph
│   └── test_workers.py            # 7 tests: worker tool factory + state
│
└── examples/                      # runnable log-analysis POC
    ├── run_deep.py                # Deep Agents middleware-powered log analyzer
    └── sample_logs/
        └── app.log                # sample application log file
```

---

## Quick start

### Install (core only — no Deep Agents dependency)

```bash
cd plan_once_langgraph
pip install -e .
```

### Install with Deep Agents middleware

```bash
pip install -e ".[deep]"
```

This pulls in `deepagents>=0.4.7` which provides `FilesystemMiddleware`,
`TodoListMiddleware`, `SummarizationMiddleware`, and more.

### Run tests (no Ollama needed)

```bash
pytest -v
# Expected: 20 passed
```

### Run the log analyzer (needs Ollama)

```bash
# Make sure Ollama is running with the model loaded
ollama pull qwen3-coder:480b-cloud

# Default query — comprehensive log analysis
python examples/run_deep.py

# Custom query via CLI argument
python examples/run_deep.py "find all ERROR lines in examples/sample_logs/app.log"
```

---

## Usage in your own code

### Basic — no Deep Agents middleware

```python
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage
from plan_once import create_plan_once_agent

model = ChatOllama(model="qwen3-coder:480b-cloud")
agent = create_plan_once_agent(model)

result = agent.invoke({
    "messages": [HumanMessage(content="Analyze these logs for errors...")]
})
print(result["messages"][-1].content)
```

### With Deep Agents middleware (filesystem, todos, summarization)

```python
from deepagents.backends import FilesystemBackend
from plan_once import create_plan_once_agent

agent = create_plan_once_agent(
    enable_deep_features=True,
    backend=FilesystemBackend(root_dir=".", virtual_mode=False),
    # Feature flags (all True by default when deep is on):
    enable_filesystem=True,       # ls, read_file, write_file, edit_file, glob, grep
    enable_todos=True,            # write_todos — workers can plan sub-tasks
    enable_summarization=True,    # auto-summarise long conversations
    # Opt-in features:
    enable_memory=False,          # load AGENTS.md files
    enable_skills=False,          # load skill definitions
    enable_subagents=False,       # workers spawn sub-sub-agents
)
```

### With custom worker tools (no Deep Agents)

```python
agent = create_plan_once_agent(
    model,
    worker_tools=[my_search_tool, my_fs_tool],
)
```

### Custom prompts

```python
agent = create_plan_once_agent(
    model,
    orchestrator_prompt="You are a code review orchestrator...",
    synthesizer_prompt="Summarize all review findings...",
    worker_prompt="You are a code reviewer. Focus on...",
)
```

---

## Dependencies

| Package | Purpose |
|---------|---------|
| `langchain-core` | Messages, base model/tool classes |
| `langchain` | `create_agent` for worker sub-agents |
| `langchain-ollama` | `ChatOllama` — connects to local Ollama |
| `langgraph` | `StateGraph`, `ToolNode`, `START`, `END` |
| `deepagents` *(optional)* | Middleware stack: filesystem, todos, summarization, memory, skills |

### Model

| Model | Provider | Load command |
|-------|----------|-------------|
| `qwen3-coder:480b-cloud` | Ollama (local) | `ollama pull qwen3-coder:480b-cloud` |

---

## Public API

| Symbol | Type | Description |
|--------|------|-------------|
| `create_plan_once_agent` | Function | Main factory — builds and compiles the DAG |
| `PlanOnceState` | TypedDict | The graph state schema (`messages` list) |
| `build_worker_tool` | Function | Creates the `delegate_to_worker` tool |
| `build_worker_middleware` | Function / `None` | Builds the Deep Agents middleware stack (gracefully `None` if deepagents is not installed) |

---

## Further reading

See [WALKTHROUGH.md](WALKTHROUGH.md) for a complete end-to-end technical breakdown — every file,
every function, execution flow diagrams, and detailed Deep Agents middleware integration docs.
