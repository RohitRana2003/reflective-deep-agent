"""Tests for the Plan Once graph builder."""

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from plan_once.graph import (
    _DEFAULT_ORCHESTRATOR_PROMPT,
    _DEFAULT_SYNTHESIZER_PROMPT,
    _build_orchestrator_node,
    _build_synthesizer_node,
    _route_orchestrator,
    create_plan_once_agent,
)
from plan_once.state import PlanOnceState
from plan_once.workers import build_worker_tool
from tests.chat_model import GenericFakeChatModel


# -------------------------------------------------------------------
# Routing
# -------------------------------------------------------------------
class TestRouteOrchestrator:
    """Tests for the ``_route_orchestrator`` routing function."""

    def test_routes_to_workers_when_tool_calls_present(self) -> None:
        message = AIMessage(
            content="",
            tool_calls=[
                {"name": "delegate_to_worker", "args": {"task_description": "do something"}, "id": "call_1", "type": "tool_call"}
            ],
        )
        state: PlanOnceState = {"messages": [message]}
        assert _route_orchestrator(state) == "execute_workers"

    def test_routes_to_synthesizer_when_no_tool_calls(self) -> None:
        message = AIMessage(content="Here is the answer directly.")
        state: PlanOnceState = {"messages": [message]}
        assert _route_orchestrator(state) == "synthesizer"

    def test_routes_to_synthesizer_when_empty_tool_calls(self) -> None:
        message = AIMessage(content="Direct answer.", tool_calls=[])
        state: PlanOnceState = {"messages": [message]}
        assert _route_orchestrator(state) == "synthesizer"


# -------------------------------------------------------------------
# Orchestrator node
# -------------------------------------------------------------------
class TestBuildOrchestratorNode:
    """Tests for the orchestrator node builder."""

    def test_orchestrator_invokes_model_with_system_prompt(self) -> None:
        response = AIMessage(content="I will delegate work.")
        model = GenericFakeChatModel(messages=iter([response]))
        worker_tool = build_worker_tool(model, name="delegate_to_worker")

        orchestrator_fn = _build_orchestrator_node(model, worker_tool, system_prompt="You are an orchestrator.")
        state: PlanOnceState = {"messages": [HumanMessage(content="Do something.")]}
        result = orchestrator_fn(state)

        assert "messages" in result
        assert len(result["messages"]) == 1
        assert len(model.call_history) == 1

    def test_orchestrator_returns_model_response(self) -> None:
        response = AIMessage(
            content="Delegating...",
            tool_calls=[{"name": "delegate_to_worker", "args": {"task_description": "task 1"}, "id": "call_1", "type": "tool_call"}],
        )
        model = GenericFakeChatModel(messages=iter([response]))
        worker_tool = build_worker_tool(model, name="delegate_to_worker")

        orchestrator_fn = _build_orchestrator_node(model, worker_tool, system_prompt=_DEFAULT_ORCHESTRATOR_PROMPT)
        state: PlanOnceState = {"messages": [HumanMessage(content="Analyze code.")]}
        result = orchestrator_fn(state)

        returned_message = result["messages"][0]
        assert returned_message.content == "Delegating..."
        assert len(returned_message.tool_calls) == 1


# -------------------------------------------------------------------
# Synthesizer node
# -------------------------------------------------------------------
class TestBuildSynthesizerNode:
    """Tests for the synthesizer node builder."""

    def test_synthesizer_produces_final_response(self) -> None:
        response = AIMessage(content="Here is the combined summary.")
        model = GenericFakeChatModel(messages=iter([response]))

        synthesizer_fn = _build_synthesizer_node(model, system_prompt=_DEFAULT_SYNTHESIZER_PROMPT)

        state: PlanOnceState = {
            "messages": [
                HumanMessage(content="Do two things."),
                AIMessage(content="", tool_calls=[{"name": "delegate_to_worker", "args": {"task_description": "t1"}, "id": "c1", "type": "tool_call"}]),
                ToolMessage(content="Result of task 1", tool_call_id="c1"),
            ]
        }
        result = synthesizer_fn(state)

        assert "messages" in result
        assert result["messages"][0].content == "Here is the combined summary."


# -------------------------------------------------------------------
# Full graph (create_plan_once_agent)
# -------------------------------------------------------------------
class TestCreatePlanOnceAgent:
    """Tests for the top-level ``create_plan_once_agent`` factory."""

    def test_graph_compiles_without_errors(self) -> None:
        model = GenericFakeChatModel(messages=iter([]))
        agent = create_plan_once_agent(model)
        assert agent is not None

    def test_graph_has_expected_nodes(self) -> None:
        model = GenericFakeChatModel(messages=iter([]))
        agent = create_plan_once_agent(model)
        graph = agent.get_graph()
        node_ids = set(graph.nodes)

        assert "orchestrator" in node_ids
        assert "execute_workers" in node_ids
        assert "synthesizer" in node_ids

    def test_direct_answer_path(self) -> None:
        """orchestrator answers directly → synthesizer → END."""
        orchestrator_response = AIMessage(content="Simple answer.")
        synthesizer_response = AIMessage(content="Final: Simple answer.")

        model = GenericFakeChatModel(messages=iter([orchestrator_response, synthesizer_response]))
        agent = create_plan_once_agent(model)

        result = agent.invoke({"messages": [HumanMessage(content="What is 2+2?")]})
        assert result["messages"][-1].content == "Final: Simple answer."

    def test_delegation_path(self) -> None:
        """orchestrator delegates → workers → synthesizer → END."""
        orchestrator_response = AIMessage(
            content="",
            tool_calls=[{"name": "delegate_to_worker", "args": {"task_description": "Summarize asyncio"}, "id": "call_worker_1", "type": "tool_call"}],
        )
        worker_response = AIMessage(content="asyncio is a Python library for async I/O.")
        synthesizer_response = AIMessage(content="Summary: asyncio handles async I/O in Python.")

        model = GenericFakeChatModel(messages=iter([orchestrator_response, worker_response, synthesizer_response]))
        agent = create_plan_once_agent(model)

        result = agent.invoke({"messages": [HumanMessage(content="Explain asyncio.")]})
        assert result["messages"][-1].content == "Summary: asyncio handles async I/O in Python."

    def test_custom_prompts_accepted(self) -> None:
        model = GenericFakeChatModel(messages=iter([]))
        agent = create_plan_once_agent(
            model,
            orchestrator_prompt="Custom orchestrator.",
            synthesizer_prompt="Custom synthesizer.",
            worker_prompt="Custom worker.",
        )
        assert agent is not None

    def test_custom_worker_tools_accepted(self) -> None:
        from langchain_core.tools import tool as tool_decorator

        @tool_decorator
        def dummy_tool(x: str) -> str:
            """A dummy tool."""
            return x

        model = GenericFakeChatModel(messages=iter([]))
        agent = create_plan_once_agent(model, worker_tools=[dummy_tool])
        assert agent is not None

    def test_no_recursion_in_graph_structure(self) -> None:
        """No edge from workers or synthesizer back to orchestrator."""
        model = GenericFakeChatModel(messages=iter([]))
        agent = create_plan_once_agent(model)
        graph = agent.get_graph()

        node_ids = set(graph.nodes)
        assert "orchestrator" in node_ids
        assert "execute_workers" in node_ids
        assert "synthesizer" in node_ids

        for edge in graph.edges:
            source = edge.source if isinstance(edge.source, str) else edge.source.id
            target = edge.target if isinstance(edge.target, str) else edge.target.id
            if source == "execute_workers":
                assert target != "orchestrator", "execute_workers must not route back to orchestrator"
            if source == "synthesizer":
                assert target != "orchestrator", "synthesizer must not route back to orchestrator"
