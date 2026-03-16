"""Tests for the Plan Once worker tool factory."""

from langchain_core.messages import AIMessage

from plan_once.workers import build_worker_tool
from tests.chat_model import GenericFakeChatModel


class TestBuildWorkerTool:
    """Tests for ``build_worker_tool``."""

    def test_returns_structured_tool(self) -> None:
        model = GenericFakeChatModel(messages=iter([]))
        tool = build_worker_tool(model)
        assert tool.name == "delegate_to_worker"

    def test_custom_name(self) -> None:
        model = GenericFakeChatModel(messages=iter([]))
        tool = build_worker_tool(model, name="my_worker")
        assert tool.name == "my_worker"

    def test_tool_has_description(self) -> None:
        model = GenericFakeChatModel(messages=iter([]))
        tool = build_worker_tool(model)
        assert "delegate" in tool.description.lower() or "worker" in tool.description.lower()

    def test_tool_invocation_returns_string(self) -> None:
        worker_response = AIMessage(content="Task completed successfully.")
        model = GenericFakeChatModel(messages=iter([worker_response]))
        tool = build_worker_tool(model)
        result = tool.invoke({"task_description": "Do something simple."})
        assert isinstance(result, str)
        assert "Task completed successfully." in result

    def test_tool_with_custom_prompt(self) -> None:
        worker_response = AIMessage(content="Done with custom prompt.")
        model = GenericFakeChatModel(messages=iter([worker_response]))
        tool = build_worker_tool(model, worker_prompt="You are a specialized code analyst.")
        result = tool.invoke({"task_description": "Analyze this code."})
        assert "Done with custom prompt." in result

    def test_tool_with_worker_tools(self) -> None:
        from langchain_core.tools import tool as tool_decorator

        @tool_decorator
        def helper(x: str) -> str:
            """Helper tool."""
            return f"helped: {x}"

        worker_response = AIMessage(content="Used helper tool.")
        model = GenericFakeChatModel(messages=iter([worker_response]))
        tool = build_worker_tool(model, worker_tools=[helper])
        result = tool.invoke({"task_description": "Use the helper."})
        assert isinstance(result, str)


class TestPlanOnceState:
    """Tests for the ``PlanOnceState`` definition."""

    def test_state_accepts_messages(self) -> None:
        from plan_once.state import PlanOnceState

        state: PlanOnceState = {"messages": [AIMessage(content="hello")]}
        assert len(state["messages"]) == 1
