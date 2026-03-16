"""Plan Once, Execute Once — State definition."""

import operator
from collections.abc import Sequence
from typing import Annotated, NotRequired, TypedDict

from langchain_core.messages import BaseMessage


class PlanOnceState(TypedDict):
    """State schema for the Plan Once, Execute Once graph.

    Uses LangGraph's standard message-appending pattern so that each node
    can add messages to the shared conversation history without overwriting
    previous entries.

    Attributes:
        messages: Accumulated messages flowing through the graph. Each node
            appends its output via ``operator.add``.
        triage_context: Pre-read file contents and complexity assessment
            injected by the triage node.  Used by the orchestrator to make
            intelligent delegation decisions (zero LLM cost).
    """

    messages: Annotated[Sequence[BaseMessage], operator.add]
    triage_context: NotRequired[str]
    file_contents_for_workers: NotRequired[str]
