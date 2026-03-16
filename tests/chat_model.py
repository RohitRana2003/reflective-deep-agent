"""Fake chat model for testing — self-contained copy for this standalone package."""

import re
from collections.abc import Callable, Iterator, Sequence
from typing import Any, cast

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models import LanguageModelInput
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, AIMessageChunk, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from langchain_core.runnables import Runnable
from langchain_core.tools import BaseTool
from typing_extensions import override


class GenericFakeChatModel(BaseChatModel):
    """Deterministic fake chat model that returns pre-programmed responses.

    Args:
        messages: An iterator over AIMessage objects (use ``iter()`` on a list).
    """

    messages: Iterator[AIMessage | str]
    call_history: list[Any] = []
    stream_delimiter: str | None = None

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)

    def bind_tools(
        self,
        tools: Sequence[dict[str, Any] | type | Callable | BaseTool],
        *,
        tool_choice: str | None = None,
        **kwargs: Any,
    ) -> Runnable[LanguageModelInput, AIMessage]:
        return self

    @override
    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        self.call_history.append({"messages": messages, "kwargs": kwargs})
        response = next(self.messages)
        if isinstance(response, str):
            response = AIMessage(content=response)
        return ChatResult(generations=[ChatGeneration(message=response)])

    @property
    def _llm_type(self) -> str:
        return "generic-fake-chat-model"
