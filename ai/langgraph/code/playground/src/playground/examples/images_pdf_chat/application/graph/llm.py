from typing import List, Self, TypeVar

from langchain.chat_models import init_chat_model
from langchain_core.language_models import BaseChatModel
from langchain_core.runnables import Runnable
from langchain_core.tools import Tool
from pydantic.dataclasses import dataclass

from playground.examples.images_pdf_chat.infrastructure.resources.config import (
    Config,
    BASE_CONFIG,
)

Input = TypeVar("Input", contravariant=True)
Output = TypeVar("Output", covariant=True)


@dataclass(config=BASE_CONFIG)
class LLM:
    runnable: Runnable

    @classmethod
    def create(cls, tools: List[Tool], config: Config) -> Self:
        model: BaseChatModel = init_chat_model(
            model=config.GOOGLE_GENAI_MODEL,
            temperature=config.TEMPERATURE,
        )
        runnable: Runnable = model.bind_tools(tools)
        return cls(runnable=runnable)

    def invoke(self, runnable_input: Input) -> Output:
        return self.runnable.invoke(runnable_input)
