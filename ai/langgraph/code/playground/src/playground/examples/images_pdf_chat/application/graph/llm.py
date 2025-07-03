import logging
from dataclasses import dataclass
from logging import Logger
from typing import List, Self

from langchain.chat_models import init_chat_model
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage
from langchain_core.runnables import Runnable
from langchain_core.tools import Tool

from playground.examples.images_pdf_chat.infrastructure.resources.config import (
    Config,
)

logger: Logger = logging.getLogger(__name__)


@dataclass(frozen=True)
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

    def invoke(self, runnable_input: list[BaseMessage]) -> BaseMessage:
        logger.info("Invoking LLM...")
        return self.runnable.invoke(runnable_input)
