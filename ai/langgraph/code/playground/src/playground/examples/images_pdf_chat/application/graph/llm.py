from typing import List

from langchain.chat_models import init_chat_model
from langchain_core.language_models import BaseChatModel
from langchain_core.runnables import Runnable
from langchain_core.tools import Tool

from playground.examples.images_pdf_chat.infrastructure.resources.config import Config


class LLM:
    @staticmethod
    def load(tools: List[Tool], config: Config) -> Runnable:
        model: BaseChatModel = init_chat_model(
            model=config.GOOGLE_GENAI_MODEL,
            temperature=config.TEMPERATURE,
        )
        return model.bind_tools(tools)
