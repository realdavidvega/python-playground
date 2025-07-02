from typing import List

from langchain_core.tools import Tool

from playground.examples.images_pdf_chat.application.graph.llm import LLM
from playground.examples.images_pdf_chat.application.graph.state import State
from playground.examples.images_pdf_chat.infrastructure.resources.config import Config


class ChatNode:
    def __init__(self, tools: List[Tool], config: Config) -> None:
        self.llm = LLM.load(tools, config)

    def __call__(self, state: State) -> State:
        state["messages"].extend([self.llm.invoke(state["messages"])])
        return state
