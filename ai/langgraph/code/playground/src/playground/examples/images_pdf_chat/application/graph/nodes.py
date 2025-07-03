from pydantic.dataclasses import dataclass

from playground.examples.images_pdf_chat.application.graph.llm import LLM
from playground.examples.images_pdf_chat.application.graph.state import State
from playground.examples.images_pdf_chat.infrastructure.resources.config import (
    BASE_CONFIG,
)


@dataclass(config=BASE_CONFIG)
class ChatNode:
    llm: LLM

    def __call__(self, state: State) -> State:
        state["messages"].extend([self.llm.invoke(state["messages"])])
        return state
