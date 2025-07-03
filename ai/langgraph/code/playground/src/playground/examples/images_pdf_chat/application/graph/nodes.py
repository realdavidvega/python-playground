from dataclasses import dataclass

from playground.examples.images_pdf_chat.application.graph.llm import LLM
from playground.examples.images_pdf_chat.application.graph.state import State



@dataclass(frozen=True)
class ChatNode:
    llm: LLM

    def __call__(self, state: State) -> State:
        state["messages"].extend([self.llm.invoke(state["messages"])])
        return state
