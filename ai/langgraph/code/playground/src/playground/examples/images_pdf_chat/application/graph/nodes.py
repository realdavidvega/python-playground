import sys
from dataclasses import dataclass

from playground.examples.images_pdf_chat.application.graph.llm import LLM
from playground.examples.images_pdf_chat.application.graph.state import State

CHAT_NODE: str = sys.intern("chat_node")
TOOL_NODE: str = sys.intern("tool_node")
TOOLS: str = sys.intern("tools")
CONFIRMATION: str = sys.intern("confirmation")


@dataclass(frozen=True)
class ChatNode:
    llm: LLM

    def __call__(self, state: State) -> State:
        state["messages"].extend([self.llm.invoke(state["messages"])])
        return state
