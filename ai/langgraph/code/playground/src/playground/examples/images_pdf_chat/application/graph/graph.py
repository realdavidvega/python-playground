from typing import List

from langchain_core.tools import Tool
from langgraph.checkpoint.memory import MemorySaver
from langgraph.constants import START, END
from langgraph.graph.state import CompiledStateGraph, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

from playground.examples.images_pdf_chat.application.graph.nodes import ChatNode
from playground.examples.images_pdf_chat.application.graph.state import State
from playground.examples.images_pdf_chat.infrastructure.resources.config import Config


class Graph:
    @staticmethod
    def compile(tools: List[Tool], config: Config) -> CompiledStateGraph:
        graph_builder = StateGraph(State)
        chat_node = ChatNode(tools, config)

        graph_builder.add_node("chat_node", chat_node)
        graph_builder.add_node("tool_node", ToolNode(tools=tools))

        graph_builder.add_edge(START, "chat_node")
        graph_builder.add_conditional_edges(
            "chat_node", tools_condition, {"tools": "tool_node", END: END}
        )
        graph_builder.add_edge("tool_node", "chat_node")

        return graph_builder.compile(checkpointer=MemorySaver())
