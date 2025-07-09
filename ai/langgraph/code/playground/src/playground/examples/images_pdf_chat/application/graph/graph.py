import logging
from logging import Logger
from typing import List

from langchain_core.tools import Tool
from langgraph.checkpoint.memory import MemorySaver
from langgraph.constants import START, END
from langgraph.graph.state import CompiledStateGraph, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

from playground.examples.images_pdf_chat.application.graph.llm import LLM
from playground.examples.images_pdf_chat.application.graph.nodes import (
    ChatNode,
    CHAT_NODE,
    TOOL_NODE,
    TOOLS,
)
from playground.examples.images_pdf_chat.application.graph.state import State
from playground.examples.images_pdf_chat.infrastructure.resources.config import (
    Config,
)

logger: Logger = logging.getLogger(__name__)


class Graph:
    @staticmethod
    def load(tools: List[Tool], config: Config) -> CompiledStateGraph:
        logger.info("Loading graph...")
        graph_builder: StateGraph = StateGraph(State)

        llm: LLM = LLM.create(tools, config)
        graph_builder.add_node(CHAT_NODE, ChatNode(llm=llm))
        graph_builder.add_node(TOOL_NODE, ToolNode(tools=tools))

        graph_builder.add_edge(START, CHAT_NODE)
        graph_builder.add_conditional_edges(CHAT_NODE, tools_condition, {TOOLS: TOOL_NODE, END: END})
        graph_builder.add_edge(TOOL_NODE, CHAT_NODE)

        return graph_builder.compile(checkpointer=MemorySaver())
