from langgraph.constants import START
from langgraph.graph import add_messages
from langgraph.graph.state import CompiledStateGraph, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition
from pydantic import BaseModel, Field
from typing_extensions import Annotated

from app.resources import Resources


class StockMarketState(BaseModel):
    messages: Annotated[list, add_messages] = Field(default_factory=list)
    symbol: str = Field(default="AAPL")


def graph(resources: Resources) -> CompiledStateGraph:
    def __chatbot(graph_state: StockMarketState):
        return {"messages": [resources.chat_model.invoke(graph_state.messages)]}

    graph_builder = StateGraph(StockMarketState)

    graph_builder.add_node("chatbot", __chatbot)

    tool_node = ToolNode(tools=resources.tools)
    graph_builder.add_node("tools", tool_node)

    graph_builder.add_conditional_edges(
        "chatbot",
        tools_condition,
    )

    graph_builder.add_edge("tools", "chatbot")
    graph_builder.add_edge(START, "chatbot")

    return graph_builder.compile(checkpointer=resources.memory)
