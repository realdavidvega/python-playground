from alpha_vantage.timeseries import TimeSeries
from langchain.chat_models import init_chat_model
from langchain_core.runnables import Runnable
from langgraph.checkpoint.memory import MemorySaver
from pydantic import ConfigDict
from pydantic.dataclasses import dataclass
from typing_extensions import Callable

from playground.examples.stocks_chatbot.graph.tools import tools
from playground.examples.stocks_chatbot.resources.alpha_vantage import (
    AlphaVantageResources,
)


@dataclass(config=ConfigDict(arbitrary_types_allowed=True, frozen=True))
class GraphResources:
    chat_model: Runnable
    memory: MemorySaver
    tools: list[Callable]


def init_graph_resources(
    model: str = "google_genai:gemini-2.0-flash",
) -> GraphResources:
    alpha_vantage = AlphaVantageResources(time_series=TimeSeries())
    llm_tools = tools(alpha_vantage)
    llm = init_chat_model(model, temperature=0)

    return GraphResources(
        chat_model=llm.bind_tools(llm_tools),
        memory=MemorySaver(),
        tools=llm_tools,
    )
