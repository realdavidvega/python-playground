from alpha_vantage.timeseries import TimeSeries
from langchain.chat_models import init_chat_model
from langchain_core.runnables import Runnable
from langgraph.checkpoint.memory import MemorySaver
from pydantic import BaseModel, ConfigDict
from typing_extensions import Callable

from src.examples.stocks_chatbot.graph.tools import tools
from src.examples.stocks_chatbot.resources.alpha_vantage import AlphaVantageResources


class GraphResources(BaseModel):
    chat_model: Runnable
    memory: MemorySaver
    tools: list[Callable]
    model_config = ConfigDict(arbitrary_types_allowed=True)


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
