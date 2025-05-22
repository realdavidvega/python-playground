from alpha_vantage.timeseries import TimeSeries
from langchain.chat_models import init_chat_model
from langchain_core.runnables import Runnable
from langgraph.checkpoint.memory import MemorySaver
from pydantic import BaseModel, ConfigDict
from typing_extensions import Callable

from app.resources.alpha_vantage import AlphaVantageResources
from app.graph.tools import tools


class Resources(BaseModel):
    chat_model: Runnable
    memory: MemorySaver
    tools: list[Callable]
    model_config = ConfigDict(arbitrary_types_allowed=True)


def resources(model: str = "google_genai:gemini-2.0-flash") -> Resources:
    alpha_vantage = AlphaVantageResources(time_series=TimeSeries())
    llm_tools = tools(alpha_vantage)
    llm = init_chat_model(model, temperature=0)

    return Resources(
        chat_model=llm.bind_tools(llm_tools),
        memory=MemorySaver(),
        tools=llm_tools,
    )
