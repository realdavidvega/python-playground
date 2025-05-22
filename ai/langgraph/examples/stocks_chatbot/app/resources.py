from alpha_vantage.timeseries import TimeSeries
from langchain.chat_models import init_chat_model
from langchain_core.runnables import Runnable
from langchain_core.tools import tool
from langgraph.checkpoint.memory import MemorySaver
from pydantic import BaseModel, ConfigDict
from typing_extensions import Callable

class AlphaVantageResources(BaseModel):
    time_series: TimeSeries
    model_config = ConfigDict(arbitrary_types_allowed=True)

class Resources(BaseModel):
    chat_model: Runnable
    memory: MemorySaver
    tools: list[Callable]
    model_config = ConfigDict(arbitrary_types_allowed=True)


def resources() -> Resources:
    @tool
    def __get_intraday_data(symbol: str) -> str:
        """Search for a stock symbol and get intraday data."""
        return alpha_vantage.time_series.get_intraday(symbol)

    alpha_vantage = AlphaVantageResources(time_series=TimeSeries())
    get_intraday_tool = __get_intraday_data
    llm_tools = [get_intraday_tool]
    llm = init_chat_model("google_genai:gemini-2.0-flash", temperature=0)

    return Resources(
        chat_model=llm.bind_tools(llm_tools),
        memory=MemorySaver(),
        tools=llm_tools,
    )
