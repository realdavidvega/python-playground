from langchain_core.messages import ToolMessage
from langchain_core.tools import tool, InjectedToolCallId
from langgraph.prebuilt import InjectedState
from langgraph.types import Command
from typing_extensions import Annotated

from app.resources.alpha_vantage import AlphaVantageResources


def tools(alpha_vantage: AlphaVantageResources):
    @tool
    def get_intraday_data(symbol: str) -> str:
        """Search for a stock symbol and get intraday data."""
        try:
            return alpha_vantage.time_series.get_intraday(symbol)
        except ValueError:
            return f"Oops, we couldn't retrieve the intraday data for {symbol}"

    @tool
    def get_favorite_symbol(state: Annotated[dict, InjectedState]) -> str:
        """Get your favorite stock symbol."""
        favorite_symbol = state.get("favorite_symbol")
        return f"Your favorite symbol is {favorite_symbol}" if favorite_symbol else "You have no favorite symbol set."

    @tool
    def set_favorite_symbol(
            symbol: str,
            tool_call_id: Annotated[str, InjectedToolCallId]
    ) -> Command:
        """Set your favorite stock symbol."""
        if symbol == "":
            response = "You have no favorite symbol set."
        else:
            response = f"Your favorite symbol is now {symbol}."

        state_update = {
            "favorite_symbol": symbol,
            "messages": [ToolMessage(response, tool_call_id=tool_call_id)],
        }

        return Command(update=state_update)

    return [get_intraday_data, get_favorite_symbol, set_favorite_symbol]
