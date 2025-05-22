from langchain_core.tools import tool

from app.resources.alpha_vantage import AlphaVantageResources


@tool
def get_intraday(alpha_vantage: AlphaVantageResources, symbol: str) -> str:
    """Search for a stock symbol and get intraday data."""
    return alpha_vantage.time_series.get_intraday(symbol)
