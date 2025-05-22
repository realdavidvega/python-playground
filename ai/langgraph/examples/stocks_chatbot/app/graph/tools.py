from langchain_core.tools import tool

from app.resources.alpha_vantage import AlphaVantageResources


def tools(alpha_vantage: AlphaVantageResources):
    @tool
    def get_intraday_data(symbol: str) -> str:
        """Search for a stock symbol and get intraday data."""
        try:
            return alpha_vantage.time_series.get_intraday(symbol)
        except ValueError:
            return f"Oops, we couldn't retrieve the intraday data for {symbol}"

    return [get_intraday_data]
