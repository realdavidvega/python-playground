from langchain_core.language_models import LanguageModelLike
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent

from playground.examples.spy_signals.resources.finance_resources import FinanceResources


def build_spy_agent(
    finance: FinanceResources,
    model: LanguageModelLike,
    debug: bool = False,
    mocked: bool = False,
):
    @tool
    def get_daily_data() -> list[float]:
        """Get a list with last 20 close prices from intraday data for SPY (SPDR S&P 500 ETF)."""
        if mocked:
            return [
                585,
                580,
                575,
                570,
                565,
                560,
                555,
                550,
                545,
                540,
                538,
                542,
                547,
                553,
                560,
                568,
                575,
                583,
                590,
                595,
            ]
        else:
            stock_data = finance.time_series.get_daily("SPY", outputsize="compact")[0]
            close_prices = [float(stock_data[date]["4. close"]) for date in stock_data][-20:]
            print("Called get_daily_data tool")
            return close_prices

    @tool
    def get_current_price() -> float:
        """Get close price from intraday data for SPY (SPDR S&P 500 ETF)."""
        if mocked:
            return 585
        else:
            stock_data = finance.time_series.get_daily("SPY", outputsize="compact")[0]
            first_day_key = next(iter(stock_data))
            close_price = float(stock_data[first_day_key]["4. close"])
            print(f"Called get_daily_data tool: {close_price}")
            return close_price

    return create_react_agent(
        model=model,
        tools=[get_daily_data],
        prompt=(
            """
            You are a spy agent that has access to intraday data for SPY.
            INSTRUCTIONS:
            - Assist ONLY with SPY (SPDR S&P 500 ETF) price related tasks, do not perform calculations or use technical analysis
            - Be very detailed, even verbose, and not ambiguous in your queries
            - After you're done with your tasks, respond to the supervisor directly
            - Respond ONLY with the results of your work, do NOT include ANY other text
            """
        ),
        name="spy_agent",
        debug=debug,
    )
