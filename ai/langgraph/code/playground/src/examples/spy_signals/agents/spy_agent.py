from langchain_core.language_models import LanguageModelLike
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent

from src.examples.spy_signals.resources.finance_resources import FinanceResources


def build_spy_agent(
    finance: FinanceResources, model: LanguageModelLike, debug: bool = False
):
    @tool
    def get_daily_data():
        """Get intraday data for SPY (SPDR S&P 500 ETF)."""
        daily_data = finance.time_series.get_daily("SPY", outputsize="compact")

        print(f"Called get_daily_data tool: {daily_data}")
        return daily_data

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
