from langchain_core.language_models import LanguageModelLike
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent

from src.examples.spy_signals.resources.finance_resources import FinanceResources


def build_usd_agent(
    finance: FinanceResources, model: LanguageModelLike, debug: bool = False
):
    @tool
    def get_usd__eur_rate():
        """Get the current USD/EUR rate."""
        exchange_rate = finance.foreign_exchange.get_currency_exchange_rate(
            "USD", "EUR"
        )

        float_exchange_rate = float(exchange_rate[-1]["5. Exchange Rate"])

        print(f"Called get_usd__eur_rate tool: {float_exchange_rate}")
        return float_exchange_rate

    return create_react_agent(
        model=model,
        tools=[get_usd__eur_rate],
        prompt=(
            """
            You are a USD agent that has access to the current USD rate.
            INSTRUCTIONS:
            - Assist ONLY with USD-related tasks, do not perform calculations or use technical analysis
            - Be very detailed, even verbose, and not ambiguous in your queries
            - After you're done with your tasks, respond to the supervisor directly
            - Respond ONLY with the results of your work, do NOT include ANY other text
            """
        ),
        name="usd_agent",
        debug=debug,
    )
