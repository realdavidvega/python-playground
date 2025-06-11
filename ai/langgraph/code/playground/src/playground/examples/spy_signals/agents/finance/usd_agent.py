from langchain_core.language_models import LanguageModelLike
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent

from playground.examples.spy_signals.resources.finance_resources import FinanceResources


def build_usd_agent(
    finance: FinanceResources,
    model: LanguageModelLike,
    debug: bool = False,
    mocked: bool = False,
):
    @tool
    def get_usd__eur_rate() -> list[float]:
        """Get the current USD/EUR rate from last 20 days."""
        if mocked:
            return [
                0.90,
                0.90,
                0.90,
                0.90,
                0.90,
                0.90,
                0.90,
                0.90,
                0.90,
                0.90,
                0.90,
                0.90,
                0.90,
                0.90,
                0.89,
                0.88,
                0.87,
                0.86,
                0.85,
                0.84,
            ]
        else:
            exchange_rate_data = finance.foreign_exchange.get_currency_exchange_rate(
                "USD", "EUR"
            )

            # get last 20 exchange rates
            exchange_rates = [
                rate["5. Exchange Rate"] for rate in exchange_rate_data[:-20]
            ]

            print(f"Called get_usd__eur_rate tool: {exchange_rates}")
            return exchange_rates

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
