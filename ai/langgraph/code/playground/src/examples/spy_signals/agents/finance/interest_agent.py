from langchain_core.language_models import LanguageModelLike
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent

from src.examples.spy_signals.resources.finance_resources import FinanceResources


def build_interest_agent(
    finance: FinanceResources, model: LanguageModelLike, debug: bool = False
):
    @tool
    def get_interest_rates() -> list[float]:
        """Get the current FED interest rates from last 5 months."""
        return finance.fred.get_series("FEDFUNDS")[-5:]

    return create_react_agent(
        model=model,
        tools=[get_interest_rates],
        prompt=(
            """
            You are an interest agent that has access to the treasury's current interest rate for a given period.
            By default, the start_date is the first day of the current month and the end_date is today.
            INSTRUCTIONS:
            - Assist ONLY with interest rate-related tasks, do not perform calculations or use technical analysis
            - Be very detailed, even verbose, and not ambiguous in your queries
            - After you're done with your tasks, respond to the supervisor directly
            - Respond ONLY with the results of your work, do NOT include ANY other text
            """
        ),
        name="interest_agent",
        debug=debug,
    )
