from alpha_vantage.foreignexchange import ForeignExchange
from alpha_vantage.timeseries import TimeSeries
from fredapi import Fred
from langchain_core.runnables import RunnableConfig
from langgraph.graph.state import CompiledStateGraph

from src.examples.spy_signals.agents.finance.finance_supervisor import (
    build_finance_supervisor,
)
from src.examples.spy_signals.agents.finance.interest_agent import build_interest_agent
from src.examples.spy_signals.agents.finance.spy_agent import build_spy_agent
from src.examples.spy_signals.agents.finance.usd_agent import build_usd_agent
from src.examples.spy_signals.agents.trading.trading_agent import build_trading_agent
from src.examples.spy_signals.config.agents_config import AgentsConfig
from src.examples.spy_signals.resources.finance_resources import FinanceResources
from src.utils.env_utils import set_env
from src.utils.print_utils import pretty_print_messages


def __init_chat(stocks_graph: CompiledStateGraph, graph_config: RunnableConfig):
    def stream_graph_updates(content: str):
        events = stocks_graph.stream(
            input={"messages": [{"role": "user", "content": content}]},
            config=graph_config,
            stream_mode="values",
        )
        for event in events:
            event["messages"][-1].pretty_print()

    while True:
        user_input = input("User: ")
        if user_input.lower() in ["quit", "exit", "q"]:
            print("Goodbye!")
            break

        stream_graph_updates(user_input)


def main():
    agents_config = AgentsConfig(debug=True)
    debug_mode = agents_config.debug

    finance_resources = FinanceResources(
        time_series=TimeSeries(),
        foreign_exchange=ForeignExchange(),
        fred=Fred(api_key=set_env("FRED_API_KEY")),
    )

    spy_agent = build_spy_agent(
        finance_resources, model=agents_config.spy_agent_model, debug=debug_mode
    )

    # Test the SPY agent
    # for event in spy_agent.stream(
    #     input={"messages": [{"role": "user", "content": "What is the current price of SPY?"}]}
    # ): pretty_print_messages(event, last_message=True)

    usd_agent = build_usd_agent(
        finance_resources, model=agents_config.usd_agent_model, debug=debug_mode
    )

    # Test the USD agent
    for event in usd_agent.stream(
        input={"messages": [{"role": "user", "content": "What is the current USD/EUR rate?"}]}
    ): pretty_print_messages(event, last_message=True)

    interest_agent = build_interest_agent(
        finance=finance_resources,
        model=agents_config.interest_agent_model,
        debug=debug_mode,
    )

    # Test the interest agent
    # for event in interest_agent.stream(
    #     input={
    #         "messages": [
    #             {"role": "user", "content": "What is the current interest rate?"}
    #         ]
    #     }
    # ):
    #     pretty_print_messages(event, last_message=True)

    finance_supervisor = build_finance_supervisor(
        agents=[spy_agent, usd_agent, interest_agent],
        model=agents_config.supervisor_model,
    )

    # Test the finance supervisor
    # for event in supervisor.stream(
    #     input={
    #         "messages": [
    #             {"role": "user", "content": "What is the current interest rate, USD/EUR rate, and SP500 price?"}
    #         ]
    #     }
    # ):
    #     pretty_print_messages(event, last_message=True)

    trading_agent = build_trading_agent(
        model=agents_config.trading_agent_model, debug=debug_mode
    )

    # Test the trading agent
    # for event in trading_agent.stream(
    #     input={
    #         "messages": [
    #             {
    #                 "role": "user",
    #                 "content": "If the current interest rate were (4.48 last month, 4.33 the previous 4 months), the current USD/EUR rate is 0.88, and the current SP500 price is 587.73, what should I do?",
    #             }
    #         ]
    #     }
    # ):
    #     pretty_print_messages(event, last_message=True)

    # runnable_config = RunnableConfig(configurable={"thread_id": uuid.uuid4().hex})
    # __init_chat(finance_supervisor, runnable_config)


if __name__ == "__main__":
    main()
