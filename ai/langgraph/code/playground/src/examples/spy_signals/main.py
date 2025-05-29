import uuid

from alpha_vantage.foreignexchange import ForeignExchange
from alpha_vantage.timeseries import TimeSeries
from fredapi import Fred
from langchain_core.runnables import RunnableConfig
from langgraph.graph.state import CompiledStateGraph
from langgraph_supervisor import create_supervisor

from src.examples.spy_signals.agents.interest_agent import build_interest_agent
from src.examples.spy_signals.agents.spy_agent import build_spy_agent
from src.examples.spy_signals.agents.usd_agent import build_usd_agent
from src.examples.spy_signals.config.agents_config import AgentsConfig
from src.examples.spy_signals.resources.finance_resources import FinanceResources
from src.utils.env_utils import set_env


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
    # for event in usd_agent.stream(
    #     input={"messages": [{"role": "user", "content": "What is the current USD/EUR rate?"}]}
    # ): pretty_print_messages(event, last_message=True)

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

    runnable_config = RunnableConfig(configurable={"thread_id": uuid.uuid4().hex})

    supervisor = create_supervisor(
        agents=[spy_agent, usd_agent, interest_agent],
        model=agents_config.supervisor_model,
        prompt=(
            """
            You are a supervisor managing three agents:
            - a SP500 (SPY) agent. Assign SPY-related tasks to this agent
            - a USD/EUR agent. Assign USD/EUR prices related tasks to this agent
            - an interest agent. Assign interest rate-related tasks to this agent
            Assign work to one agent at a time, do not call agents in parallel
            Be very detailed, even verbose, and not ambiguous when assigning tasks
            Do not do any work yourself, do not call any agent with empty parameters
            Once you're done with your tasks, respond directly with the results to the user
            """
        ),
    ).compile()

    __init_chat(supervisor, runnable_config)


if __name__ == "__main__":
    main()
