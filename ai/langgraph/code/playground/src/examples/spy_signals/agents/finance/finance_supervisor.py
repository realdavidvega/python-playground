from langchain_core.language_models import LanguageModelLike
from langgraph.graph.graph import CompiledGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph_supervisor import create_supervisor


def build_finance_supervisor(
    agents: list[CompiledGraph],
    model: LanguageModelLike,
) -> CompiledStateGraph:
    return create_supervisor(
        agents=agents,
        model=model,
        prompt=(
            """
            You are a supervisor managing the following agents:
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
