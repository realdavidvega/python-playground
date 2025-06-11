from langchain_core.language_models import LanguageModelLike
from langgraph.graph.state import CompiledStateGraph
from langgraph.pregel import Pregel
from langgraph_supervisor import create_supervisor


def build_signal_supervisor(
    agents: list[Pregel],
    model: LanguageModelLike,
) -> CompiledStateGraph:
    return create_supervisor(
        agents=agents,
        model=model,
        prompt=(
            """
            You are a supervisor managing the following agents:
            - a finance supervisor. Assign finance-related tasks to this agent, like prices and interest rates
            - a trading agent. Assign trading-related tasks to this agent, like generating signals
            Assign work to one agent at a time, do not call agents in parallel
            Be very detailed, even verbose, and not ambiguous when assigning tasks
            Do not do any work yourself, do not call any agent with empty parameters
            Once you're done with your tasks, respond directly with the results to the user
            """
        ),
    ).compile(name="signal_supervisor")
