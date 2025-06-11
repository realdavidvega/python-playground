# Multi-agent supervisor
# This architecture specializes on agents coordinated by a central supervisor agent.
# The supervisor agent controls all communication flow and task delegation, making decisions about which agent to
# invoke based on the current context and task requirements.

from langchain.chat_models import init_chat_model
from langchain_core.tools import InjectedToolCallId, tool
from langchain_tavily import TavilySearch
from langgraph.constants import END, START
from langgraph.graph import MessagesState, StateGraph
from langgraph.graph.graph import CompiledGraph
from langgraph.prebuilt import InjectedState, create_react_agent
from langgraph.pregel import Pregel
from langgraph.types import Command, Send
from langgraph_supervisor import create_supervisor
from typing_extensions import Annotated

from playground.utils.env_utils import set_env
from playground.utils.print_utils import pretty_print_messages


# Create worker agents
# Research agent will have access to a web search tool using Tavily API
def _create_research_agent(model: str, search_tool: TavilySearch) -> CompiledGraph:
    return create_react_agent(
        model=init_chat_model(model=model, temperature=0),
        tools=[search_tool],
        prompt=(
            """
            You are a research agent
            INSTRUCTIONS:
            - Assist ONLY with research-related tasks, DO NOT do any math
            - Be very detailed, even verbose, and not ambiguous in your queries
            - After you're done with your tasks, respond to the supervisor directly
            - Respond ONLY with the results of your work, do NOT include ANY other text
            """
        ),
        name="research_agent",
        debug=True,
    )


# Math agent will have access to simple math tools (add, multiply, divide)
def _create_math_agent(model: str) -> CompiledGraph:
    def add(a: float, b: float):
        """Add two numbers."""
        return a + b

    def multiply(a: float, b: float):
        """Multiply two numbers."""
        return a * b

    def divide(a: float, b: float):
        """Divide two numbers."""
        return a / b

    return create_react_agent(
        model=init_chat_model(model=model, temperature=0),
        tools=[add, multiply, divide],
        prompt=(
            """
            You are a math agent
            INSTRUCTIONS:
            - Assist ONLY with math-related tasks
            - Be exact and not ambiguous in your inputs
            - After you're done with your tasks, respond to the supervisor directly
            - Respond ONLY with the results of your work, do NOT include ANY other text
            """
        ),
        name="math_agent",
        debug=True,
    )


# Create supervisor with langgraph-supervisor
# To implement out multi-agent system, we will use create_supervisor from the prebuilt langgraph-supervisor library
def _create_supervisor(model: str, agents: list[Pregel]) -> CompiledGraph:
    return create_supervisor(
        model=init_chat_model(model=model, temperature=0),
        agents=agents,
        prompt=(
            """
            You are a supervisor managing two agents:
            - a research agent. Assign research-related tasks to this agent
            - a math agent. Assign math-related tasks to this agent
            Assign work to one agent at a time, do not call agents in parallel
            Be very detailed, even verbose, and not ambiguous when assigning tasks
            Do not do any work yourself, do not call any agent with empty parameters
            If you couldn't do the task, respond precisely why and what agents didn't do the task as expected
            Once you're done with your tasks, respond directly with the results
            """
        ),
        add_handoff_back_messages=True,
        output_mode="full_history",
    ).compile()


# Create supervisor from scratch
# Let's now implement this same multi-agent system from scratch. We will need to:
# - Set up how the supervisor communicates with individual agents
# - Create the supervisor agent
# - Combine supervisor and worker agents into a single multi-agent graph.
def _run_multi_agent_supervisor(model: str):
    # Set up agent communication
    # We will need to define a way for the supervisor agent to communicate with the worker agents.
    # A common way to implement this in multi-agent architectures is using handoffs, where one agent hands off
    # control to another. Handoffs allow you to specify:
    # - destination: target agent to transfer to
    # - payload: information to pass to that agent
    # We will implement handoffs via handoff tools and give these tools to the supervisor agent:
    # when the supervisor calls these tools, it will hand off control to a worker agent, passing the full message
    # history to that agent.
    def create_handoff_tool(*, agent_name: str, description: str | None = None):
        name = f"transfer_to_{agent_name}"
        description = description or f"Ask {agent_name} for help."

        @tool(name, description=description)
        def handoff_tool(
            state: Annotated[MessagesState, InjectedState],
            tool_call_id: Annotated[str, InjectedToolCallId],
        ) -> Command:
            tool_message = {
                "role": "tool",
                "content": f"Successfully transferred to {agent_name}",
                "name": name,
                "tool_call_id": tool_call_id,
            }

            # Pass the full message history to the worker agent
            return Command(
                goto=Send(agent_name, state),
                update={**state, "messages": state["messages"] + [tool_message]},
                graph=Command.PARENT,
            )

        return handoff_tool

    def research_agent(state):
        agent = _create_research_agent(
            model=model,
            search_tool=TavilySearch(
                max_results=3,
                search_depth="advanced",
                topic="finance",
                time_range="year",
            ),
        )
        response = agent.invoke(state)
        return MessagesState(
            messages=[{"role": "user", "content": response["messages"][-1].content}]
        )

    def math_agent(state):
        agent = _create_math_agent(model=model)
        response = agent.invoke(state)
        return MessagesState(
            messages=[{"role": "user", "content": response["messages"][-1].content}]
        )

    # Handoffs
    assign_to_research_agent = create_handoff_tool(
        agent_name="research_agent",
        description="Assign task to a researcher agent.",
    )

    assign_to_math_agent = create_handoff_tool(
        agent_name="math_agent",
        description="Assign task to a math agent.",
    )

    # Supervisor
    # Then, let's create the supervisor agent with the handoff tools we just defined.
    # We will use the prebuilt create_react_agent
    supervisor_agent = create_react_agent(
        model=model,
        tools=[assign_to_research_agent, assign_to_math_agent],
        prompt=(
            """
            You are a supervisor managing two agents:
            - a research agent. Assign research-related tasks to this agent
            - a math agent. Assign math-related tasks to this agent
            Assign work to one agent at a time, do not call agents in parallel
            Be very detailed, even verbose, and not ambiguous when assigning tasks
            Do not do any work yourself, do not call any agent with empty parameters
            If you couldn't do the task, respond precisely why and what agents didn't do the task as expected
            Once you're done with your tasks, respond directly with the results
            """
        ),
        name="supervisor",
    )

    # Create multi-agent graph
    # Putting this all together, let's create a graph for our overall multi-agent system.
    # We will add the supervisor and the individual agents as subgraph nodes.
    supervisor_graph = (
        StateGraph(MessagesState)
        # NOTE: `destinations` is only needed for visualization and doesn't affect runtime behavior
        .add_node(supervisor_agent, destinations=("research_agent", "math_agent", END))
        .add_node(research_agent)
        .add_node(math_agent)
        .add_edge(START, "supervisor")
        # always return back to the supervisor
        .add_edge("research_agent", "supervisor")
        .add_edge("math_agent", "supervisor")
        .compile()
    )

    # Notice that we've added explicit edges from worker agents back to the supervisor — this means that they are
    # guaranteed to return control back to the supervisor. If you want the agents to respond directly to the
    # user (i.e., turn the system into a router, you can remove these edges).

    for chunk in supervisor_graph.stream(
        {
            "messages": [
                {
                    "role": "user",
                    "content": "find the current exchange rate of USD to EUR, then exchange 100 USD to EUR",
                }
            ]
        },
    ):
        pretty_print_messages(chunk, last_message=True)


# Create delegation tasks
# So far the individual agents relied on interpreting full message history to determine their tasks.
# An alternative approach is to ask the supervisor to formulate a task explicitly.
# We can do so by adding a task_description parameter to the handoff_tool function.
def _run_multi_agent_supervisor_delegating(model: str):
    def create_task_description_handoff_tool(
        *, agent_name: str, description: str | None = None
    ):
        name = f"transfer_to_{agent_name}"
        description = description or f"Ask {agent_name} for help."

        @tool(name, description=description)
        def handoff_tool(
            # this is populated by the supervisor LLM
            task_description: Annotated[
                str,
                "Description of what the next agent should do, including all of the relevant context.",
            ],
            # these parameters are ignored by the LLM
            state: Annotated[MessagesState, InjectedState],
        ) -> Command:
            task_description_message = {"role": "user", "content": task_description}
            agent_input = {**state, "messages": [task_description_message]}
            return Command(
                goto=[Send(agent_name, agent_input)],
                graph=Command.PARENT,
            )

        return handoff_tool

    def research_agent(state):
        agent = _create_research_agent(
            model=model,
            search_tool=TavilySearch(
                max_results=3,
                search_depth="advanced",
                topic="finance",
                time_range="year",
            ),
        )
        response = agent.invoke(state)
        return MessagesState(
            messages=[{"role": "user", "content": response["messages"][-1].content}]
        )

    def math_agent(state):
        agent = _create_math_agent(model=model)
        response = agent.invoke(state)
        return MessagesState(
            messages=[{"role": "user", "content": response["messages"][-1].content}]
        )

    assign_to_research_agent_with_description = create_task_description_handoff_tool(
        agent_name="research_agent",
        description="Assign task to a researcher agent.",
    )

    assign_to_math_agent_with_description = create_task_description_handoff_tool(
        agent_name="math_agent",
        description="Assign task to a math agent.",
    )

    supervisor_agent_with_description = create_react_agent(
        model=model,
        tools=[
            assign_to_research_agent_with_description,
            assign_to_math_agent_with_description,
        ],
        prompt=(
            """
            You are a supervisor managing two agents:
            - a research agent. Assign research-related tasks to this agent
            - a math agent. Assign math-related tasks to this agent
            Assign work to one agent at a time, do not call agents in parallel
            Be very detailed, even verbose, and not ambiguous when assigning tasks
            Do not do any work yourself, do not call any agent with empty parameters
            If you couldn't do the task, respond precisely why and what agents didn't do the task as expected
            Once you're done with your tasks, respond directly with the results
            """
        ),
        name="supervisor",
    )

    supervisor_with_description = (
        StateGraph(MessagesState)
        .add_node(
            supervisor_agent_with_description,
            destinations=("research_agent", "math_agent"),
        )
        .add_node(research_agent)
        .add_node(math_agent)
        .add_edge(START, "supervisor")
        .add_edge("research_agent", "supervisor")
        .add_edge("math_agent", "supervisor")
        .compile()
    )

    for chunk in supervisor_with_description.stream(
        {
            "messages": [
                {
                    "role": "user",
                    "content": "find the current exchange rate of USD to EUR, then exchange 100 USD to EUR",
                }
            ]
        },
    ):
        pretty_print_messages(chunk, last_message=True)


def main():
    # Create the web search tool
    web_search = TavilySearch(
        max_results=3, search_depth="advanced", topic="finance", time_range="year"
    )

    # Call the web search tool
    # web_search_results = web_search.invoke("who is the mayor of NYC?")
    # print(web_search_results["results"][0]["content"])

    model = "google_genai:gemini-2.0-flash"

    # Create the research agent
    research_agent = _create_research_agent(
        model=model,
        search_tool=web_search,
    )

    # Call the research agent
    # for chunk in research_agent.stream(
    #         {"messages": [{"role": "user", "content": "who is the mayor of NYC?"}]}
    # ):
    #     pretty_print_messages(chunk)

    # Create the math agent
    math_agent = _create_math_agent(model=model)

    # Call the math agent
    # for chunk in math_agent.stream(
    #         {"messages": [{"role": "user", "content": "what's (3 + 5) x 7"}]}
    # ):
    #     pretty_print_messages(chunk)

    # Create the supervisor
    supervisor = _create_supervisor(
        model=model,
        agents=[research_agent, math_agent],
    )

    # Call the supervisor of our multi-agent system
    for chunk in supervisor.stream(
        {
            "messages": [
                {
                    "role": "user",
                    "content": "find the current exchange rate of USD to EUR, then exchange 100 USD to EUR",
                }
            ]
        },
    ):
        pretty_print_messages(chunk, last_message=True)

    # Run the multi-agent system we built from scratch (with full message history)
    # _run_multi_agent_supervisor(model=model)

    # Run the multi-agent system we built from scratch with delegation
    _run_multi_agent_supervisor_delegating(model=model)


if __name__ == "__main__":
    set_env("GOOGLE_API_KEY")
    set_env("TAVILY_API_KEY")
    main()
