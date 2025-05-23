from langchain_core.messages import convert_to_messages
from langchain_core.tools import tool, InjectedToolCallId
from langgraph.constants import START
from langgraph.prebuilt import InjectedState, create_react_agent
from langgraph.graph import MessagesState, StateGraph
from langgraph.types import Command, Send
from typing_extensions import Annotated


# Handoffs

# A single agent might struggle if it needs to specialize in multiple domains or manage many tools.
# To tackle this, you can break your agent into smaller, independent agents and composing them into a
# multi-agent system.


# We'll use `pretty_print_messages` helper to render the streamed agent outputs nicely later on
def pretty_print_messages(update, last_message=False):
    def __pretty_print_message(message, indent=False):
        pretty_message = message.pretty_repr(html=True)
        if not indent:
            print(pretty_message)
            return

        indented = "\n".join("\t" + c for c in pretty_message.split("\n"))
        print(indented)

    is_subgraph = False
    if isinstance(update, tuple):
        ns, update = update
        # skip parent graph updates in the printouts
        if len(ns) == 0:
            return

        graph_id = ns[-1].split(":")[0]
        print(f"Update from subgraph {graph_id}:")
        print("\n")
        is_subgraph = True

    for node_name, node_update in update.items():
        update_label = f"Update from node {node_name}:"
        if is_subgraph:
            update_label = "\t" + update_label

        print(update_label)
        print("\n")

        messages = convert_to_messages(node_update["messages"])
        if last_message:
            messages = messages[-1:]

        for m in messages:
            __pretty_print_message(m, indent=is_subgraph)
        print("\n")


# In multi-agent systems, agents need to communicate between each other. They do so via handoffs: a primitive
# that describes which agent to hand control to and the payload to send to that agent. Handoffs allow you to specify:
# - destination: target agent to navigate to (e.g., name of the LangGraph node to go to)
# - payload: information to pass to that agent (e.g., state update)
def create_handoff_tool(*, agent_name: str, description: str | None = None):
    name = f"transfer_to_{agent_name}"
    description = description or f"Transfer to {agent_name}"

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
        return Command(
            goto=agent_name,
            update={"messages": state["messages"] + [tool_message]},
            graph=Command.PARENT,
        )

    return handoff_tool


# If you want to use tools that return Command, you can either use prebuilt create_react_agent / ToolNode components,
# or implement your own tool-executing node that collects Command objects returned by the tools and returns a
# list of them, e.g.:
def __call_tools(state):
    tools_by_name = {}
    tool_calls = state["messages"]
    commands = [
        tools_by_name[tool_call["name"]].invoke(tool_call) for tool_call in tool_calls
    ]
    return commands


# This handoff implementation assumes that:
# - each agent receives overall message history (across all agents) in the multi-agent system as its input.
# - each agent outputs its internal messages history to the overall message history of the multi-agent system.
# If you want more control over how agent outputs are added, wrap the agent in a separate node function:
def __call_hotel_assistant(state):
    hotel_assistant = {}
    # return agent's final response,
    # excluding inner monologue
    response = hotel_assistant.invoke(state)
    return {"messages": response["messages"][-1]}


# Control agent inputs
# You can use the Send() primitive to directly send data to the worker agents during the handoff.
# For example, you can request that the calling agent populate a task description for the next agent
def create_task_description_handoff_tool(
    *, agent_name: str, description: str | None = None
):
    name = f"transfer_to_{agent_name}"
    description = description or f"Ask {agent_name} for help."

    @tool(name, description=description)
    def handoff_tool(
        # this is populated by the calling agent
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


# Build a multi-agent system
# You can use handoffs in any agents built with LangGraph. We recommend using the prebuilt agent or ToolNode,
# as they natively support handoffs tools returning Command.
# Below is an example of how you can implement a multi-agent system for booking travel using handoffs
def main():
    transfer_to_hotel_assistant = create_handoff_tool(agent_name="hotel_assistant")
    transfer_to_flight_assistant = create_handoff_tool(agent_name="flight_assistant")

    # Simple agent tools
    def book_hotel(hotel_name: str):
        """Book a hotel"""
        return f"Successfully booked a stay at {hotel_name}."

    def book_flight(from_airport: str, to_airport: str):
        """Book a flight"""
        return f"Successfully booked a flight from {from_airport} to {to_airport}."

    # Define agents
    flight_assistant = create_react_agent(
        model="google_genai:gemini-2.5-flash-preview-05-20",
        tools=[book_flight, transfer_to_hotel_assistant],
        name="flight_assistant",
    )
    hotel_assistant = create_react_agent(
        model="google_genai:gemini-2.5-flash-preview-05-20",
        tools=[book_hotel, transfer_to_flight_assistant],
        name="hotel_assistant",
    )

    # Define multi-agent graph
    multi_agent_graph = (
        StateGraph(MessagesState)
        .add_node(flight_assistant)
        .add_node(hotel_assistant)
        .add_edge(START, "flight_assistant")
        .compile()
    )

    # Run the multi-agent graph
    for chunk in multi_agent_graph.stream(
        {
            "messages": [
                {
                    "role": "user",
                    "content": "book a flight from BOS to JFK and a stay at McKittrick Hotel",
                }
            ]
        },
        subgraphs=True,
    ):
        pretty_print_messages(chunk)


if __name__ == "__main__":
    main()
