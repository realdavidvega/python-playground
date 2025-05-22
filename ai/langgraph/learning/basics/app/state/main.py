from typing import Annotated

from langchain.chat_models import init_chat_model
from langchain_core.runnables import RunnableConfig
from langchain_tavily import TavilySearch
from langchain_core.messages import ToolMessage
from langchain_core.tools import InjectedToolCallId, tool
from typing_extensions import TypedDict

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, START
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.types import Command, interrupt

# 5. Customize state
# In this tutorial, you will add additional fields to the state to define complex behavior without relying
# on the message list. The chatbot will use its search tool to find specific information and forward them
# to a human for review.


def main():
    # Update the chatbot to research the birthday of an entity by adding name and birthday keys to the state
    class State(TypedDict):
        messages: Annotated[list, add_messages]
        name: str
        birthday: str

    # Adding this information to the state makes it easily accessible by other graph nodes (like a downstream node
    # that stores or processes the information), as well as the graph's persistence layer.

    # os.environ["GOOGLE_API_KEY"] = "..."
    llm = init_chat_model("google_genai:gemini-2.0-flash")

    # Now, populate the state keys inside the human_assistance tool. This allows a human to review the
    # information before it is stored in the state. Use Command to issue a state update from inside the tool.

    @tool
    def human_assistance(
        name: str, birthday: str, tool_call_id: Annotated[str, InjectedToolCallId]
    ) -> Command:
        """Request assistance from a human."""
        human_response = interrupt(
            {
                "question": "Is this correct?",
                "name": name,
                "birthday": birthday,
            },
        )
        # If the human approves the information, store it in the state
        if human_response.get("correct", "").lower().startswith("y"):
            verified_name = name
            verified_birthday = birthday
            response = "Correct"
        # If the human corrects the information, update the state
        else:
            verified_name = human_response.get("name", name)
            verified_birthday = human_response.get("birthday", birthday)
            response = f"Made a correction: {human_response}"

        # Update the state
        state_update = {
            "name": verified_name,
            "birthday": verified_birthday,
            "messages": [ToolMessage(response, tool_call_id=tool_call_id)],
        }
        # Return a Command to update the state
        return Command(update=state_update)

    search_tool = TavilySearch(max_results=2)

    # Add the human_assistance tool to the chatbot
    tools = [search_tool, human_assistance]
    llm_with_tools = llm.bind_tools(tools)

    def chatbot(state: State):
        message = llm_with_tools.invoke(state["messages"])
        # Again, we assert that only one tool was called (no parallel tool calling)
        assert len(message.tool_calls) <= 1
        return {"messages": [message]}

    graph_builder = StateGraph(State)
    graph_builder.add_node("chatbot", chatbot)

    tool_node = ToolNode(tools=tools)
    graph_builder.add_node("tools", tool_node)

    graph_builder.add_conditional_edges(
        "chatbot",
        tools_condition,
    )
    graph_builder.add_edge("tools", "chatbot")
    graph_builder.add_edge(START, "chatbot")

    memory = MemorySaver()
    graph = graph_builder.compile(checkpointer=memory)

    # Prompt the chatbot to look up the "birthday" of the LangGraph library and direct the chatbot to
    # reach out to the human_assistance tool once it has the required information.
    # By setting name and birthday in the arguments for the tool, you force the chatbot to generate
    # proposals for these fields.
    user_input = (
        "Can you look up when LangGraph was released? "
        "When you have the answer, use the human_assistance tool for review."
    )
    config = RunnableConfig(configurable={"thread_id": "1"})

    events = graph.stream(
        {"messages": [{"role": "user", "content": user_input}]},
        config,
        stream_mode="values",
    )
    for event in events:
        if "messages" in event:
            event["messages"][-1].pretty_print()

    # We've hit the interrupt in the human_assistance tool again.

    # Add human assistance
    # The chatbot failed to identify the correct date, so supply it with information

    human_command = Command(
        resume={
            "name": "LangGraph",
            "birthday": "Jan 17, 2024",
        },
    )

    events = graph.stream(human_command, config, stream_mode="values")
    for event in events:
        if "messages" in event:
            event["messages"][-1].pretty_print()

    # Note that these fields are now reflected in the state
    snapshot = graph.get_state(config)
    print(f"Snapshot: {snapshot}")

    # This makes them easily accessible to downstream nodes (e.g., a node that
    # further processes or stores the information).

    # Manually update the state
    # LangGraph gives a high degree of control over the application state.
    # For instance, at any point (including when interrupted), you can manually override a key using
    # graph.update_state
    graph.update_state(config, {"name": "LangGraph (library)"})

    # If you call graph.get_state, you can see the new value is reflected
    snapshot = graph.get_state(config)
    print(f"Snapshot: {snapshot}")

    # Manual state updates will generate a trace in LangSmith. If desired, they can also be used to control
    # human-in-the-loop workflows. Use of the interrupt function is generally recommended instead,
    # as it allows data to be transmitted in a human-in-the-loop interaction independently of state updates.


if __name__ == "__main__":
    main()
