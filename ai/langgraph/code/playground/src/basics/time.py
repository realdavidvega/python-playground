from langchain.chat_models import init_chat_model
from langchain_core.runnables import RunnableConfig
from langchain_tavily import TavilySearch
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, START
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from typing_extensions import TypedDict, Annotated


# 6. Time travel
# In a typical chatbot workflow, the user interacts with the bot one or more times to accomplish a task.
# Memory and a human-in-the-loop enable checkpoints in the graph state and control future responses.

# What if you want a user to be able to start from a previous response and explore a different outcome?
# Or what if you want users to be able to rewind your chatbot's work to fix mistakes or try a different strategy,
# something that is common in applications like autonomous software engineers?

# You can create these types of experiences using LangGraph's built-in time travel functionality.


def main():
    llm = init_chat_model("google_genai:gemini-2.0-flash")

    # Rewind your graph
    # Rewind your graph by fetching a checkpoint using the graph's get_state_history method.
    # You can then resume execution at this previous point in time.

    class State(TypedDict):
        messages: Annotated[list, add_messages]

    graph_builder = StateGraph(State)

    tool = TavilySearch(max_results=2)
    tools = [tool]
    llm_with_tools = llm.bind_tools(tools)

    def chatbot(state: State):
        return {"messages": [llm_with_tools.invoke(state["messages"])]}

    graph_builder.add_node("chatbot", chatbot)

    tool_node = ToolNode(tools=[tool])
    graph_builder.add_node("tools", tool_node)

    graph_builder.add_conditional_edges(
        "chatbot",
        tools_condition,
    )
    graph_builder.add_edge("tools", "chatbot")
    graph_builder.add_edge(START, "chatbot")

    memory = MemorySaver()
    graph = graph_builder.compile(checkpointer=memory)

    config = RunnableConfig(configurable={"thread_id": "1"})
    events = graph.stream(
        {
            "messages": [
                {
                    "role": "user",
                    "content": (
                        "I'm learning LangGraph. "
                        "Could you do some research on it for me?"
                    ),
                },
            ],
        },
        config,
        stream_mode="values",
    )
    for event in events:
        if "messages" in event:
            event["messages"][-1].pretty_print()

    # Replay the full state history
    # Now that you have added steps to the chatbot, you can replay the full state history to see everything
    # that occurred
    to_replay = None
    for state in graph.get_state_history(config):
        print("Num Messages: ", len(state.values["messages"]), "Next: ", state.next)
        print("-" * 80)
        if len(state.values["messages"]) == 2:
            # We are somewhat arbitrarily selecting a specific state based on the number of chat messages in the state.
            to_replay = state

    # Checkpoints are saved for every step of the graph.
    # This spans invocations so you can rewind across a full thread's history.

    # Resume from a checkpoint
    # Resume from the to_replay state, which is after the chatbot node in the second graph invocation.
    # Resuming from this point will call the action node next.
    print(to_replay.next)
    print(to_replay.config)

    # Load a state from a moment-in-time
    # The checkpoint's to_replay.config contains a checkpoint_id timestamp.
    # Providing this checkpoint_id value tells LangGraph's checkpointer to load the state from that moment in time.

    # The `checkpoint_id` in the `to_replay.config` corresponds to a state we've persisted to our checkpointer.
    for event in graph.stream(None, to_replay.config, stream_mode="values"):
        if "messages" in event:
            event["messages"][-1].pretty_print()


if __name__ == "__main__":
    main()
