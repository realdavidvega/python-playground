from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import MemorySaver

from langchain.chat_models import init_chat_model
from langchain_tavily import TavilySearch
from langgraph.constants import START
from langgraph.graph import add_messages, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition
from typing_extensions import TypedDict, Annotated

# 3. Add memory
# The chatbot can now use tools to answer user questions, but it does not remember the context of previous
# interactions. This limits its ability to have coherent, multi-turn conversations.

# LangGraph solves this problem through persistent checkpointing. If you provide a checkpointer
# when compiling the graph and a thread_id when calling your graph, LangGraph automatically saves the state
# after each step. When you invoke the graph again using the same thread_id, the graph loads its saved state,
# allowing the chatbot to pick up where it left off.

# We will see later that checkpointing is much more powerful than simple chat memory - it lets you save and
# resume complex state at any time for error recovery, human-in-the-loop workflows,
# time travel interactions, and more. But first, let's add checkpointing to enable multi-turn conversations.


def main():
    ###########################################
    # START  CODE FROM PREVIOUS TOOLS EXAMPLE #
    ###########################################

    class State(TypedDict):
        messages: Annotated[list, add_messages]

    graph_builder = StateGraph(State)

    # llm = init_chat_model("anthropic:claude-3-5-sonnet-latest")

    # os.environ["GOOGLE_API_KEY"] = "..."
    llm = init_chat_model("google_genai:gemini-2.0-flash")

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

    # Any time a tool is called, we return to the chatbot to decide the next step
    graph_builder.add_edge("tools", "chatbot")
    graph_builder.add_edge(START, "chatbot")

    ########################################
    # END CODE FROM PREVIOUS TOOLS EXAMPLE #
    ########################################

    # Create a MemorySaver checkpointer
    memory = MemorySaver()

    # This is in-memory checkpointer, which is convenient for the tutorial. However, in a production application,
    # you would likely change this to use SqliteSaver or PostgresSaver and connect a database.

    # Compile the graph with the provided checkpointer,
    # which will checkpoint the State as the graph works through each node
    graph = graph_builder.compile(checkpointer=memory)

    # Now you can interact with your bot!
    # 1. Pick a thread to use as the key for this conversation.
    config = RunnableConfig(configurable={"thread_id": "1"})

    # 2. Send a message to the chatbot
    user_input = "Hi there! My name is Will."

    # The config is the **second positional argument** to stream() or invoke()!
    events = graph.stream(
        {"messages": [{"role": "user", "content": user_input}]},
        config,
        stream_mode="values",
    )
    for event in events:
        event["messages"][-1].pretty_print()

    # Ask a follow-up question
    user_input = "Remember my name?"

    # The config is the **second positional argument** to stream() or invoke()!
    events = graph.stream(
        {"messages": [{"role": "user", "content": user_input}]},
        config,
        stream_mode="values",
    )
    for event in events:
        event["messages"][-1].pretty_print()

    # Notice that we aren't using an external list for memory: it's all handled by the checkpointer!
    # You can inspect the full execution in this LangSmith trace to see what's going on.

    # Don't believe me? Try this using a different config.
    # The only difference is we change the `thread_id` here to "2" instead of "1"
    events = graph.stream(
        input={"messages": [{"role": "user", "content": user_input}]},
        config=RunnableConfig(configurable={"thread_id": "2"}),
        stream_mode="values",
    )
    for event in events:
        event["messages"][-1].pretty_print()

    # Inspect the state
    # By now, we have made a few checkpoints across two different threads.
    # But what goes into a checkpoint? To inspect a graph's state for a given config at any time,
    # call get_state(config).
    snapshot = graph.get_state(config)
    print(f"Checkpoint: {snapshot}\n")

    # (since the graph ended this turn, `next` is empty.
    # If you fetch a state from within a graph invocation, next tells which node will execute next)
    print(f"Next: {snapshot.next}\n")

    # The snapshot above contains the current state values, corresponding config, and the next node to process.
    # In our case, the graph has reached an END state, so next is empty.


if __name__ == "__main__":
    main()
