from langchain.chat_models import init_chat_model
from langchain_core.runnables import RunnableConfig
from langchain_tavily import TavilySearch
from langchain_core.tools import tool
from typing_extensions import TypedDict, Annotated

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, START
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition

from langgraph.types import interrupt, Command


# 4. Add human-in-the-loop controls

# Agents can be unreliable and may need human input to successfully accomplish tasks.
# Similarly, for some actions, you may want to require human approval before running to ensure that everything
# is running as intended.

# LangGraph's persistence layer supports human-in-the-loop workflows, allowing execution to pause and resume
# based on user feedback. The primary interface to this functionality is the interrupt function.
# Calling interrupt inside a node will pause execution. Execution can be resumed, together with new input
# from a human, by passing in a Command. interrupt is ergonomically similar to Python's built-in input(),
# with some caveats.


def main():
    # Starting with the existing code from the Add memory to the chatbot tutorial, add
    # the human_assistance tool to the chatbot. This tool uses interrupt to receive information from a human.

    class State(TypedDict):
        messages: Annotated[list, add_messages]

    graph_builder = StateGraph(State)

    # os.environ["GOOGLE_API_KEY"] = "..."
    llm = init_chat_model("google_genai:gemini-2.0-flash")

    # We will use the interrupt function to request human assistance
    @tool
    def human_assistance(query: str) -> str:
        """Request assistance from a human."""
        human_response = interrupt({"query": query})
        return human_response["data"]

    # Similar to Python's built-in input() function, calling interrupt inside the tool will pause execution.
    # Progress is persisted based on the checkpointer; so if it is persisting with Postgres, it can resume at any
    # time as long as the database is alive. In this example, it is persisting with the in-memory checkpointer and
    # can resume any time if the Python kernel is running.

    search_tool = TavilySearch(max_results=2)
    tools = [search_tool, human_assistance]
    llm_with_tools = llm.bind_tools(tools)

    def chatbot(state: State):
        message = llm_with_tools.invoke(state["messages"])
        # Because we will be interrupting during tool execution,
        # we disable parallel tool calling to avoid repeating any
        # tool invocations when we resume.

        # assert(len(message.tool_calls) <= 1)
        assert len(message.tool_calls) <= 1
        return {"messages": [message]}

    graph_builder.add_node("chatbot", chatbot)

    tool_node = ToolNode(tools=tools)
    graph_builder.add_node("tools", tool_node)

    graph_builder.add_conditional_edges(
        "chatbot",
        tools_condition,
    )
    graph_builder.add_edge("tools", "chatbot")
    graph_builder.add_edge(START, "chatbot")

    # We compile the graph with a checkpointer, as before
    memory = MemorySaver()
    graph = graph_builder.compile(checkpointer=memory)

    # Now, prompt the chatbot with a question that will engage the new human_assistance tool:
    user_input = "I need some expert guidance for building an AI agent. Could you request assistance for me?"
    config = {"configurable": {"thread_id": "1"}}
    runnable_config = RunnableConfig(**config)

    events = graph.stream(
        {"messages": [{"role": "user", "content": user_input}]},
        runnable_config,
        stream_mode="values",
    )
    for event in events:
        if "messages" in event:
            event["messages"][-1].pretty_print()

    # The chatbot generated a tool call, but then execution has been interrupted.
    # If you inspect the graph state, you see that it stopped at the tools' node:
    snapshot = graph.get_state(runnable_config)
    print(f"Snapshot: {snapshot}\n")

    print(f"Next: {snapshot.next}\n")

    # To resume execution, pass a Command object containing data expected by the tool.
    # The format of this data can be customized based on needs. For this example, use a dict with a key "data"
    human_response = (
        "We, the experts are here to help! We'd recommend you check out LangGraph to build your agent."
        " It's much more reliable and extensible than simple autonomous agents."
    )

    human_command = Command(resume={"data": human_response})

    events = graph.stream(human_command, runnable_config, stream_mode="values")
    for event in events:
        if "messages" in event:
            event["messages"][-1].pretty_print()


if __name__ == "__main__":
    main()
