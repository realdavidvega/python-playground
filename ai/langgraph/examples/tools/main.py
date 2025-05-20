import json

from langchain.chat_models import init_chat_model
from langchain_core.messages import ToolMessage
from langchain_tavily import TavilySearch
from langgraph.constants import START, END
from langgraph.graph import add_messages, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition
from typing_extensions import TypedDict, Annotated

# 2. Add tools
# To handle queries you chatbot can't answer "from memory", integrate a web search tool.
# The chatbot can use this tool to find relevant information and provide better responses.


def main_basic_tools():
    # Define the web search tool
    search_tool = TavilySearch(max_results=2)
    llm_tools = [search_tool]
    # tool.invoke("What's a 'node' in LangGraph?")

    class State(TypedDict):
        messages: Annotated[list, add_messages]

    graph_builder = StateGraph(State)

    # llm = init_chat_model("anthropic:claude-3-5-sonnet-latest")
    llm = init_chat_model("google_genai:gemini-2.0-flash")

    # For the StateGraph you created in the chat example, add bind_tools on the LLM.
    # This lets the LLM know the correct JSON format to use if it wants to use the search engine.

    # Modification: tell the LLM which tools it can call
    # highlight-next-line
    llm_with_tools = llm.bind_tools(llm_tools)

    def chatbot(state: State):
        return {"messages": [llm_with_tools.invoke(state["messages"])]}

    graph_builder.add_node("chatbot", chatbot)

    # Now, create a function to run the tools if they are called.
    # Do this by adding the tools to a new node calledBasicToolNode that checks the most recent message in
    # the state and calls tools if the message contains tool_calls. It relies on the LLM's tool_calling support,
    # which is available in Anthropic, OpenAI, Google Gemini, and a number of other LLM providers.

    # If you do not want to build this yourself in the future, you can use LangGraph's prebuilt ToolNode.
    class BasicToolNode:
        """A node that runs the tools requested in the last AIMessage."""

        def __init__(self, tools: list) -> None:
            self.tools_by_name = {tool.name: tool for tool in tools}

        def __call__(self, inputs: dict):
            if messages := inputs.get("messages", []):
                message = messages[-1]
            else:
                raise ValueError("No message found in input")
            outputs = []
            for tool_call in message.tool_calls:
                tool_result = self.tools_by_name[tool_call["name"]].invoke(
                    tool_call["args"]
                )
                outputs.append(
                    ToolMessage(
                        content=json.dumps(tool_result),
                        name=tool_call["name"],
                        tool_call_id=tool_call["id"],
                    )
                )
            return {"messages": outputs}

    tool_node = BasicToolNode(tools=[search_tool])
    graph_builder.add_node("tools", tool_node)

    # With the tool node added, now you can define the conditional_edges.
    # Edges route the control flow from one node to the next. Conditional edges start from a single node and usually
    # contain "if" statements to route to different nodes depending on the current graph state.
    # These functions receive the current graph state and return a string or list of strings indicating
    # which node(s) to call next.

    # Next, define a router function called route_tools that checks for tool_calls in the chatbot's output.
    # Provide this function to the graph by calling add_conditional_edges, which tells the graph that whenever the
    # chatbot node completes to check this function to see where to go next.

    # The condition will route to tools if tool calls are present and END if not.
    # Because the condition can return END, you do not need to explicitly set a finish_point this time.
    def route_tools(
        state: State,
    ):
        """
        Use in the conditional_edge to route to the ToolNode if the last message
        has tool calls. Otherwise, route to the end.
        """
        if isinstance(state, list):
            ai_message = state[-1]
        elif messages := state.get("messages", []):
            ai_message = messages[-1]
        else:
            raise ValueError(f"No messages found in input state to tool_edge: {state}")
        if hasattr(ai_message, "tool_calls") and len(ai_message.tool_calls) > 0:
            return "tools"
        return END

    # The `tools_condition` function returns "tools" if the chatbot asks to use a tool, and "END" if
    # it is fine directly responding. This conditional routing defines the main agent loop.
    graph_builder.add_conditional_edges(
        "chatbot",
        route_tools,
        # The following dictionary lets you tell the graph to interpret the condition's outputs as a specific node
        # It defaults to the identity function, but if you
        # want to use a node named something else apart from "tools",
        # You can update the value of the dictionary to something else
        # e.g., "tools": "my_tools"
        {"tools": "tools", END: END},
    )
    # Any time a tool is called, we return to the chatbot to decide the next step
    graph_builder.add_edge("tools", "chatbot")
    graph_builder.add_edge(START, "chatbot")
    graph = graph_builder.compile()

    def stream_graph_updates(user_input: str):
        for event in graph.stream(
            {"messages": [{"role": "user", "content": user_input}]}
        ):
            for value in event.values():
                print("Assistant:", value["messages"][-1].content)

    while True:
        # What do you know about LangGraph?
        user_input = input("User: ")
        if user_input.lower() in ["quit", "exit", "q"]:
            print("Goodbye!")
            break

        stream_graph_updates(user_input)


def main():
    class State(TypedDict):
        messages: Annotated[list, add_messages]

    graph_builder = StateGraph(State)

    # llm = init_chat_model("anthropic:claude-3-5-sonnet-latest")
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
    graph = graph_builder.compile()

    def stream_graph_updates(user_input: str):
        for event in graph.stream(
            {"messages": [{"role": "user", "content": user_input}]}
        ):
            for value in event.values():
                print("Assistant:", value["messages"][-1].content)

    while True:
        # What do you know about LangGraph?
        user_input = input("User: ")
        if user_input.lower() in ["quit", "exit", "q"]:
            print("Goodbye!")
            break

        stream_graph_updates(user_input)


if __name__ == "__main__":
    main()
