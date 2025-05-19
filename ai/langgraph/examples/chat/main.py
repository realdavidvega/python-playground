from typing import Annotated

from langchain.chat_models import init_chat_model
from typing_extensions import TypedDict

from langgraph.graph import StateGraph, START
from langgraph.graph.message import add_messages

# When defining a graph, the first step is to define its State. The State includes the graph's
# schema and reducer functions that handle state updates. In our example, State is a TypedDict with
# one key: messages. The add_messages reducer function is used to append new messages to the list instead of
# overwriting it. Keys without a reducer annotation will overwrite previous values.

# To learn more about state, reducers, and related concepts, see LangGraph reference docs.
class State(TypedDict):
    messages: Annotated[list, add_messages]

def main():
    # A StateGraph object defines the structure of our chatbot as a "state machine".
    # We'll add nodes to represent the llm and functions our chatbot can call and edges to specify how the bot should
    # transition between these functions.
    graph_builder = StateGraph(State)

    # Our graph can now handle two key tasks:
    #
    # - Each node can receive the current State as input and output an update to the state.
    # - Updates to messages will be appended to the existing list rather than overwriting it, thanks to the prebuilt
    # add_messages function used with the Annotated syntax.

    # Let's first select a chat model

    # llm = init_chat_model("anthropic:claude-3-5-sonnet-latest")
    llm = init_chat_model("google_genai:gemini-2.0-flash")

    # Next, add a "chatbot" node
    # Nodes represent units of work and are typically regular Python functions.

    def chatbot(state: State):
        return {"messages": [llm.invoke(state["messages"])]}

    # The first argument is the unique node name
    # The second argument is the function or object that will be called whenever
    # the node is used.
    graph_builder.add_node("chatbot", chatbot)

    # Notice how the chatbot node function takes the current State as input and returns a dictionary containing an
    # updated messages list under the key "messages". This is the basic pattern for all LangGraph node functions.
    #
    # The add_messages function in our State will append the LLM's response messages to whatever messages are
    # already in the state.

    # Add an entry point to tell the graph where to start its work each time it is run
    graph_builder.add_edge(START, "chatbot")

    # Before running the graph, we'll need to compile it. We can do so by calling compile() on the graph builder.
    # This creates a CompiledGraph we can invoke on our state.
    graph = graph_builder.compile()

    def stream_graph_updates(user_input: str):
        for event in graph.stream({"messages": [{"role": "user", "content": user_input}]}):
            for value in event.values():
                print("Assistant:", value["messages"][-1].content)

    while True:
        try:
            user_input = input("User: ")

            # You can exit the chat loop at any time by typing quit, exit, or q.
            if user_input.lower() in ["quit", "exit", "q"]:
                print("Goodbye!")
                break
            stream_graph_updates(user_input)
        except:
            # fallback if input() is not available
            user_input = "What do you know about LangGraph?"
            print("User: " + user_input)
            stream_graph_updates(user_input)
            break

if __name__ == "__main__":
    main()
