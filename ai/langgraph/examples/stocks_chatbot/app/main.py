from langchain_core.runnables import RunnableConfig
from langgraph.graph.state import CompiledStateGraph

from app.resources.resources import resources
from app.graph.graph import graph


def init_chat(stocks_graph: CompiledStateGraph):
    def stream_graph_updates(content: str):
        events = stocks_graph.stream(
            input={"messages": [{"role": "user", "content": content}]},
            config=RunnableConfig(configurable={"thread_id": "1"}),
            stream_mode="values",
        )
        for event in events:
            event["messages"][-1].pretty_print()

    while True:
        user_input = input("User: ")
        if user_input.lower() in ["quit", "exit", "q"]:
            print("Goodbye!")
            break

        stream_graph_updates(user_input)


if __name__ == "__main__":
    resources = resources()
    graph = graph(resources)
    init_chat(graph)
