import uuid

from langchain_core.runnables import RunnableConfig
from langgraph.graph.state import CompiledStateGraph

from playground.examples.stocks_chatbot.graph.graph import init_stocks_graph
from playground.examples.stocks_chatbot.resources.resources import init_graph_resources


def __init_chat(stocks_graph: CompiledStateGraph, graph_config: RunnableConfig):
    def stream_graph_updates(content: str):
        events = stocks_graph.stream(
            input={"messages": [{"role": "user", "content": content}]},
            config=graph_config,
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


def __runnable_config():
    return RunnableConfig(configurable={"thread_id": uuid.uuid4().hex})


if __name__ == "__main__":
    resources = init_graph_resources()
    graph = init_stocks_graph(resources)
    config = __runnable_config()

    graph.update_state(config, {"favorite_symbol": "AAPL"})

    __init_chat(graph, config)
