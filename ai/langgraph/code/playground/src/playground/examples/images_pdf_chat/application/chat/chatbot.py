import uuid

from langchain_core.runnables import RunnableConfig
from langgraph.graph.state import CompiledStateGraph
from pydantic.dataclasses import dataclass

from playground.examples.images_pdf_chat.infrastructure.resources.config import (
    BASE_CONFIG,
)


@dataclass(config=BASE_CONFIG)
class Chatbot:
    graph: CompiledStateGraph

    def run(self):
        while True:
            user_input = input("User: ")
            if user_input.lower() in ["quit", "exit", "q"]:
                print("Goodbye!")
                break

            graph_config: RunnableConfig = RunnableConfig(
                configurable={"thread_id": uuid.uuid4().hex}
            )
            self.__stream_graph_updates(user_input, graph_config)

    def __stream_graph_updates(self, content: str, graph_config: RunnableConfig):
        events = self.graph.stream(
            input={"messages": [{"role": "user", "content": content}]},
            config=graph_config,
            stream_mode="values",
        )
        for event in events:
            event["messages"][-1].pretty_print()
