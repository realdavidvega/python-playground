from langchain_core.tools import Tool
from langgraph.graph.state import CompiledStateGraph
from typing_extensions import List

from playground.examples.images_pdf_chat.application.chat.chatbot import Chatbot
from playground.examples.images_pdf_chat.application.graph.graph import Graph
from playground.examples.images_pdf_chat.application.graph.tools.tools import Tools
from playground.examples.images_pdf_chat.infrastructure.resources.config import Config
from playground.examples.images_pdf_chat.infrastructure.resources.resources import (
    Resources,
)


if __name__ == "__main__":
    file_name: str = input("Please enter the name of the PDF file: ")

    config: Config = Config()
    resources: Resources = Resources(file_name)
    tools: List[Tool] = Tools.load(resources)

    graph: CompiledStateGraph = Graph.compile(tools, config)

    Chatbot(graph).run()
