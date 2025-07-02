import logging

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

logging.basicConfig(level=logging.INFO)

if __name__ == "__main__":
    file: str = input("Please enter the absolute path of the PDF file: ")

    config: Config = Config()
    resources: Resources = Resources(file, config)
    tools: List[Tool] = Tools.load(resources)

    graph: CompiledStateGraph = Graph.compile(tools, config)

    Chatbot(graph).run()
