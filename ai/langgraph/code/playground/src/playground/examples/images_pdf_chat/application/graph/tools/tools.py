from typing import List

from langchain_core.tools import Tool

from playground.examples.images_pdf_chat.application.graph.tools.retrieval import (
    RetrievalTool,
)
from playground.examples.images_pdf_chat.infrastructure.resources.resources import (
    Resources,
)


class Tools:
    @staticmethod
    def load(resources: Resources) -> List[Tool]:
        retrieval_tool: Tool = RetrievalTool(resources.vector_store).tool
        return [retrieval_tool]
