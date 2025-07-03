from typing import Self

from langchain_core.tools import create_retriever_tool, Tool
from pydantic.dataclasses import dataclass

from playground.examples.images_pdf_chat.infrastructure.resources.config import (
    BASE_CONFIG,
)
from playground.examples.images_pdf_chat.infrastructure.vector_store import VectorStore


@dataclass(config=BASE_CONFIG)
class RetrievalTool:
    tool: Tool

    @classmethod
    def create(cls, vector_store: VectorStore) -> Self:
        tool: Tool = create_retriever_tool(
            retriever=vector_store.retriever,
            name="retrieve_information",
            description="Search and return information about the document.",
        )

        return cls(tool=tool)
