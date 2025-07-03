from dataclasses import dataclass
from typing import Self

from langchain_core.tools import create_retriever_tool, Tool
from playground.examples.images_pdf_chat.infrastructure.vector_store import VectorStore


@dataclass(frozen=True)
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
