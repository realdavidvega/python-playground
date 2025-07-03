from dataclasses import dataclass
from typing import Self, ClassVar

from langchain_core.tools import create_retriever_tool, Tool
from playground.examples.images_pdf_chat.infrastructure.vector_store import VectorStore


@dataclass(frozen=True)
class RetrievalTool:
    tool: Tool

    NAME: ClassVar[str] = "retrieve_tool"
    DESCRIPTION: ClassVar[str] = "Search and return information about the document."

    @classmethod
    def create(cls, vector_store: VectorStore) -> Self:
        tool: Tool = create_retriever_tool(
            retriever=vector_store.retriever,
            name=RetrievalTool.NAME,
            description=RetrievalTool.DESCRIPTION,
        )

        return cls(tool=tool)
