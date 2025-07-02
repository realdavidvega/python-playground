from langchain_core.tools import create_retriever_tool, Tool

from playground.examples.images_pdf_chat.infrastructure.vector_store import VectorStore


class RetrievalTool:
    tool: Tool

    def __init__(self, vector_store: VectorStore):
        self._tool = create_retriever_tool(
            retriever=vector_store.retriever,
            name="retrieve_information",
            description="Search and return information about the document.",
        )
