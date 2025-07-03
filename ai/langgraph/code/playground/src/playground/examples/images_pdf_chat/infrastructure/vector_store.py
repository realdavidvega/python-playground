from typing import Self

from langchain_core.vectorstores import InMemoryVectorStore, VectorStoreRetriever
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from pydantic.dataclasses import dataclass

from playground.examples.images_pdf_chat.infrastructure.resources.config import (
    BASE_CONFIG,
)


@dataclass(config=BASE_CONFIG)
class VectorStore:
    retriever: VectorStoreRetriever

    @classmethod
    def create(cls, text: str, model: str) -> Self:
        vectorstore: InMemoryVectorStore = InMemoryVectorStore.from_texts(
            texts=[text],
            embedding=GoogleGenerativeAIEmbeddings(model=model),
        )
        return cls(retriever=vectorstore.as_retriever())
