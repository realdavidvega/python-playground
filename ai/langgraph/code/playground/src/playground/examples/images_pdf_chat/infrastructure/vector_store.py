from dataclasses import dataclass
from typing import Self

from langchain_core.vectorstores import InMemoryVectorStore, VectorStoreRetriever
from langchain_google_genai import GoogleGenerativeAIEmbeddings


@dataclass(frozen=True)
class VectorStore:
    retriever: VectorStoreRetriever

    @classmethod
    def create(cls, text: str, model: str) -> Self:
        vectorstore: InMemoryVectorStore = InMemoryVectorStore.from_texts(
            texts=[text],
            embedding=GoogleGenerativeAIEmbeddings(model=model),
        )
        return cls(retriever=vectorstore.as_retriever())
