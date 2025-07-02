from langchain_core.vectorstores import InMemoryVectorStore
from langchain_google_genai import GoogleGenerativeAIEmbeddings


class VectorStore:
    def __init__(self, text: str, model: str):
        vectorstore: InMemoryVectorStore = InMemoryVectorStore.from_texts(
            texts=[text],
            embedding=GoogleGenerativeAIEmbeddings(model=model),
        )
        self.retriever = vectorstore.as_retriever()
