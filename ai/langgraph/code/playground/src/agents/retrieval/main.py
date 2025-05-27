import getpass
import os

from langchain_core.messages import convert_to_messages
from langgraph.constants import START, END
from langgraph.prebuilt import ToolNode, tools_condition
from typing_extensions import Literal

from langchain.chat_models import init_chat_model
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.documents import Document
from langchain_core.tools import create_retriever_tool, Tool
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.graph import MessagesState, StateGraph
from pydantic import BaseModel, Field


# Agentic RAG
# Retrieval agents are useful when you want an LLM to make a decision about whether to retrieve context from a
# vectorstore or respond to the user directly. The goal of this app is to:
# - Fetch and preprocess documents that will be used for retrieval.
# - Index those documents for semantic search and create a retriever tool for the agent.
# - Build an agentic RAG system that can decide when to use the retriever tool.


# Helper function for setting environment variables
def _set_env(key: str):
    if key not in os.environ:
        os.environ[key] = getpass.getpass(f"{key}:")


# Preprocess documents to use in our RAG system
def _pre_process_docs() -> list[Document]:
    # Fetch documents to use in our RAG system.
    # We will use three of the most recent pages from Lilian Weng's excellent blog.
    urls = [
        "https://lilianweng.github.io/posts/2024-11-28-reward-hacking/",
        "https://lilianweng.github.io/posts/2024-07-07-hallucination/",
        "https://lilianweng.github.io/posts/2024-04-12-diffusion-video/",
    ]

    # We'll start by fetching the content of the pages using WebBaseLoader utility
    docs = [WebBaseLoader(url).load() for url in urls]

    # Flatten the list
    docs_list = [item for sublist in docs for item in sublist]

    # Split the fetched documents into smaller chunks for indexing into our vectorstore
    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=100, chunk_overlap=50
    )

    return text_splitter.split_documents(docs_list)


# Build our retriever tool
# Now that we have our split documents, we can index them into a vector store that we'll use for semantic search.
def _create_retriever_tool(documents: list[Document]) -> Tool:
    vectorstore = InMemoryVectorStore.from_documents(
        documents=documents,
        embedding=GoogleGenerativeAIEmbeddings(model="models/text-embedding-004"),
    )
    retriever = vectorstore.as_retriever()

    # Create a retriever tool using LangChain's prebuilt create_retriever_tool
    return create_retriever_tool(
        retriever,
        "retrieve_blog_posts",
        "Search and return information about Lilian Weng blog posts.",
    )


# Generate query
# Now we will start building components (nodes and edges) for our agentic RAG graph.
# Note that the components will operate on the MessagesState — graph state that contains a messages key with a
# list of chat messages.


# Build a generate_query_or_respond node.
# It will call an LLM to generate a response based on the current graph state (list of messages).
# Given the input messages, it will decide to retrieve using the retriever tool, or respond directly to the user.
# Note that we're giving the chat model access to the retriever_tool we created earlier via .bind_tools
def _generate_query_or_respond(retriever_tool: Tool, state: MessagesState):
    """Call the model to generate a response based on the current state. Given
    the question, it will decide to retrieve using the retriever tool, or simply respond to the user.
    """
    response_model = init_chat_model("google_genai:gemini-2.0-flash", temperature=0)
    response = response_model.bind_tools([retriever_tool]).invoke(state["messages"])
    return {"messages": [response]}


# Grade documents
# Add a conditional edge — grade_documents — to determine whether the retrieved documents are relevant to the question.
# We will use a model with a structured output schema GradeDocuments for document grading.
# The grade_documents function will return the name of the node to go to based on the grading decision
# (generate_answer or rewrite_question)
def _grade_documents(
    model: str,
    state: MessagesState,
) -> Literal["generate_answer", "rewrite_question"]:
    """Determine whether the retrieved documents are relevant to the question."""

    print("grade_documents called")

    class GradeDocuments(BaseModel):
        """Grade documents using a binary score for relevance check."""

        binary_score: str = Field(
            description="Relevance score: 'yes' if relevant, or 'no' if not relevant"
        )

    grade_prompt = """
        You are a grader assessing relevance of a retrieved document to a user question.
        Here is the retrieved document:
        {context}
        Here is the user question:
        {question}
        If the document contains keyword(s) or semantic meaning related to the user question, grade it as relevant.
        Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question.
        """

    grader_model = init_chat_model(model=model, temperature=0)
    prompt = grade_prompt.format(
        question=state["messages"][0].content, context=state["messages"][-1].content
    )
    response = grader_model.with_structured_output(GradeDocuments).invoke(
        [{"role": "user", "content": prompt}]
    )
    score = response.binary_score

    if score == "yes":
        return "generate_answer"
    else:
        return "rewrite_question"


# Build the rewrite_question node.
# The retriever tool can return potentially irrelevant documents, which indicates a need to improve the original user
# question. To do so, we will call the rewrite_question node
def _rewrite_question(model: str, state: MessagesState) -> MessagesState:
    """Rewrite the original user question."""

    print("rewrite_question called")

    rewrite_prompt = """
        Look at the input and try to reason about the underlying semantic intent / meaning.
        Here is the initial question:
        {question}
        Formulate an improved question:
        """

    response_model = init_chat_model(model=model, temperature=0)
    prompt = rewrite_prompt.format(question=state["messages"][0].content)
    response = response_model.invoke([{"role": "user", "content": prompt}])
    return MessagesState(messages=[{"role": "user", "content": response.content}])


# Generate an answer
# Build generate_answer node: if we pass the grader checks, we can generate the final answer based on the original
# question and the retrieved context
def _generate_answer(model: str, state: MessagesState) -> MessagesState:
    """Generate an answer."""

    print("generate_answer called")

    generate_prompt = """
        You are an assistant for question-answering tasks. 
        Use the following pieces of retrieved context to answer the question. 
        If you don't know the answer, just say that you don't know. 
        Use three sentences maximum and keep the answer concise.
        Question: {question} 
        Context: {context}
        """

    response_model = init_chat_model(model=model, temperature=0)

    question = state["messages"][0].content
    context = state["messages"][-1].content
    prompt = generate_prompt.format(question=question, context=context)
    response = response_model.invoke([{"role": "user", "content": prompt}])
    return MessagesState(messages=[response])


# Build the graph
# - Start with a generate_query_or_respond and determine if we need to call retriever_tool
# - Route to next step using tools_condition:
#   - If generate_query_or_respond returned tool_calls, call retriever_tool to retrieve context
#   - Otherwise, respond directly to the user
# - Grade retrieved document content for relevance to the question (grade_documents) and route to next step:
#   - If not relevant, rewrite the question using rewrite_question and then call generate_query_or_respond again
#   - If relevant, proceed to generate_answer and generate final response using the ToolMessage with the retrieved
#   document context
def _build_graph(model: str, retriever_tool: Tool):
    def grade_documents(state: MessagesState):
        return _grade_documents(model, state)

    def generate_query_or_respond(state: MessagesState):
        return _generate_query_or_respond(retriever_tool, state)

    def rewrite_question(state: MessagesState):
        return _rewrite_question(model, state)

    def generate_answer(state: MessagesState):
        return _generate_answer(model, state)

    workflow = StateGraph(MessagesState)

    # Define the nodes we will cycle between
    workflow.add_node(generate_query_or_respond)
    workflow.add_node("retrieve", ToolNode([retriever_tool]))
    workflow.add_node(rewrite_question)
    workflow.add_node(generate_answer)

    workflow.add_edge(START, "generate_query_or_respond")

    # Decide whether to retrieve
    workflow.add_conditional_edges(
        "generate_query_or_respond",
        # Assess LLM decision (call `retriever_tool` tool or respond to the user)
        tools_condition,
        {
            # Translate the condition outputs to nodes in our graph
            "tools": "retrieve",
            END: END,
        },
    )

    # Edges taken after the `action` node is called.
    workflow.add_conditional_edges(
        "retrieve",
        # Assess agent decision
        grade_documents,
    )

    workflow.add_edge("generate_answer", END)
    workflow.add_edge("rewrite_question", "generate_query_or_respond")

    # Compile
    return workflow.compile()


def main():
    docs_embeddings = _pre_process_docs()
    retriever_tool = _create_retriever_tool(docs_embeddings)

    # Try it on a random input
    input_messages = MessagesState(messages=[{"role": "user", "content": "hello!"}])
    # _generate_query_or_respond(retriever_tool, input_messages)["messages"][-1].pretty_print()

    # Ask a question that requires semantic search
    input_messages = MessagesState(
        messages=[
            {
                "role": "user",
                "content": "What does Lilian Weng say about types of reward hacking?",
            }
        ]
    )
    # _generate_query_or_respond(retriever_tool, input_messages)["messages"][-1].pretty_print()

    # Run this with irrelevant documents in the tool response
    input_messages = MessagesState(
        messages=convert_to_messages(
            [
                {
                    "role": "user",
                    "content": "What does Lilian Weng say about types of reward hacking?",
                },
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": "1",
                            "name": "retrieve_blog_posts",
                            "args": {"query": "types of reward hacking"},
                        }
                    ],
                },
                {"role": "tool", "content": "meow", "tool_call_id": "1"},
            ]
        )
    )
    # graded_docs = _grade_documents(model="google_genai:gemini-2.0-flash", state=input_messages)
    # print(f"graded docs result: {graded_docs}\n")

    # Confirm that the relevant documents are classified as such
    input_messages = MessagesState(
        messages=convert_to_messages(
            [
                {
                    "role": "user",
                    "content": "What does Lilian Weng say about types of reward hacking?",
                },
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": "1",
                            "name": "retrieve_blog_posts",
                            "args": {"query": "types of reward hacking"},
                        }
                    ],
                },
                {
                    "role": "tool",
                    "content": "reward hacking can be categorized into two types: environment or goal misspecification, and reward tampering",
                    "tool_call_id": "1",
                },
            ]
        )
    )
    # graded_docs = _grade_documents(model="google_genai:gemini-2.0-flash", state=input_messages)
    # print(f"graded docs result: {graded_docs}\n")

    # Confirm that the relevant documents are classified as such
    input_messages = MessagesState(
        messages=convert_to_messages(
            [
                {
                    "role": "user",
                    "content": "What does Lilian Weng say about types of reward hacking??",
                },
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": "1",
                            "name": "retrieve_blog_posts",
                            "args": {"query": "types of reward hacking"},
                        }
                    ],
                },
                {"role": "tool", "content": "meow", "tool_call_id": "1"},
            ]
        )
    )
    # response = _rewrite_question(model="google_genai:gemini-2.0-flash", state=input_messages)
    # print(response["messages"][-1]["content"])

    # Try to generate an answer
    input_messages = MessagesState(
        messages=convert_to_messages(
            [
                {
                    "role": "user",
                    "content": "What does Lilian Weng say about types of reward hacking??",
                },
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": "1",
                            "name": "retrieve_blog_posts",
                            "args": {"query": "types of reward hacking"},
                        }
                    ],
                },
                {
                    "role": "tool",
                    "content": "reward hacking can be categorized into two types: environment or goal misspecification, and reward tampering",
                    "tool_call_id": "1",
                },
            ]
        )
    )

    response = _generate_answer(
        model="google_genai:gemini-2.0-flash", state=input_messages
    )
    response["messages"][-1].pretty_print()

    retrieval_graph = _build_graph(
        model="google_genai:gemini-2.0-flash", retriever_tool=retriever_tool
    )

    for chunk in retrieval_graph.stream(
        {
            "messages": [
                {
                    "role": "user",
                    "content": "What does Lilian Weng say about types of reward hacking?",
                }
            ]
        }
    ):
        for node, update in chunk.items():
            print("Update from node", node)
            update["messages"][-1].pretty_print()
            print("\n\n")


if __name__ == "__main__":
    _set_env("GOOGLE_API_KEY")
    main()
