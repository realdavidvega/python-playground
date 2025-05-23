import uuid

from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent
from langgraph_supervisor import create_supervisor


# Supervisor
# Individual agents are coordinated by a central supervisor agent. The supervisor controls all communication
# flow and task delegation, making decisions about which agent to invoke based on the current context and
# task requirements.


@tool
def book_hotel(hotel_name: str):
    """
    Book a hotel.
    Do not call this tool more than once.
    """
    return f"Successfully booked a stay at {hotel_name}."


@tool
def book_flight(from_airport: str, to_airport: str):
    """
    Book a flight.
    Do not call this tool more than once.
    """
    return f"Successfully booked a flight from {from_airport} to {to_airport}."


def main():
    flight_assistant = create_react_agent(
        model="google_genai:gemini-2.0-flash",
        tools=[book_flight],
        prompt="You are a flight booking assistant, do not call your tools more than once.",
        name="flight_assistant",
    )

    hotel_assistant = create_react_agent(
        model="google_genai:gemini-2.0-flash",
        tools=[book_hotel],
        prompt="You are a hotel booking assistant, do not call your tools more than once.",
        name="hotel_assistant",
    )

    # create a supervisor (graph), and compile it adding a memory saver so we can continue where we left off
    supervisor = create_supervisor(
        agents=[flight_assistant, hotel_assistant],
        model=ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0),
        prompt=(
            "You manage a hotel booking assistant and a"
            "flight booking assistant. Assign work to them."
            "Do not call any agent more than once per task."
        ),
    ).compile(checkpointer=MemorySaver())

    # adding a thread id to the config
    config = RunnableConfig(configurable={"thread_id": uuid.uuid4().hex})

    def stream_updates(user_content: str):
        for chunk in supervisor.stream(
            {"messages": [{"role": "user", "content": user_content}]},
            config=config,
            stream_mode="values"
        ):
            print(f"Output: {chunk}\n")

    while True:
        # Book a flight from BOS to JFK and a stay at McKittrick Hotel
        user_input = input("User: ")
        if user_input.lower() in ["quit", "exit", "q"]:
            print("Goodbye!")
            break

        stream_updates(user_input)


if __name__ == "__main__":
    main()
