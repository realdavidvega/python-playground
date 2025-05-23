from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.prebuilt import create_react_agent
from langgraph_supervisor import create_supervisor


# Supervisor
# individual agents are coordinated by a central supervisor agent. The supervisor controls all communication
# flow and task delegation, making decisions about which agent to invoke based on the current context and
# task requirements.


def book_hotel(hotel_name: str):
    """
    Book a hotel.
    Do not call this tool more than once.
    """
    return f"Successfully booked a stay at {hotel_name}."


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
        prompt="You are a flight booking assistant",
        name="flight_assistant",
    )

    hotel_assistant = create_react_agent(
        model="google_genai:gemini-2.0-flash",
        tools=[book_hotel],
        prompt="You are a hotel booking assistant",
        name="hotel_assistant",
    )

    supervisor = create_supervisor(
        agents=[flight_assistant, hotel_assistant],
        model=ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0),
        prompt=(
            "You manage a hotel booking assistant and a"
            "flight booking assistant. Assign work to them."
            "Do not call any tool more than once."
        ),
    ).compile()

    def stream_updates(user_content: str):
        for chunk in supervisor.stream(
            {"messages": [{"role": "user", "content": user_content}]}
        ):
            print(f"Output: {chunk}\n")

    while True:
        # book a flight from BOS to JFK and a stay at McKittrick Hotel
        user_input = input("User: ")
        if user_input.lower() in ["quit", "exit", "q"]:
            print("Goodbye!")
            break

        stream_updates(user_input)


if __name__ == "__main__":
    main()
