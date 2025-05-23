from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent
from langgraph_swarm import create_handoff_tool, create_swarm


# Swarm
# Agents dynamically hand off control to one another based on their specializations. The system remembers
# which agent was last active, ensuring that on subsequent interactions, the conversation resumes with that agent.


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
    transfer_to_hotel_assistant = create_handoff_tool(
        agent_name="hotel_assistant",
        description="Transfer user to the hotel-booking assistant.",
    )

    transfer_to_flight_assistant = create_handoff_tool(
        agent_name="flight_assistant",
        description="Transfer user to the flight-booking assistant.",
    )

    flight_assistant = create_react_agent(
        model="google_genai:gemini-2.0-flash",
        tools=[book_flight, transfer_to_hotel_assistant],
        prompt="You are a flight booking assistant, do not call your tools more than once.",
        name="flight_assistant",
    )

    hotel_assistant = create_react_agent(
        model="google_genai:gemini-2.0-flash",
        tools=[book_hotel, transfer_to_flight_assistant],
        prompt="You are a hotel booking assistant, do not call your tools more than once.",
        name="hotel_assistant",
    )

    swarm = create_swarm(
        agents=[flight_assistant, hotel_assistant],
        default_active_agent="flight_assistant",
    ).compile()

    def stream_updates(user_content: str):
        for chunk in swarm.stream(
            {"messages": [{"role": "user", "content": user_content}]}
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
