import asyncio
import os

from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.prebuilt import create_react_agent

# Model Context Protocol (MCP) is an open protocol that standardizes how applications provide tools and context to
# language models. LangGraph agents can use tools defined on MCP servers through the langchain-mcp-adapters library.


# The langchain-mcp-adapters package enables agents to use tools defined across one or more MCP servers.
# To create your own MCP servers, you can use the mcp library.
# This library provides a simple way to define tools and run them as servers.
async def main():
    # Get the project root directory
    project_root = os.path.dirname(os.path.abspath(__file__))

    # Initialize a MultiServerMCPClient with MCP servers connections.
    client = MultiServerMCPClient(
        {
            "math": {
                "command": "python",
                "args": [os.path.join(project_root, "math_server.py")],
                "transport": "stdio",
            },
            "weather": {
                "url": "http://localhost:8000/mcp",
                "transport": "streamable_http",
            },
            "time": {
                "command": "python",
                "args": ["-m", "mcp_server_time", "--local-timezone=UTC"],
                "transport": "stdio",
            },
        }
    )

    # Get a list of all tools from all connected servers.
    # NOTE: a new session will be created for each tool call
    tools = await client.get_tools()

    # Create a LangChain agent that uses the tools defined on the MCP servers.
    agent = create_react_agent("google_genai:gemini-2.0-flash", tools)

    # Use the agent to interact with the MCP servers.
    math_response = await agent.ainvoke(
        {"messages": [{"role": "user", "content": "what's (3 + 5) x 12?"}]}
    )

    weather_response = await agent.ainvoke(
        {"messages": [{"role": "user", "content": "what is the weather in NYC?"}]}
    )

    time_response = await agent.ainvoke(
        {"messages": [{"role": "user", "content": "what the time in NYC?"}]}
    )

    print(math_response["messages"][-1].content)
    print(weather_response["messages"][-1].content)
    print(time_response["messages"][-1].content)


if __name__ == "__main__":
    asyncio.run(main())
