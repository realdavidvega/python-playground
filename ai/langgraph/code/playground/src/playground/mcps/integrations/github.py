import asyncio

from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_mcp_adapters.tools import load_mcp_tools
from langgraph.prebuilt import create_react_agent


async def main():
    client = MultiServerMCPClient(
        {
            "github": {
                "url": "http://127.0.0.1:8001/sse",
                "transport": "sse",
            }
        }
    )

    # Different approach, opening a session for each MCP instead
    async with client.session("github") as session:
        # Load tools
        tools = await load_mcp_tools(session)

        # Create agent as usual
        agent = create_react_agent("google_genai:gemini-2.0-flash", tools)

        github_response = await agent.ainvoke(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": "what was the last pull request in realdavidvega/python-playground?",
                    }
                ]
            }
        )

        print(github_response["messages"][-1].content)


if __name__ == "__main__":
    asyncio.run(main())
