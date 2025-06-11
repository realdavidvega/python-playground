import asyncio

from langchain_mcp_adapters.client import MultiServerMCPClient
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

    tools = await client.get_tools()
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
