# GLOBAL
##########################################################################

lint:
	uv run ruff format . && uv run ruff check . --fix

check:
	uv run ty check

sync:
	uv sync

# BASICS
##########################################################################

basics-chat:
	uv run python -m src.basics.chat

basics-tools:
	uv run python -m src.basics.tools

basics-memory:
	uv run python -m src.basics.memory

basics-human:
	uv run python -m src.basics.human

basics-state:
	uv run python -m src.basics.state

basics-time:
	uv run python -m src.basics.time

# AGENTS
##########################################################################

agents-handoffs:
	uv run python -m src.agents.handoffs

agents-supervisor:
	uv run python -m src.agents.supervisor

agents-swarm:
	uv run python -m src.agents.swarm

agents-retrieval:
	uv run python -m src.agents.retrieval

agents-multiagent:
	uv run python -m src.agents.multiagent

# MCP
##########################################################################

mcp-example:
	uv run python -m src.mcp.example.main

mcp-github:
	uv run python -m src.mcp.integrations.github

# EXAMPLES
##########################################################################

stocks-chatbot:
	uv run python -m src.examples.stocks_chatbot.main

spy-signals:
	uv run python -m src.examples.spy_signals.main
