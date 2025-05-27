# GLOBAL
##########################################################################

lint:
	uv run ruff format . && uv run ruff check . --fix

sync:
	uv sync

# BASICS
##########################################################################

basics-chat:
	uv run python -m src.basics.chat.main

basics-tools:
	uv run python -m src.basics.tools.main

basics-memory:
	uv run python -m src.basics.memory.main

basics-human:
	uv run python -m src.basics.human.main

basics-state:
	uv run python -m src.basics.state.main

basics-time:
	uv run python -m src.basics.time.main

# AGENTS
##########################################################################

agents-handoffs:
	uv run python -m src.agents.handoffs.main

agents-supervisor:
	uv run python -m src.agents.supervisor.main

agents-swarm:
	uv run python -m src.agents.swarm.main

agents-retrieval:
	uv run python -m src.agents.retrieval.main

agents-multiagent:
	uv run python -m src.agents.multiagent.main

# EXAMPLES
##########################################################################

stocks-chatbot:
	uv run python -m src.examples.stocks_chatbot.main
