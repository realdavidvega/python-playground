# GLOBAL
##########################################################################

CHECK_DIRS := .

check:
	uv run pyrefly check
	uv run ruff check
	# uv run ty check

format:
	uv run ruff format $(CHECK_DIRS)

fixes:
	uv run ruff format $(CHECK_DIRS)
	uv run ruff check $(CHECK_DIRS) --fix
	uv run pyrefly check
	# uv run ty check

sync:
	uv sync

# INFRA
##########################################################################

infra-up:
	@if ! docker compose -p py-playground up --remove-orphans --wait; then \
        		echo "Failed to start containers"; \
        		docker compose -p py-playground logs; \
        		exit 1; \
    	fi

infra-down:
	docker-compose -p py-playground down

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
