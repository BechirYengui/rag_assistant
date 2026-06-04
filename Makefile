.PHONY: help install dev db up down run ingest test lint fmt

help:
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN{FS=":.*?## "}{printf "\033[36m%-12s\033[0m %s\n", $$1, $$2}'

install:  ## Install runtime dependencies
	pip install -r requirements.txt

dev:  ## Install dev + runtime dependencies
	pip install -r requirements-dev.txt

db:  ## Start only the pgvector database
	docker compose up -d db

up:  ## Build and start the full stack (db + api)
	docker compose up --build

down:  ## Stop and remove containers
	docker compose down

run:  ## Run the API locally with autoreload (needs a running db)
	uvicorn app.main:app --reload

ingest:  ## Ingest the sample documents (needs a running db)
	python -m scripts.ingest_cli sample_docs/*.md

test:  ## Run the test suite
	pytest -q

lint:  ## Lint with ruff
	ruff check app tests scripts

fmt:  ## Auto-format with ruff
	ruff format app tests scripts
