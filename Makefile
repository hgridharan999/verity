.PHONY: help install dev-install run test lint format clean docker-build docker-up docker-down migrate init

help:
	@echo "Verity - Content Verification Engine"
	@echo ""
	@echo "Available commands:"
	@echo "  make install        - Install production dependencies"
	@echo "  make dev-install    - Install development dependencies"
	@echo "  make run            - Run the application locally"
	@echo "  make test           - Run tests"
	@echo "  make test-cov       - Run tests with coverage"
	@echo "  make lint           - Run linters"
	@echo "  make format         - Format code with black and isort"
	@echo "  make clean          - Clean temporary files"
	@echo "  make docker-build   - Build Docker images"
	@echo "  make docker-up      - Start Docker services"
	@echo "  make docker-down    - Stop Docker services"
	@echo "  make docker-logs    - View Docker logs"
	@echo "  make migrate        - Run database migrations"
	@echo "  make migration      - Create new migration"
	@echo "  make init           - Initialize database and directories"
	@echo "  make shell          - Open Python shell with app context"

install:
	pip install -r requirements.txt

dev-install:
	pip install -r requirements-dev.txt

run:
	uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

test:
	pytest

test-cov:
	pytest --cov=app --cov-report=html --cov-report=term

lint:
	flake8 app tests
	mypy app

format:
	black app tests
	isort app tests

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type f -name "*.log" -delete
	rm -rf .pytest_cache .coverage htmlcov .mypy_cache

docker-build:
	docker-compose build

docker-up:
	docker-compose up -d
	@echo "Waiting for services to be ready..."
	@sleep 5
	@echo "Services are up!"
	@echo "API: http://localhost:8000"
	@echo "Docs: http://localhost:8000/docs"
	@echo "Health: http://localhost:8000/api/v1/health"

docker-down:
	docker-compose down

docker-logs:
	docker-compose logs -f app

docker-shell:
	docker-compose exec app /bin/bash

migrate:
	alembic upgrade head

migration:
	@read -p "Enter migration message: " msg; \
	alembic revision --autogenerate -m "$$msg"

downgrade:
	alembic downgrade -1

init:
	python scripts/init_db.py

shell:
	python -i -c "from app.config import settings; from app.db.session import AsyncSessionLocal; print('Verity shell loaded. Available: settings, AsyncSessionLocal')"

db-reset:
	@echo "WARNING: This will delete all data!"
	@read -p "Are you sure? [y/N] " confirm; \
	if [ "$$confirm" = "y" ]; then \
		alembic downgrade base; \
		alembic upgrade head; \
		echo "Database reset complete"; \
	fi

# Development workflow
dev: dev-install
	@echo "Development environment ready!"

# Full local setup
setup: dev-install init migrate
	@echo "Setup complete! Run 'make run' to start the application"

# Docker setup
docker-setup: docker-build docker-up
	@echo "Waiting for database..."
	@sleep 10
	docker-compose exec app alembic upgrade head
	@echo "Docker setup complete!"
	@echo "Access the API at http://localhost:8000/docs"

# Production build
build-prod:
	docker build -t verity:latest -t verity:$(VERSION) .

# Check code quality
check: lint test
	@echo "All checks passed!"
