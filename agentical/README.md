# Agentical

**Agentic framework built on Pydantic AI for creating powerful AI agents and workflows**

[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-green.svg)](https://fastapi.tiangolo.com/)
[![Pydantic AI](https://img.shields.io/badge/Pydantic%20AI-latest-orange.svg)](https://ai.pydantic.dev/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Overview

Agentical is a modern, production-ready framework for building intelligent AI agents and workflows. Built on top of Pydantic AI and FastAPI, it provides a robust foundation for creating scalable agentic systems with comprehensive observability, security, and testing capabilities.

## Features

### 🤖 Agent System
- **Base Agent Architecture**: Extensible base classes for creating custom agents
- **Agent Registry**: Centralized management and discovery of agent instances
- **Generic & Super Agents**: Pre-built agent types for common use cases
- **State Management**: Built-in agent state tracking and persistence

### 🚀 FastAPI Integration
- **RESTful API**: Complete API endpoints for agent management and execution
- **Health Checks**: Comprehensive health monitoring endpoints
- **Async Support**: Full asynchronous operation support
- **Auto Documentation**: Generated OpenAPI/Swagger documentation

### 🔍 Observability & Monitoring
- **Logfire Integration**: Advanced observability with Pydantic Logfire
- **Structured Logging**: Context-aware logging with request tracing
- **Performance Metrics**: Request timing and performance monitoring
- **Error Tracking**: Comprehensive error logging and debugging

### 🔒 Security & Middleware
- **Rate Limiting**: Configurable rate limiting with Redis backend
- **Security Headers**: CORS, CSP, and other security headers
- **Request Validation**: Input sanitization and validation
- **Bot Protection**: Advanced bot detection and mitigation

### 🗄️ Database & Storage
- **SQLAlchemy Integration**: Async and sync database operations
- **Repository Pattern**: Clean data access layer implementation
- **User Management**: Complete user authentication and authorization
- **Caching**: Redis-based caching for performance optimization

### 🧪 Testing & Quality
- **PyTest Framework**: Comprehensive test suite with 90%+ coverage
- **Test Fixtures**: Pre-configured fixtures for database and async testing
- **Integration Tests**: Full API endpoint testing
- **Mock Utilities**: Helper utilities for testing external dependencies

## Quick Start

### Prerequisites

- Python 3.12 or higher
- Redis (for caching and rate limiting)
- PostgreSQL or SQLite (for database)

### Installation

```bash
# Clone the repository
git clone https://github.com/devq-ai/agentical.git
cd agentical

# Install dependencies
pip install -e .

# Set up environment variables
cp .env.example .env
# Edit .env with your configuration
```

### Environment Configuration

Create a `.env` file with the following variables:

```bash
# FastAPI Configuration
DEBUG=true
ENVIRONMENT=development
SECRET_KEY=your-secret-key-here

# Logfire Observability
LOGFIRE_TOKEN=your-logfire-token
LOGFIRE_PROJECT_NAME=agentical-dev
LOGFIRE_SERVICE_NAME=agentical-api

# Database Configuration
DATABASE_URL=postgresql+asyncio://user:password@localhost/agentical
REDIS_URL=redis://localhost:6379/0

# AI Configuration
ANTHROPIC_API_KEY=your-anthropic-api-key
MODEL=claude-3-7-sonnet-20250219
```

### Running the Application

```bash
# Start the FastAPI server
python main.py

# Or use uvicorn directly
uvicorn main:app --reload --port 8000
```

The API will be available at `http://localhost:8000` with interactive documentation at `http://localhost:8000/docs`.

## Usage Examples

### Creating a Custom Agent

```python
from agentical.agents.base_agent import BaseAgent
from agentical.agents.agent_registry import AgentRegistry

class CustomAgent(BaseAgent):
    """Custom agent for specific tasks."""
    
    def __init__(self, metadata):
        super().__init__(metadata)
        # Initialize custom agent logic
    
    async def execute(self, input_data):
        # Implement agent execution logic
        return {"result": "processed"}

# Register the agent
registry = AgentRegistry()
agent = registry.create_agent(
    agent_id="custom-001",
    agent_type="custom",
    metadata={"name": "Custom Agent", "version": "1.0"}
)
```

### Using the API

```python
import httpx

# Execute an agent
response = httpx.post(
    "http://localhost:8000/api/agents/execute",
    json={
        "agent_id": "custom-001",
        "input_data": {"query": "process this data"}
    }
)

result = response.json()
print(result)
```

### Database Operations

```python
from agentical.db.repositories.user import UserRepository
from agentical.db.models.user import User

# Create a user
user_repo = UserRepository(db_session)
user = user_repo.create_user(
    data={"username": "testuser", "email": "test@example.com"},
    password="secure_password"
)

# Authenticate user
authenticated_user = user_repo.authenticate("testuser", "secure_password")
```

## Architecture

### Core Components

```
agentical/
├── agents/          # Agent system and registry
├── api/             # FastAPI routes and endpoints
├── core/            # Core utilities and exceptions
├── db/              # Database models and repositories
├── middlewares/     # Security and logging middleware
├── main.py          # Application entry point
└── pyproject.toml   # Project configuration
```

### Agent Lifecycle

1. **Registration**: Agents are registered with the AgentRegistry
2. **Initialization**: Agent metadata and infrastructure setup
3. **Execution**: Request processing with observability
4. **State Management**: Agent state persistence and tracking
5. **Cleanup**: Resource cleanup and metrics collection

### Middleware Stack

1. **Security Headers**: CORS, CSP, security headers
2. **Rate Limiting**: Request rate limiting and throttling
3. **Request Logging**: Structured logging with context
4. **Error Handling**: Centralized error processing
5. **Health Check Filtering**: Health endpoint optimization

## Development

### Setting Up Development Environment

```bash
# Install development dependencies
pip install -e ".[dev]"

# Run tests
pytest

# Run tests with coverage
pytest --cov=agentical --cov-report=html

# Format code
black agentical/
isort agentical/

# Type checking
mypy agentical/
```

### Project Standards

- **Code Style**: Black formatter, 88 character line length
- **Import Order**: isort with Google-style docstrings
- **Type Hints**: Comprehensive type annotations
- **Testing**: 90% minimum test coverage
- **Documentation**: Docstrings for all public APIs

### Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes with tests
4. Ensure all tests pass (`pytest`)
5. Commit your changes (`git commit -m 'Add amazing feature'`)
6. Push to the branch (`git push origin feature/amazing-feature`)
7. Open a Pull Request

## Configuration

### Logfire Observability

Agentical integrates with Pydantic Logfire for comprehensive observability:

```python
import logfire

# Automatic FastAPI instrumentation
logfire.configure(
    service_name="agentical-api",
    service_version="1.0.0"
)
logfire.instrument_fastapi(app)
```

### Security Configuration

```python
from agentical.middlewares.security import (
    RateLimitConfig,
    SecurityHeadersMiddleware
)

# Rate limiting configuration
rate_limit_config = RateLimitConfig(
    requests_per_minute=100,
    burst_requests=20
)

# Security headers
app.add_middleware(SecurityHeadersMiddleware)
```

## API Reference

### Agent Endpoints

- `POST /api/agents/execute` - Execute an agent
- `GET /api/agents/{agent_id}` - Get agent information
- `GET /api/agents` - List all agents
- `POST /api/agents/register` - Register a new agent

### Health Endpoints

- `GET /health` - Basic health check
- `GET /ready` - Readiness probe
- `GET /live` - Liveness probe

For complete API documentation, visit `/docs` when the server is running.

## Deployment

### Docker

```dockerfile
FROM python:3.12-slim

WORKDIR /app
COPY . .
RUN pip install -e .

EXPOSE 8000
CMD ["python", "main.py"]
```

### Production Configuration

- Use environment-specific configuration files
- Enable SSL/TLS termination
- Configure database connection pooling
- Set up monitoring and alerting
- Implement backup and recovery procedures

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Support

- **Issues**: [GitHub Issues](https://github.com/devq-ai/agentical/issues)
- **Discussions**: [GitHub Discussions](https://github.com/devq-ai/agentical/discussions)
- **Documentation**: [Project Wiki](https://github.com/devq-ai/agentical/wiki)

## Acknowledgments

- Built with [Pydantic AI](https://ai.pydantic.dev/)
- Powered by [FastAPI](https://fastapi.tiangolo.com/)
- Observability by [Logfire](https://logfire.pydantic.dev/)
- Part of the [DevQ.ai](https://github.com/devq-ai) ecosystem

---

**DevQ.ai** - Engineered for AI-Assisted Development Excellence