# Claude Configuration

## Project Information

- **Organization**: DevQ.ai (https://github.com/devq-ai)
- **Developer**: Dion Edge <dion@devq.ai>
- **Project Structure**: Multi-repository monorepo with five-component stack
- **Rules Reference**: [Zed IDE Agent Rules](./zed_agent_rules.md)

## DevQ.ai Standard Five-Component Stack

### 1. FastAPI Foundation Framework
- Modern, fast web framework with automatic API documentation
- Dependency injection and middleware patterns
- RESTful API design with OpenAPI specifications
- Pydantic model validation and serialization

### 2. Logfire Observability (Required)
- Comprehensive observability and monitoring integration
- Automatic FastAPI, SQLAlchemy, and HTTP client instrumentation
- Structured logging with spans and context
- Performance metrics and error tracking

### 3. PyTest Build-to-Test Development
- Test-driven development with 90% minimum coverage
- Integration tests for all API endpoints
- Async test clients and database fixtures
- Continuous testing during development

### 4. TaskMaster AI Project Management
- Task-driven development via MCP integration
- Iterative task breakdown and dependency management
- Progress tracking and milestone management
- AI-assisted project planning and execution

### 5. MCP Server Integration
- Model Context Protocol for AI-enhanced development
- Comprehensive server ecosystem for specialized tasks
- Real-time communication and tool orchestration
- Cross-server compatibility and communication

## Claude Code Core Development Tools (Built-in)

- `task` - Launch specialized agents for complex searches and analysis
- `bash` - Execute shell commands with git, file operations, and system tasks
- `glob` - Fast file pattern matching (e.g., **/*.js, src/**/*.ts)
- `grep` - Content search using regex patterns across codebases
- `ls` - Directory listing and file system navigation
- `read` - File content reading including images and notebooks
- `edit/multiedit` - Precise file editing with find-and-replace
- `write` - Create new files and overwrite existing ones
- `todoread/todowrite` - Task management and progress tracking

## MCP (Model Context Protocol) Servers Available

### NPX-Based Core Servers

- `filesystem` - File read/write operations for project directories
- `git` - Version control operations, commits, and DevQ.ai workflow integration
- `fetch` - API calls and external resource access
- `memory` - Persistent memory across development sessions
- `sequentialthinking` - Enhanced step-by-step problem solving
- `github` - GitHub API integration for devq-ai organization
- `inspector` - Debug MCP server connections

### DevQ.ai Python-Based Servers

- `taskmaster-ai` - Project management and task-driven development
- `ptolemies` - Custom knowledge base with SurrealDB integration
- `context7` - Advanced contextual reasoning with Redis memory
- `bayes` - Bayesian inference and statistical modeling
- `crawl4ai` - Intelligent web scraping and content extraction
- `dart` - Dart AI integration for smart code assistance
- `surrealdb` - Multi-model database operations with graph capabilities
- `logfire` - Enhanced observability integration and monitoring

### Specialized Development Servers

- `agentql` - Web automation using natural language queries
- `calendar` - Google Calendar integration for project scheduling
- `jupyter` - Notebook execution and data science workflows
- `stripe` - Payment processing integration for SaaS projects
- `shadcn-ui` - React component library integration
- `magic` - AI-powered code generation utilities

### Scientific Computing & Solvers

- `solver-z3` - Z3 theorem prover for formal verification
- `solver-pysat` - Boolean satisfiability problem solver
- `solver-mzn` - MiniZinc constraint satisfaction solver

### Registry & Infrastructure

- `registry` - Official MCP server registry and discovery
- `memory` - Persistent memory management with contextual recall

## Key Development Capabilities

### Multi-Modal Development
- Handle text, code, images, and structured data
- Support for Python, TypeScript, and configuration files
- Integration with Jupyter notebooks and documentation

### Database Integration
- **SurrealDB**: Multi-model database (primary)
- **Redis**: Caching and session management
- **Neo4j**: Graph database for knowledge graphs
- **PostgreSQL**: Relational data when needed

### AI/ML Development Tools
- Multiple LLM provider integration (Anthropic, OpenAI, Google)
- Genetic algorithm optimization (PyGAD)
- Scientific computing (NumPy, PyTorch, PyMC)
- Statistical modeling and analysis

### Web & Automation
- Web scraping and content extraction
- Browser automation for testing
- API integration and testing
- Real-time data processing

### Full-Stack Development
- FastAPI backend development
- Next.js frontend with TypeScript
- Tailwind CSS and Shadcn UI components
- Docker containerization and deployment

## Required Project Configuration Files

### Standard Project Structure
```
{project_name}/
├── .claude/
│   └── settings.local.json      # Claude Code permissions and MCP discovery
├── .git/
│   └── config                   # Git configuration with DevQ.ai team settings
├── .logfire/
│   └── logfire_credentials.json # Logfire observability credentials
├── .zed/
│   └── settings.json            # Zed IDE configuration with MCP servers
├── mcp/
│   └── mcp-servers.json         # MCP server registry definitions
├── src/                         # Source code directory
├── tests/                       # PyTest test suite
├── .env                         # Environment variables
├── .gitignore                   # Git ignore patterns (DevQ.ai standard)
├── .rules                       # Development rules and guidelines
├── requirements.txt             # Python dependencies
├── docker-compose.yml           # Container orchestration
└── main.py                      # FastAPI application entry point
```

### Configuration File Integration

**`.claude/settings.local.json`** (Claude Code Configuration)
- Enables MCP server auto-discovery: `"enableAllProjectMcpServers": true`
- Sets bash command permissions for security
- Does NOT define which MCP servers to load - enables discovery only

**`.git/config`** (DevQ.ai Team Configuration)
```ini
[user]
    name = DevQ.ai Team
    email = dion@devq.ai
[remote "origin"]
    url = https://github.com/devq-ai/{project-name}.git
```

**`.logfire/logfire_credentials.json`** (Observability Configuration)
```json
{
  "token": "pylf_v1_us_...",
  "project_name": "devq-ai-{project}",
  "project_url": "https://logfire-us.pydantic.dev/devq-ai/devq-ai-{project}",
  "logfire_api_url": "https://logfire-us.pydantic.dev"
}
```

**`.zed/settings.json`** (Zed IDE Configuration)
- Terminal configuration with DevQ.ai environment sourcing
- MCP servers for TaskMaster AI and Dart integration
- Python/TypeScript formatting and LSP settings
- Project-specific environment variables

**`mcp/mcp-servers.json`** (MCP Server Registry)
- Actual MCP server definitions - the authoritative configuration
- Executable specifications with commands, arguments, environment variables
- Real server registry that Claude Code loads and executes

## Tool Access Architecture

### Configuration Chain
1. **Claude Code** reads `mcp-servers.json` for actual MCP server definitions
2. **`.claude/settings.local.json`** enables auto-discovery of those servers
3. **`CLAUDE.md`** documents what should be available (reference specification)

### Built-in vs MCP Tools
- **Built-in tools** (`task`, `bash`, `glob`, etc.) are always available
- **MCP tools** require the configuration chain above to be accessible
- **TaskMaster AI** integration available through both CLI and MCP protocols

## Development Environment Integration

### DevQ.ai Development Environment
```bash
# Available after sourcing .zshrc.devqai
🚀 DevQ.ai development environment loaded!
📁 Project root: /Users/dionedge/devqai
🔧 Available commands:
   • Quick nav: ag, bayes, darwin, nash, ptolemies, breiman, gompertz, tokenator
   • Zoxide: z <dir>, zi (interactive), zz (back), zq (query), zr (remove)
   • MCP: start-context7, start-crawl4ai, start-ptolemies, start-dart, mcp-inspect
   • Dev tools: devq-test, devq-format, devq-lint, start-surreal, verify-db
   • Utils: new-component, find-dir, find-edit, show_env_vars, dart-test
```

### Hardware & Software Configuration
- **Hardware**: Mac Studio M2 Max (12 cores, 32GB RAM)
- **OS**: macOS
- **IDE**: Zed Editor with comprehensive MCP integration
- **Terminal**: Ghostty with DevQ.ai environment configuration
- **AI Models**: Claude 4 Sonnet (primary), Claude 3.7 Sonnet (backup)

### Development Standards
- **Python**: 3.12 with Black formatting (88 char line length)
- **TypeScript**: Strict mode, ES2022 target, 100 char line length
- **Code Quality**: 90% test coverage minimum, comprehensive type hints
- **Git Workflow**: Feature branches, squash merges, CodeRabbit AI reviews
- **Security**: Environment variables, never hardcode credentials

## Project Standards

### Code Formatting (Language-Specific)
- **Backend Python**: 88 character line length, Black formatter, Google-style docstrings
- **Frontend TypeScript**: 100 character line length, single quotes, required semicolons
- **Import Order**: typing → standard → third-party → first-party → local

### Documentation Guidelines
- All public APIs must have complete documentation
- Follow Google-style docstrings for Python, JSDoc for TypeScript
- Document React components with props and state descriptions
- Include code examples for non-trivial functions
- Maintain API documentation with OpenAPI/Swagger

### Project-Specific Integration
- **Agent Components**: Must implement error handling for all tool usage
- **Frontend Components**: Must support both light and dark themes
- **API Endpoints**: Must include proper validation and error responses
- **Database Operations**: Use connection pooling and transaction management

### Testing Requirements
- **Framework**: PyTest for Python, Vitest for TypeScript/Frontend
- **Coverage**: Minimum 90% line coverage for all projects
- **Types**: Unit tests, integration tests, end-to-end tests
- **API Testing**: All endpoints must have comprehensive test coverage
- **Mock Strategy**: External services mocked, database fixtures for integration

## MCP Integration Patterns

### TaskMaster AI Development Workflow
1. **Session Initialization**: `get_tasks` to review current work
2. **Task Selection**: `next_task` to identify next priority item
3. **Task Breakdown**: `expand_task --research` for complex features
4. **Progress Tracking**: `set_task_status` as work completes
5. **Discovery**: `add_task` for newly identified requirements

### Available MCP Capabilities by Category

**Project Management**
- TaskMaster AI: Complete project lifecycle management
- Dart AI: Advanced project planning and resource allocation

**Knowledge Management**
- Ptolemies: SurrealDB-based knowledge base with semantic search
- Context7: Advanced contextual reasoning with Redis memory
- Memory: Persistent session memory across development cycles

**Development Assistance**
- Magic: AI-powered code generation and refactoring
- Shadcn-UI: React component library integration
- AgentQL: Natural language web automation

**Data & Analytics**
- Bayes: Bayesian statistical modeling and inference
- Jupyter: Notebook execution and data science workflows
- SurrealDB: Multi-model database operations

**Infrastructure & Deployment**
- Logfire: Enhanced observability and monitoring
- Registry: MCP server discovery and management
- Inspector: Debug and troubleshoot MCP connections

## Environment Variables Configuration

### Core Application Variables
```bash
# FastAPI Configuration
DEBUG=true
ENVIRONMENT=development
SECRET_KEY=your-secret-key-here

# Logfire Observability (Required)
LOGFIRE_TOKEN=pylf_v1_us_...
LOGFIRE_PROJECT_NAME={project-name}
LOGFIRE_SERVICE_NAME={project-name}-api
LOGFIRE_ENVIRONMENT=development

# Database Configuration
SURREALDB_URL=ws://localhost:8000/rpc
SURREALDB_USERNAME=root
SURREALDB_PASSWORD=root
SURREALDB_NAMESPACE={project}
SURREALDB_DATABASE=main

# TaskMaster AI Integration
ANTHROPIC_API_KEY=sk-ant-...
MODEL=claude-3-7-sonnet-20250219
MAX_TOKENS=64000
TEMPERATURE=0.2

# MCP Server Configuration
MCP_SERVER_NAME={project}-mcp-server
MCP_SERVER_VERSION=1.0.0
MCP_BIND_ADDRESS=127.0.0.1:8000

# DevQ.ai Development Environment
DEVQAI_ROOT=/Users/dionedge/devqai
PYTHONPATH=/Users/dionedge/devqai:$PYTHONPATH
MCP_SERVERS_PATH=/Users/dionedge/devqai/mcp/mcp-servers
```

## Security & Best Practices

### Authentication & Authorization
- OAuth2 with JWT tokens for API authentication
- Role-based access control (Admin, User, Viewer minimum)
- MCP session-based authentication for tool access
- Rate limiting on all public endpoints

### Data Protection
- AES-256 encryption at rest, TLS 1.3 in transit
- Pydantic model validation for all API inputs
- SQLAlchemy ORM for SQL injection protection
- Comprehensive audit logging with Logfire

### Development Security
- Never commit sensitive data to version control
- Use environment variables for all credentials
- Implement proper error handling without exposing internals
- Regular dependency security audits

## Current Integration Status

✅ **Five-Component Stack**: FastAPI + Logfire + PyTest + TaskMaster AI + MCP  
✅ **Configuration Alignment**: All required config files standardized  
✅ **MCP Servers**: 20+ servers configured and operational  
✅ **Development Environment**: Zed + Claude integration optimized  
✅ **Team Workflow**: Git, testing, and deployment patterns established  
✅ **Knowledge Integration**: Ptolemies knowledge base with framework documentation  

## System Relationships

### Zed Editor Configuration
- **Purpose**: IDE-specific configuration for development workflow
- **Features**: Terminal, formatting, LSP, themes, MCP integration
- **Integration**: Direct MCP server configuration for Zed's AI features

### Claude Code Configuration  
- **Purpose**: AI assistant tool access and capabilities
- **Features**: MCP server registry, permission system, tool orchestration
- **Integration**: Centralized server definitions and security controls

### Shared Resources
- **Environment Variables**: Both systems reference unified `.env` configuration
- **Project Paths**: Common project structure and file organization
- **MCP Ecosystem**: Complementary tool access for enhanced productivity

The Zed configuration enhances the development environment while Claude configuration enables AI-powered assistance tools. Together, they provide comprehensive support for the DevQ.ai development ecosystem with seamless integration between human developers and AI assistants.

---

*DevQ.ai Development Environment - Engineered for AI-Assisted Development Excellence*  
*Copyright © 2025 DevQ.ai - All Rights Reserved*