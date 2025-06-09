# Claude Configuration

## Project Information
- **Organization**: DevQ.ai
- **Project Structure**: Multi-repository monorepo
- **Rules Reference**: [./rules](./rules) directory

## Project Standards

### Code Formatting
- Backend: 88 character line length, Black formatter
- Frontend: 100 character line length, single quotes, required semicolons
- Python: 3.12, Black formatter, Google-style docstrings
- TypeScript: Strict mode, ES2022 target

### Documentation Guidelines
- All public APIs must have complete documentation
- Follow Google-style docstrings for Python
- Document React components with props and state descriptions
- Always include code examples for non-trivial functions

### Project-Specific Rules
- Agent components must implement error handling for all tool usage
- Frontend components must support both light and dark themes
- API endpoints must include proper validation and error responses
- Database operations should use connection pooling and transaction management

### Database Systems
- **SurrealDB**: Use for knowledge bases with vector search capabilities
- **PostgreSQL**: Default choice for relational data
- **Redis**: For caching and pub/sub messaging
- **Neo4j**: For RAG, Knowledge Graph

### MCP Integration
- MCP servers follow [./mcp/mcp-servers.json](./mcp/mcp-servers.json) specification
- Create Python clients with pydantic-ai tools
- Expose vector search capabilities for knowledge bases
- Available MCP servers:
  - mcp-server-context7: For contextual reasoning with 7-hop depth
  - mcp-server-sequential-thinking: For step-by-step problem solving
  - mcp-memory: For persistent memory across sessions
  - dart-mcp-server: For Dart AI integration and smart code assistance

### Knowledge Base Setup
- Ptolemies knowledge base uses SurrealDB
- Run `./setup-database.sh` in the ptolemies directory to initialize
- Verify with `./verify-database.py`
- CLI available at `./cli.py`
- MCP server: `python -m ptolemies.mcp.ptolemies_mcp`

### Default Review Checklist
- [ ] Follows code style guidelines for the relevant language
- [ ] Includes appropriate documentation
- [ ] Has sufficient test coverage
- [ ] Handles errors gracefully
- [ ] Maintains backward compatibility
- [ ] Follows security best practices

### Claude Code MCP Servers

#### Available MCP Tools for Claude Code

```yaml
# Core Development Tools (NPX-based)
filesystem:
  command: npx
  args: ["-y", "@modelcontextprotocol/server-filesystem", "."]

git:
  command: npx  
  args: ["-y", "@modelcontextprotocol/server-git"]

fetch:
  command: npx
  args: ["-y", "@modelcontextprotocol/server-fetch"]

memory:
  command: npx
  args: ["-y", "@modelcontextprotocol/server-memory"]

sequentialthinking:
  command: npx
  args: ["-y", "@modelcontextprotocol/server-sequentialthinking"]

# Local MCP Servers (your existing setup)
context7:
  command: python
  args: ["-m", "context7_mcp.server"]
  cwd: "./devqai/mcp/mcp-servers/context7-mcp"
  env:
    UPSTASH_REDIS_REST_URL: "${UPSTASH_REDIS_REST_URL}"
    UPSTASH_REDIS_REST_TOKEN: "${UPSTASH_REDIS_REST_TOKEN}"

crawl4ai:
  command: python
  args: ["-m", "crawl4ai_mcp.server"]
  cwd: "./devqai/mcp/mcp-servers/crawl4ai-mcp"

ptolemies:
  command: python
  args: ["-m", "ptolemies.mcp.ptolemies_mcp"]
  cwd: "./devqai/ptolemies"

bayes:
  command: python
  args: ["-m", "bayes_mcp"]
  cwd: "./devqai/bayes"

memory:
  command: npx
  args: ["-y", "@modelcontextprotocol/server-memory"]

# Additional MCP Servers
dart:
  command: npx
  args: ["-y", "dart-mcp-server"]
  env:
    DART_TOKEN: "${DART_TOKEN}"

surrealdb:
  command: python
  args: ["-m", "surrealdb_mcp.server"]
  cwd: "./devqai/mcp/mcp-servers/surrealdb-mcp"
  env:
    SURREALDB_URL: "${SURREALDB_URL}"
    SURREALDB_USERNAME: "${SURREALDB_USERNAME}"
    SURREALDB_PASSWORD: "${SURREALDB_PASSWORD}"

github:
  command: npx
  args: ["-y", "@modelcontextprotocol/server-github"]
  env:
    GITHUB_PERSONAL_ACCESS_TOKEN: "${GITHUB_PERSONAL_ACCESS_TOKEN}"

calendar:
  command: python
  args: ["-m", "calendar_mcp.server"]
  cwd: "./devqai/mcp/mcp-servers/calendar-mcp"
  env:
    GOOGLE_CALENDAR_CREDENTIALS: "${GOOGLE_CALENDAR_CREDENTIALS}"

# Development & UI Tools
shadcn-ui:
  command: python
  args: ["-m", "shadcn_ui_mcp.server"]
  cwd: "./devqai/mcp/mcp-servers/shadcn-ui-mcp-server"

magic:
  command: python
  args: ["-m", "magic_mcp.server"]
  cwd: "./devqai/mcp/mcp-servers/magic-mcp"

# Financial & Data Tools
stripe:
  command: python
  args: ["-m", "stripe_mcp.server"]
  cwd: "./devqai/mcp/mcp-servers/stripe-mcp"
  env:
    STRIPE_API_KEY: "${STRIPE_API_KEY}"

jupyter:
  command: python
  args: ["-m", "jupyter_mcp.server"]
  cwd: "./devqai/mcp/mcp-servers/jupyter-mcp"

# Constraint Solvers
solver-mzn:
  command: python
  args: ["-m", "solver_mzn_mcp.server"]
  cwd: "./devqai/mcp/mcp-servers/solver-mzn-mcp"

solver-pysat:
  command: python
  args: ["-m", "solver_pysat_mcp.server"]
  cwd: "./devqai/mcp/mcp-servers/solver-pysat-mcp"

solver-z3:
  command: python
  args: ["-m", "solver_z3_mcp.server"]
  cwd: "./devqai/mcp/mcp-servers/solver-z3-mcp"

# Registry & Management
registry:
  command: python
  args: ["-m", "registry_mcp.server"]
  cwd: "./devqai/mcp/mcp-servers/registry-mcp"

# Optional: Inspector for debugging
inspector:
  command: npx
  args: ["-y", "@modelcontextprotocol/inspector"]
```

#### MCP Server Configuration Notes
- **filesystem**: File read/write operations for the current project directory
- **git**: Version control operations, commits, and branch management  
- **fetch**: API calls and external resource access
- **memory**: Persistent memory across Claude Code sessions
- **sequentialthinking**: Enhanced step-by-step problem solving
- **context7**: Advanced contextual reasoning with Redis-backed memory (your local server)
- **crawl4ai**: Web scraping and content extraction (your local server)
- **ptolemies**: Your custom knowledge base integration
- **bayes**: Bayesian analysis and statistical modeling MCP server
- **dart**: Dart AI integration for smart code assistance and development
- **surrealdb**: SurrealDB multi-model database operations with graph capabilities
- **github**: GitHub API integration for repository management, issues, and pull requests
- **calendar**: Google Calendar integration for event management and scheduling
- **shadcn-ui**: shadcn/ui component library integration for React development
- **magic**: AI-powered code generation and transformation utilities
- **stripe**: Stripe payment processing integration with transaction management
- **jupyter**: Jupyter notebook execution and data science workflow management
- **solver-mzn**: MiniZinc constraint satisfaction and optimization solver
- **solver-pysat**: PySAT Boolean satisfiability problem solver with advanced algorithms
- **solver-z3**: Z3 theorem prover for formal verification and constraint solving
- **registry**: Official MCP server registry with discovery and installation tools
- **inspector**: Debug MCP server connections (optional)

#### Prerequisites
- Node.js installed for NPX-based servers
- Your existing Python environment with MCP servers already set up
- SurrealDB running locally for Ptolemies knowledge base
- Redis credentials configured for Context7 (in your .env file)

#### Environment Variables Required
Add these to your `.env` file (if not already present):
```bash
# Context7 Redis Configuration
UPSTASH_REDIS_REST_URL=your_redis_url
UPSTASH_REDIS_REST_TOKEN=your_redis_token

# SurrealDB Configuration (for Ptolemies)
SURREALDB_URL=ws://localhost:8000/rpc
SURREALDB_USERNAME=root
SURREALDB_PASSWORD=root
SURREALDB_NAMESPACE=ptolemies
SURREALDB_DATABASE=knowledge

# Dart AI Configuration
DART_TOKEN=your_dart_ai_token

# GitHub Configuration
GITHUB_PERSONAL_ACCESS_TOKEN=your_github_token

# Google Calendar Configuration
GOOGLE_CALENDAR_CREDENTIALS=path_to_credentials_json

# Stripe Configuration
STRIPE_API_KEY=your_stripe_api_key
```

#### Usage with Claude Code
These MCP tools are automatically available when Claude Code runs in this directory. Claude Code reads this CLAUDE.md file and enables the specified tools using your existing local MCP server implementations.