# Agentical MCP Registry & Agent Framework

## 📋 Table of Contents

- [Tools Registry](#tools-registry)
  - [Anthropic Core Tools](#anthropic-core-tools) 8
  - [Core Agent Tools](#core-agent-tools) 14
  - [MCP Servers](#mcp-servers) 18
  - [Pydantic AI Tools](#pydantic-ai-tools) 11
  - [Zed MCP Tools](#zed-mcp-tools) 9
  - [Optional Tools](#optional-tools) 17

## 🛠 Tools Registry

### Anthropic Core Tools (count=8)

Built-in tools available in Claude 4 Sonnet for enhanced AI capabilities.

| Tool | Description | Enabled | Reference |
|------|-------------|---------|-----------|
| **audit_logging** | Comprehensive logging and audit trail system for AI interactions and tool usage | ✅ `true` | [Anthropic Cookbook - Tool Use](https://github.com/anthropics/anthropic-cookbook/blob/main/tool_use/) |
| **code_execution** | Native Python code execution in secure sandboxed environment with persistent memory | ✅ `true` | [Anthropic Cookbook - Code Execution](https://github.com/anthropics/anthropic-cookbook/blob/main/tool_use/calculator_tool.ipynb) |
| **extended_thinking** | Deep reasoning mode with enhanced problem-solving and multi-step analysis | ✅ `true` | [Anthropic Cookbook - Extended Thinking](https://github.com/anthropics/anthropic-cookbook/blob/main/extended_thinking/) |
| **file_access** | Local file system operations including read, write, and directory navigation | ✅ `true` | [Anthropic Cookbook - File Operations](https://github.com/anthropics/anthropic-cookbook/) |
| **memory** | Persistent context and memory management across conversation sessions | ✅ `true` | [Anthropic Cookbook - Memory](https://github.com/anthropics/anthropic-cookbook/) |
| **parallel_tool** | Concurrent execution of multiple tools simultaneously for enhanced efficiency | ✅ `true` | [Anthropic Cookbook - Parallel Tools](https://github.com/anthropics/anthropic-cookbook/blob/main/tool_use/parallel_tools_claude_3_7_sonnet.ipynb) |
| **usage_monitoring** | Real-time monitoring of API usage, token consumption, and performance metrics | ✅ `true` | [Anthropic Cookbook - Monitoring](https://github.com/anthropics/anthropic-cookbook/) |
| **web_search** | Real-time web search capabilities during extended thinking mode | ✅ `true` | [Anthropic Cookbook - Web Search](https://github.com/anthropics/anthropic-cookbook/blob/main/tool_use/tool_choice.ipynb) |

### Core Agent Tools (count=14)

Essential tools for agent-based operations and enterprise workflows.

| Tool | Description | Enabled | Reference |
|------|-------------|---------|-----------|
| **batch_process** | Process multiple operations in batches with error handling and retry logic | ✅ `true` | [Pydantic AI - Batch Processing](https://github.com/pydantic/pydantic-ai/blob/main/docs/tools.md) |
| **create_report** | Generate comprehensive reports with charts, tables, and formatted output | ✅ `true` | [Pydantic AI - Reporting](https://github.com/pydantic/pydantic-ai/blob/main/docs/tools.md) |
| **doc_gen** | Automated documentation generation with markdown, HTML, and PDF support | ✅ `true` | [Pydantic AI - Documentation](https://github.com/pydantic/pydantic-ai/blob/main/docs/tools.md) |
| **expensive_calc** | High-performance computational operations with resource optimization | ✅ `true` | [Pydantic AI - Calculations](https://github.com/pydantic/pydantic-ai/blob/main/docs/tools.md) |
| **external_api** | Secure integration with external APIs including authentication and rate limiting | ✅ `true` | [Pydantic AI - External APIs](https://github.com/pydantic/pydantic-ai/blob/main/docs/tools.md) |
| **generate_chart** | Data visualization tool supporting multiple chart types and export formats | ✅ `true` | [Pydantic AI - Visualization](https://github.com/pydantic/pydantic-ai/blob/main/docs/tools.md) |
| **test_gen** | Automated test case generation for code validation and quality assurance | ✅ `true` | [Pydantic AI - Testing](https://github.com/pydantic/pydantic-ai/blob/main/docs/testing.md) |
| **output** | Structured output formatting with validation and schema enforcement | ✅ `true` | [Pydantic AI - Output Types](https://github.com/pydantic/pydantic-ai/blob/main/docs/output.md) |
| **plan_gen** | Strategic planning and roadmap generation with milestone tracking | ✅ `true` | [Pydantic AI - Planning](https://github.com/pydantic/pydantic-ai/blob/main/docs/tools.md) |
| **plan_run** | Execute generated plans with progress monitoring and adaptive scheduling | ✅ `true` | [Pydantic AI - Execution](https://github.com/pydantic/pydantic-ai/blob/main/docs/tools.md) |
| **process_data** | ETL operations with data transformation, validation, and quality checks | ✅ `true` | [Pydantic AI - Data Processing](https://github.com/pydantic/pydantic-ai/blob/main/docs/tools.md) |
| **test_run** | Execute test suites with coverage reporting and result analysis | ✅ `true` | [Pydantic AI - Test Execution](https://github.com/pydantic/pydantic-ai/blob/main/docs/testing.md) |
| **viz_playbook** | Interactive visualization playbooks for data exploration and analysis | ✅ `true` | [Pydantic AI - Visualization](https://github.com/pydantic/pydantic-ai/blob/main/docs/tools.md) |
| **web_search** | Enhanced web search with result filtering, ranking, and content extraction | ✅ `true` | [Pydantic AI - Common Tools](https://github.com/pydantic/pydantic-ai/blob/main/docs/common-tools.md) |

### MCP Servers (count=18)

Production-ready Model Context Protocol servers for specialized capabilities.

| Tool | Repository | Description | Enabled | Reference |
|------|------------|-------------|---------|-----------|
| **agentql-mcp** | [tinyfish-io/agentql-mcp](https://github.com/tinyfish-io/agentql-mcp/) | Web scraping and browser automation using natural language queries | ✅ `true` | [MCP Docs - Browser Tools](https://github.com/modelcontextprotocol/docs/blob/main/docs/concepts/tools.mdx) |
| **bayes-mcp** | [devq-ai/bayes](https://github.com/devq-ai/bayes) | Bayesian inference and statistical analysis with MCMC sampling capabilities | ✅ `true` | [Bayes MCP Documentation](https://github.com/devq-ai/bayes/blob/main/README.md) |
| **browser-tools** | [AgentDeskAI/browser-tools-mcp](https://github.com/AgentDeskAI/browser-tools-mcp/) | Complete browser automation toolkit with screenshot and interaction capabilities | ✅ `true` | [MCP Docs - Browser Integration](https://github.com/modelcontextprotocol/docs/blob/main/docs/concepts/tools.mdx) |
| **calendar-mcp** | [Zawad99/Google-Calendar-MCP-Server](https://github.com/Zawad99/Google-Calendar-MCP-Server.git) | Google Calendar integration for event management and scheduling | ✅ `true` | [MCP Docs - Calendar Integration](https://github.com/modelcontextprotocol/docs/blob/main/docs/concepts/tools.mdx) |
| **context7-mcp** | [upstash/context7](https://github.com/upstash/context7) | Advanced context management and semantic search with vector embeddings | ✅ `true` | [Context7 Documentation](https://github.com/upstash/context7/blob/main/README.md) |
| **crawl4ai-mcp** | [wyattowalsh/crawl4ai-mcp](https://github.com/wyattowalsh/crawl4ai-mcp) | Intelligent web crawling with AI-powered content extraction and analysis | ✅ `true` | [MCP Docs - Web Crawling](https://github.com/modelcontextprotocol/docs/blob/main/docs/concepts/tools.mdx) |
| **dart-mcp** | [its-dart/dart-mcp-server](https://github.com/its-dart/dart-mcp-server/) | Dart/Flutter development tools with package management and testing | ✅ `true` | [MCP Docs - Development Tools](https://github.com/modelcontextprotocol/docs/blob/main/docs/concepts/tools.mdx) |
| **github-mcp** | [github/github-mcp-server](https://github.com/github/github-mcp-server) | GitHub API integration for repository management, issues, and pull requests | ✅ `true` | [GitHub MCP Documentation](https://github.com/github/github-mcp-server/blob/main/README.md) |
| **jupyter-mcp** | [datalayer/jupyter-mcp-server](https://github.com/datalayer/jupyter-mcp-server.git) | Jupyter notebook execution and data science workflow management | ✅ `true` | [MCP Docs - Jupyter Integration](https://github.com/modelcontextprotocol/docs/blob/main/docs/concepts/tools.mdx) |
| **magic-mcp** | [21st-dev/magic-mcp](https://github.com/21st-dev/magic-mcp/) | AI-powered code generation and transformation utilities | ✅ `true` | [MCP Docs - Code Magic](https://github.com/modelcontextprotocol/docs/blob/main/docs/concepts/tools.mdx) |
| **memory-mcp** | [mem0ai/mem0-mcp](https://github.com/mem0ai/mem0-mcp) | Persistent memory management with contextual recall and learning | ✅ `true` | [Mem0 MCP Documentation](https://github.com/mem0ai/mem0-mcp/blob/main/README.md) |
| **registry-mcp** | [modelcontextprotocol/registry](https://github.com/modelcontextprotocol/registry/) | Official MCP server registry with discovery and installation tools | ✅ `true` | [MCP Registry Documentation](https://github.com/modelcontextprotocol/registry/blob/main/README.md) |
| **shadcn-ui-mcp-server** | [ymadd/shadcn-ui-mcp-server](https://github.com/ymadd/shadcn-ui-mcp-server/) | shadcn/ui component library integration for React development | ✅ `true` | [MCP Docs - UI Components](https://github.com/modelcontextprotocol/docs/blob/main/docs/concepts/tools.mdx) |
| **solver-mzn-mcp** | [szeider/mcp-solver](https://github.com/szeider/mcp-solver) | MiniZinc constraint satisfaction and optimization solver | ✅ `true` | [MCP Solver Documentation](https://github.com/szeider/mcp-solver/blob/main/README.md) |
| **solver-pysat-mcp** | [szeider/mcp-solver](https://github.com/szeider/mcp-solver) | PySAT Boolean satisfiability problem solver with advanced algorithms | ✅ `true` | [MCP Solver Documentation](https://github.com/szeider/mcp-solver/blob/main/README.md) |
| **solver-z3-mcp** | [szeider/mcp-solver](https://github.com/szeider/mcp-solver) | Z3 theorem prover for formal verification and constraint solving | ✅ `true` | [MCP Solver Documentation](https://github.com/szeider/mcp-solver/blob/main/README.md) |
| **stripe-mcp** | [stripe/agent-toolkit](https://github.com/stripe/agent-toolkit) | Stripe payment processing integration with transaction management | ✅ `true` | [Stripe Agent Toolkit](https://github.com/stripe/agent-toolkit/blob/main/README.md) |
| **surrealdb-mcp** | [nsxdavid/surrealdb-mcp-server](https://github.com/nsxdavid/surrealdb-mcp-server.git) | SurrealDB multi-model database integration with graph capabilities | ✅ `true` | [MCP Docs - Database Integration](https://github.com/modelcontextprotocol/docs/blob/main/docs/concepts/tools.mdx) |

### Pydantic AI Tools (count=11)

Type-safe AI tools with Pydantic validation and structured output.

| Tool | Description | Enabled | Reference |
|------|-------------|---------|-----------|
| **calculate_math** | Advanced mathematical computations with symbolic math and numerical analysis | ✅ `true` | [Pydantic AI - Mathematical Tools](https://github.com/pydantic/pydantic-ai/blob/main/docs/tools.md) |
| **evals** | Comprehensive evaluation framework for model performance and accuracy testing | ✅ `true` | [Pydantic AI - Evaluation](https://github.com/pydantic/pydantic-ai/blob/main/docs/testing.md) |
| **execute_query** | Database query execution with connection pooling and result validation | ✅ `true` | [Pydantic AI - Database Operations](https://github.com/pydantic/pydantic-ai/blob/main/docs/tools.md) |
| **format_text** | Advanced text formatting with markdown, HTML, and custom template support | ✅ `true` | [Pydantic AI - Text Processing](https://github.com/pydantic/pydantic-ai/blob/main/docs/tools.md) |
| **get_timestamp** | Timezone-aware timestamp generation with multiple format options | ✅ `true` | [Pydantic AI - Time Utilities](https://github.com/pydantic/pydantic-ai/blob/main/docs/tools.md) |
| **get_user_data** | Secure user data retrieval with privacy compliance and access controls | ✅ `true` | [Pydantic AI - User Management](https://github.com/pydantic/pydantic-ai/blob/main/docs/tools.md) |
| **logfire-mcp** | Integrated observability and logging with structured monitoring | ✅ `true` | [Pydantic Logfire](https://github.com/pydantic/logfire/blob/main/README.md) |
| **mcp-run-python** | Secure Python code execution in isolated environments with dependency management | ✅ `true` | [Pydantic AI - Python Runner](https://github.com/pydantic/pydantic-ai/blob/main/docs/mcp/client.md) |
| **message_chat_history** | Conversation history management with search and context preservation | ✅ `true` | [Pydantic AI - Chat History](https://github.com/pydantic/pydantic-ai/blob/main/docs/tools.md) |
| **multimodal_input** | Process text, images, audio, and video inputs with unified interface | ✅ `true` | [Pydantic AI - Multimodal](https://github.com/pydantic/pydantic-ai/blob/main/docs/tools.md) |
| **unit_test** | Automated unit test generation and execution with coverage analysis | ✅ `true` | [Pydantic AI - Unit Testing](https://github.com/pydantic/pydantic-ai/blob/main/docs/testing.md) |

### Zed MCP Tools (count=9)

Zed editor integration tools for seamless development workflows.

| Tool | Description | Enabled | Reference |
|------|-------------|---------|-----------|
| **browser-tools-context-server** | Browser automation within Zed editor with context-aware interactions | ✅ `true` | [Zed MCP Documentation](https://zed.dev/docs/ai/mcp) |
| **mcp-server-buildkite** | Buildkite CI/CD integration for pipeline management and build monitoring | ✅ `true` | [Zed MCP Integration](https://zed.dev/blog/mcp) |
| **mcp-server-context7** | Context7 semantic search integration for enhanced code understanding | ✅ `true` | [Zed MCP Features](https://zed.dev/docs/ai/mcp) |
| **mcp-server-exa-search** | Exa search engine integration for developer-focused web search | ✅ `true` | [Zed MCP Tools](https://zed.dev/docs/ai/mcp) |
| **mcp-server-github** | GitHub integration with repository browsing and code management | ✅ `true` | [Zed GitHub Integration](https://zed.dev/docs/ai/mcp) |
| **mcp-server-grafana** | Grafana monitoring integration with dashboard access and alerting | ✅ `true` | [Zed Observability Tools](https://zed.dev/docs/ai/mcp) |
| **mcp-server-sequential-thinking** | Sequential reasoning capabilities for complex problem-solving workflows | ✅ `true` | [Zed AI Features](https://zed.dev/docs/ai/overview) |
| **mcp-server-slack** | Slack integration for team communication and notification management | ✅ `true` | [Zed Collaboration Tools](https://zed.dev/docs/ai/mcp) |
| **zed-slack-mcp** | Enhanced Slack integration with message threading and file sharing | ✅ `true` | [Zed Slack MCP](https://zed.dev/docs/ai/mcp) |

### Optional Tools (count=17)

| Tool | Repository | Description | Enabled |
|------|------------|-------------|---------|
| **bigquery-mcp** | [LucasHild/mcp-server-bigquery](https://github.com/LucasHild/mcp-server-bigquery) | Google BigQuery integration for large-scale data analytics | ❌ `false` |
| **databricks-mcp** | [JordiNeil/mcp-databricks-server](https://github.com/JordiNeil/mcp-databricks-server) | Databricks platform integration for big data processing | ❌ `false` |
| **esignatures-mcp** | [esignaturescom/mcp-server-esignatures](https://github.com/esignaturescom/mcp-server-esignatures) | Electronic signature workflow management | ❌ `false` |
| **financial-mcp** | [financial-datasets/mcp-server](https://github.com/financial-datasets/mcp-server) | Financial data analysis and market research tools | ❌ `false` |
| **gcp-mcp** | [eniayomi/gcp-mcp](https://github.com/eniayomi/gcp-mcp) | Google Cloud Platform integration and resource management | ❌ `false` |
| **gmail-mcp** | [ykuchiki/gmail-mcp](https://github.com/ykuchiki/gmail-mcp.git) | Gmail integration for email automation and management | ❌ `false` |
| **markdownify-mcp** | [zcaceres/markdownify-mcp](https://github.com/zcaceres/markdownify-mcp/) | HTML to Markdown conversion with formatting preservation | ❌ `false` |
| **paypal-mcp** | [PayPal Developer Tools](https://developer.paypal.com/tools/mcp-server/) | PayPal payment processing and transaction management | ❌ `false` |
| **puppeteer-mcp** | [@modelcontextprotocol/server-puppeteer](https://www.npmjs.com/package/@modelcontextprotocol/server-puppeteer) | Headless browser automation with Puppeteer | ❌ `false` |
| **redis-mcp** | [modelcontextprotocol/servers/redis](https://github.com/modelcontextprotocol/servers/tree/HEAD/src/redis/) | Redis cache and data store integration | ❌ `false` |
| **scholarly-mcp** | [adityak74/mcp-scholarly](https://github.com/adityak74/mcp-scholarly) | Academic research and scholarly article access | ❌ `false` |
| **slack-mcp** | [modelcontextprotocol/servers/slack](https://github.com/modelcontextprotocol/servers/tree/main/src/slack) | Basic Slack integration for messaging | ❌ `false` |
| **snowflake-mcp** | [isaacwasserman/mcp-snowflake-server](https://github.com/isaacwasserman/mcp-snowflake-server) | Snowflake data warehouse integration | ❌ `false` |
| **sqlite-mcp** | [modelcontextprotocol/servers/sqlite](https://github.com/modelcontextprotocol/servers/tree/HEAD/src/sqlite/) | SQLite database operations and queries | ❌ `false` |
| **typescript-mcp** | [modelcontextprotocol/typescript-sdk](https://github.com/modelcontextprotocol/typescript-sdk/tree/main) | TypeScript development tools and type checking | ❌ `false` |
| **wikidata-mcp** | [zzaebok/mcp-wikidata](https://github.com/zzaebok/mcp-wikidata) | Wikidata knowledge base integration and queries | ❌ `false` |
| **xero-mcp** | [XeroAPI/xero-mcp-server](https://github.com/XeroAPI/xero-mcp-server) | Xero accounting software integration | ❌ `false` |
