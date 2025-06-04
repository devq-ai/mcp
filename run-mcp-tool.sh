#!/bin/bash
# Run an MCP tool with proper environment setup

# Activate the virtual environment
source "/Users/dionedge/devqai/mcp/pydantic_ai_env/bin/activate"

# Load environment variables
if [ -f "/Users/dionedge/devqai/mcp/pydantic_ai_env/mcp.env" ]; then
  set -o allexport
  source "/Users/dionedge/devqai/mcp/pydantic_ai_env/mcp.env"
  set +o allexport
fi

# Tool handler function
run_tool() {
  case $1 in
    "testmodel")
      echo "Running TestModel demo..."
      python "/Users/dionedge/devqai/mcp/claude4/examples/testmodel_demo.py"
      ;;
    "bayes")
      echo "Starting Bayes MCP server..."
      cd "/Users/dionedge/devqai/bayes"
      python bayes_mcp.py
      ;;
        "agentql-mcp")
      echo "Would start agentql-mcp server if installed..."
      echo "Run mcp-servers/agentql-mcp/install.sh first to install this server."
      ;;
    "browser-tools-mcp")
      echo "Would start browser-tools-mcp server if installed..."
      echo "Run mcp-servers/browser-tools-mcp/install.sh first to install this server."
      ;;
    "calendar-mcp")
      echo "Would start calendar-mcp server if installed..."
      echo "Run mcp-servers/calendar-mcp/install.sh first to install this server."
      ;;
    "context7-mcp")
      echo "Would start context7-mcp server if installed..."
      echo "Run mcp-servers/context7-mcp/install.sh first to install this server."
      ;;
    "crawl4ai-mcp")
      echo "Would start crawl4ai-mcp server if installed..."
      echo "Run mcp-servers/crawl4ai-mcp/install.sh first to install this server."
      ;;
    "dart-mcp")
      echo "Would start dart-mcp server if installed..."
      echo "Run mcp-servers/dart-mcp/install.sh first to install this server."
      ;;
    "github-mcp")
      echo "Would start github-mcp server if installed..."
      echo "Run mcp-servers/github-mcp/install.sh first to install this server."
      ;;
    "jupyter-mcp")
      echo "Would start jupyter-mcp server if installed..."
      echo "Run mcp-servers/jupyter-mcp/install.sh first to install this server."
      ;;
    "magic-mcp")
      echo "Would start magic-mcp server if installed..."
      echo "Run mcp-servers/magic-mcp/install.sh first to install this server."
      ;;
    "memory-mcp")
      echo "Would start memory-mcp server if installed..."
      echo "Run mcp-servers/memory-mcp/install.sh first to install this server."
      ;;
    "registry-mcp")
      echo "Would start registry-mcp server if installed..."
      echo "Run mcp-servers/registry-mcp/install.sh first to install this server."
      ;;
    "shadcn-ui-mcp-server")
      echo "Would start shadcn-ui-mcp-server server if installed..."
      echo "Run mcp-servers/shadcn-ui-mcp-server/install.sh first to install this server."
      ;;
    "solver-mzn-mcp")
      echo "Would start solver-mzn-mcp server if installed..."
      echo "Run mcp-servers/solver-mzn-mcp/install.sh first to install this server."
      ;;
    "solver-pysat-mcp")
      echo "Would start solver-pysat-mcp server if installed..."
      echo "Run mcp-servers/solver-pysat-mcp/install.sh first to install this server."
      ;;
    "solver-z3-mcp")
      echo "Would start solver-z3-mcp server if installed..."
      echo "Run mcp-servers/solver-z3-mcp/install.sh first to install this server."
      ;;
    "stripe-mcp")
      echo "Would start stripe-mcp server if installed..."
      echo "Run mcp-servers/stripe-mcp/install.sh first to install this server."
      ;;
    "surrealdb-mcp")
      echo "Would start surrealdb-mcp server if installed..."
      echo "Run mcp-servers/surrealdb-mcp/install.sh first to install this server."
      ;;
    "list")
      echo "Available MCP tools:"
      echo "  testmodel - Run the TestModel demo"
      echo "  bayes     - Start the Bayes MCP server"
        echo "  agentql-mcp - Web scraping and browser automation using natural language queries"
  echo "  browser-tools-mcp - Complete browser automation toolkit with screenshot and interaction capabilities"
  echo "  calendar-mcp - Google Calendar integration for event management and scheduling"
  echo "  context7-mcp - Advanced context management and semantic search with vector embeddings"
  echo "  crawl4ai-mcp - Intelligent web crawling with AI-powered content extraction and analysis"
  echo "  dart-mcp - Dart/Flutter development tools with package management and testing"
  echo "  github-mcp - GitHub API integration for repository management, issues, and pull requests"
  echo "  jupyter-mcp - Jupyter notebook execution and data science workflow management"
  echo "  magic-mcp - AI-powered code generation and transformation utilities"
  echo "  memory-mcp - Persistent memory management with contextual recall and learning"
  echo "  registry-mcp - Official MCP server registry with discovery and installation tools"
  echo "  shadcn-ui-mcp-server - shadcn/ui component library integration for React development"
  echo "  solver-mzn-mcp - MiniZinc constraint satisfaction and optimization solver"
  echo "  solver-pysat-mcp - PySAT Boolean satisfiability problem solver with advanced algorithms"
  echo "  solver-z3-mcp - Z3 theorem prover for formal verification and constraint solving"
  echo "  stripe-mcp - Stripe payment processing integration with transaction management"
  echo "  surrealdb-mcp - SurrealDB multi-model database integration with graph capabilities"
      echo "  help      - Show this help message"
      ;;
    *)
      echo "Usage: $0 [tool]"
      echo ""
      echo "Available tools:"
      echo "  testmodel - Run the TestModel demo"
      echo "  bayes     - Start the Bayes MCP server"
        echo "  agentql-mcp - Web scraping and browser automation using natural language queries"
  echo "  browser-tools-mcp - Complete browser automation toolkit with screenshot and interaction capabilities"
  echo "  calendar-mcp - Google Calendar integration for event management and scheduling"
  echo "  context7-mcp - Advanced context management and semantic search with vector embeddings"
  echo "  crawl4ai-mcp - Intelligent web crawling with AI-powered content extraction and analysis"
  echo "  dart-mcp - Dart/Flutter development tools with package management and testing"
  echo "  github-mcp - GitHub API integration for repository management, issues, and pull requests"
  echo "  jupyter-mcp - Jupyter notebook execution and data science workflow management"
  echo "  magic-mcp - AI-powered code generation and transformation utilities"
  echo "  memory-mcp - Persistent memory management with contextual recall and learning"
  echo "  registry-mcp - Official MCP server registry with discovery and installation tools"
  echo "  shadcn-ui-mcp-server - shadcn/ui component library integration for React development"
  echo "  solver-mzn-mcp - MiniZinc constraint satisfaction and optimization solver"
  echo "  solver-pysat-mcp - PySAT Boolean satisfiability problem solver with advanced algorithms"
  echo "  solver-z3-mcp - Z3 theorem prover for formal verification and constraint solving"
  echo "  stripe-mcp - Stripe payment processing integration with transaction management"
  echo "  surrealdb-mcp - SurrealDB multi-model database integration with graph capabilities"
      echo "  list      - List all available tools"
      ;;
  esac
}

# Run the tool
run_tool "$1"
