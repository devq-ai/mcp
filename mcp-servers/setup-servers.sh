#!/bin/bash
# Setup script for MCP servers

# Define color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}===== MCP Servers Setup =====${NC}"
echo ""

# Base directories
MCP_BASE="/Users/dionedge/devqai/mcp"
SERVERS_DIR="${MCP_BASE}/mcp-servers"

# Check if servers directory exists
if [ ! -d "$SERVERS_DIR" ]; then
  echo -e "${RED}Servers directory not found. Creating...${NC}"
  mkdir -p "$SERVERS_DIR"
fi

# Create server directories and placeholder files
declare -a SERVERS=(
  "agentql-mcp:tinyfish-io/agentql-mcp:Web scraping and browser automation using natural language queries"
  "bayes-mcp:devq-ai/bayes:Bayesian inference and statistical analysis with MCMC sampling capabilities"
  "browser-tools-mcp:AgentDeskAI/browser-tools-mcp:Complete browser automation toolkit with screenshot and interaction capabilities"
  "calendar-mcp:Zawad99/Google-Calendar-MCP-Server:Google Calendar integration for event management and scheduling"
  "context7-mcp:upstash/context7:Advanced context management and semantic search with vector embeddings"
  "crawl4ai-mcp:wyattowalsh/crawl4ai-mcp:Intelligent web crawling with AI-powered content extraction and analysis"
  "dart-mcp:its-dart/dart-mcp-server:Dart/Flutter development tools with package management and testing"
  "github-mcp:github/github-mcp-server:GitHub API integration for repository management, issues, and pull requests"
  "jupyter-mcp:datalayer/jupyter-mcp-server:Jupyter notebook execution and data science workflow management"
  "magic-mcp:21st-dev/magic-mcp:AI-powered code generation and transformation utilities"
  "memory-mcp:mem0ai/mem0-mcp:Persistent memory management with contextual recall and learning"
  "registry-mcp:modelcontextprotocol/registry:Official MCP server registry with discovery and installation tools"
  "shadcn-ui-mcp-server:ymadd/shadcn-ui-mcp-server:shadcn/ui component library integration for React development"
  "solver-mzn-mcp:szeider/mcp-solver:MiniZinc constraint satisfaction and optimization solver"
  "solver-pysat-mcp:szeider/mcp-solver:PySAT Boolean satisfiability problem solver with advanced algorithms"
  "solver-z3-mcp:szeider/mcp-solver:Z3 theorem prover for formal verification and constraint solving"
  "stripe-mcp:stripe/agent-toolkit:Stripe payment processing integration with transaction management"
  "surrealdb-mcp:nsxdavid/surrealdb-mcp-server:SurrealDB multi-model database integration with graph capabilities"
)

echo -e "${YELLOW}Setting up ${#SERVERS[@]} MCP servers...${NC}"
echo ""

for server in "${SERVERS[@]}"; do
  IFS=: read -r name repo description <<< "$server"
  
  # Skip bayes-mcp as it's already set up
  if [ "$name" == "bayes-mcp" ]; then
    echo -e "${GREEN}✅ $name: Already set up via symlink${NC}"
    continue
  fi
  
  echo -e "${BLUE}Setting up $name...${NC}"
  
  # Create server directory
  mkdir -p "${SERVERS_DIR}/${name}"
  
  # Create README with installation instructions
  cat > "${SERVERS_DIR}/${name}/README.md" << EOL
# ${name}

## Description
${description}

## Repository
[${repo}](https://github.com/${repo})

## Installation

This is a placeholder for the ${name} MCP server. To install the actual server:

\`\`\`bash
# Clone the repository
git clone https://github.com/${repo}.git

# Install dependencies (example - check actual repo for specific instructions)
cd $(basename ${repo})
pip install -e .
\`\`\`

## Usage

Please refer to the [official repository](https://github.com/${repo}) for usage instructions.

## Status

- [ ] Cloned
- [ ] Dependencies installed
- [ ] Configuration complete
- [ ] Tested
EOL

  # Create placeholder configuration
  cat > "${SERVERS_DIR}/${name}/config.json" << EOL
{
  "name": "${name}",
  "repository": "https://github.com/${repo}",
  "description": "${description}",
  "enabled": false,
  "local_installed": false,
  "configuration": {
    "host": "localhost",
    "port": 8080,
    "api_key": "your_api_key_here",
    "timeout_ms": 5000
  }
}
EOL

  # Create install script
  cat > "${SERVERS_DIR}/${name}/install.sh" << EOL
#!/bin/bash
# Installation script for ${name}

echo "Installing ${name}..."
echo "Repository: https://github.com/${repo}"
echo ""
echo "This is a placeholder script. To install the actual server:"
echo "1. Clone the repository: git clone https://github.com/${repo}.git"
echo "2. Follow the installation instructions in the README"
echo ""
echo "Once installed, update the config.json file with the correct configuration."
EOL
  
  chmod +x "${SERVERS_DIR}/${name}/install.sh"
  
  echo -e "${GREEN}✅ ${name} setup complete${NC}"
done

# Create special case for surrealdb-mcp for ptolemies integration
if [ -d "/Users/dionedge/devqai/ptolemies" ]; then
  echo -e "${BLUE}Setting up surrealdb-mcp with ptolemies integration...${NC}"
  echo "# SurrealDB + Ptolemies Integration" >> "${SERVERS_DIR}/surrealdb-mcp/README.md"
  echo "" >> "${SERVERS_DIR}/surrealdb-mcp/README.md"
  echo "This MCP server can be integrated with the Ptolemies knowledge base project:" >> "${SERVERS_DIR}/surrealdb-mcp/README.md"
  echo "" >> "${SERVERS_DIR}/surrealdb-mcp/README.md"
  echo "- [Ptolemies](/Users/dionedge/devqai/ptolemies/)" >> "${SERVERS_DIR}/surrealdb-mcp/README.md"
  echo -e "${GREEN}✅ surrealdb-mcp + ptolemies integration noted${NC}"
fi

# Update run-mcp-tool.sh to include all servers
echo -e "${BLUE}Updating run-mcp-tool.sh script...${NC}"

cat > "${MCP_BASE}/run-mcp-tool.sh" << EOL
#!/bin/bash
# Run an MCP tool with proper environment setup

# Activate the virtual environment
source "${MCP_BASE}/pydantic_ai_env/bin/activate"

# Load environment variables
if [ -f "${MCP_BASE}/pydantic_ai_env/mcp.env" ]; then
  set -o allexport
  source "${MCP_BASE}/pydantic_ai_env/mcp.env"
  set +o allexport
fi

# Tool handler function
run_tool() {
  case \$1 in
    "testmodel")
      echo "Running TestModel demo..."
      python "${MCP_BASE}/claude4/examples/testmodel_demo.py"
      ;;
    "bayes")
      echo "Starting Bayes MCP server..."
      cd "/Users/dionedge/devqai/bayes"
      python bayes_mcp.py
      ;;
    $(for server in "${SERVERS[@]}"; do
        IFS=: read -r name repo description <<< "$server"
        if [ "$name" != "bayes-mcp" ]; then
          echo "    \"$name\")
      echo \"Would start $name server if installed...\"
      echo \"Run mcp-servers/$name/install.sh first to install this server.\"
      ;;"
        fi
      done)
    "list")
      echo "Available MCP tools:"
      echo "  testmodel - Run the TestModel demo"
      echo "  bayes     - Start the Bayes MCP server"
      $(for server in "${SERVERS[@]}"; do
        IFS=: read -r name repo description <<< "$server"
        if [ "$name" != "bayes-mcp" ]; then
          echo "  echo \"  $name - $description\""
        fi
      done)
      echo "  help      - Show this help message"
      ;;
    *)
      echo "Usage: \$0 [tool]"
      echo ""
      echo "Available tools:"
      echo "  testmodel - Run the TestModel demo"
      echo "  bayes     - Start the Bayes MCP server"
      $(for server in "${SERVERS[@]}"; do
        IFS=: read -r name repo description <<< "$server"
        if [ "$name" != "bayes-mcp" ]; then
          echo "  echo \"  $name - $description\""
        fi
      done)
      echo "  list      - List all available tools"
      ;;
  esac
}

# Run the tool
run_tool "\$1"
EOL

chmod +x "${MCP_BASE}/run-mcp-tool.sh"
echo -e "${GREEN}✅ run-mcp-tool.sh updated${NC}"

# Create registry-update script to mark local servers
cat > "${SERVERS_DIR}/update-tools-registry.py" << EOL
#!/usr/bin/env python3
"""
Update the MCP tools registry to mark locally available servers.
"""

import os
import re
import json
from pathlib import Path

# Constants
TOOLS_MD_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tools.md")
SERVERS_DIR = os.path.abspath(os.path.dirname(__file__))

def get_local_servers():
    """Get list of locally available MCP servers."""
    local_servers = []
    for item in os.listdir(SERVERS_DIR):
        item_path = os.path.join(SERVERS_DIR, item)
        if os.path.isdir(item_path) and not item.startswith('.'):
            # Check if it's installed
            config_path = os.path.join(item_path, "config.json")
            if os.path.exists(config_path):
                try:
                    with open(config_path, 'r') as f:
                        config = json.load(f)
                        installed = config.get('local_installed', False)
                        if installed:
                            local_servers.append(item)
                        else:
                            # Check if it's a symlink (like bayes-mcp)
                            if os.path.islink(item_path):
                                local_servers.append(item)
                except:
                    pass
            elif os.path.islink(item_path):
                local_servers.append(item)
    return local_servers

def update_tools_registry():
    """Update the tools.md file to mark locally available servers."""
    if not os.path.exists(TOOLS_MD_PATH):
        print(f"Error: tools.md not found at {TOOLS_MD_PATH}")
        return
    
    # Get locally available servers
    local_servers = get_local_servers()
    print(f"Found {len(local_servers)} locally available servers: {', '.join(local_servers)}")
    
    # Read the current tools.md
    with open(TOOLS_MD_PATH, 'r') as f:
        content = f.read()
    
    # Find the MCP Servers section
    mcp_servers_section = re.search(r'### MCP Servers \(count=\d+\)(.*?)(?=### [A-Za-z]|\Z)', content, re.DOTALL)
    if not mcp_servers_section:
        print("Error: MCP Servers section not found in tools.md")
        return
    
    section = mcp_servers_section.group(1)
    
    # Process each server row
    for server in local_servers:
        # Normalize server name for matching
        server_match = server.replace('-', '[_\\-]')
        # Look for the server row
        server_pattern = rf'\| \*\*{server_match}\*\* \|(.*?)\| (✅|❌) [`"]true[`"] \|'
        
        # Replace with local availability marker
        if re.search(server_pattern, section, re.IGNORECASE):
            modified_section = re.sub(
                server_pattern, 
                f'| **{server}** |\\1| ✅ `true` 🏠 |', 
                section, 
                flags=re.IGNORECASE
            )
            # Update the content
            content = content.replace(section, modified_section)
            section = modified_section
            print(f"Updated {server} in registry with local availability marker")
    
    # Write the updated content
    with open(TOOLS_MD_PATH, 'w') as f:
        f.write(content)
    
    print(f"Updated tools.md with local server availability")

if __name__ == "__main__":
    update_tools_registry()
EOL

chmod +x "${SERVERS_DIR}/update-tools-registry.py"
echo -e "${GREEN}✅ Created registry update script${NC}"

# Update registry to mark bayes-mcp as locally available
# Skipped for now as we'll run the Python script directly later

echo ""
echo -e "${GREEN}✅ All MCP servers directories created!${NC}"
echo ""
echo -e "${YELLOW}Summary:${NC}"
echo "1. Created directory structure for all 18 MCP servers"
echo "2. Added documentation, configuration, and installation scripts"
echo "3. Updated run-mcp-tool.sh to include all servers"
echo "4. Added special integration for ptolemies knowledge base"
echo ""
echo -e "${YELLOW}Next steps:${NC}"
echo "1. Run ./mcp-servers/update-tools-registry.py to mark local servers in registry"
echo "2. Run ./run-mcp-tool.sh list to see all available servers"
echo "3. Install additional servers as needed with their install scripts"
echo ""