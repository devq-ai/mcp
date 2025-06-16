#!/bin/bash
# Update MCP server status locally

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Ensure we're in the repository root
cd "$(dirname "$0")"

echo -e "${BLUE}Updating MCP server status...${NC}"

# Create directories if they don't exist
mkdir -p mcp-status-site

# Activate virtual environment if it exists
if [ -d "pydantic_ai_env" ]; then
  echo -e "${BLUE}Activating Python virtual environment...${NC}"
  source pydantic_ai_env/bin/activate
fi

# Install required packages if needed
echo -e "${BLUE}Checking dependencies...${NC}"
pip install gitpython requests

# Run the status monitor
echo -e "${BLUE}Running status monitor...${NC}"
python mcp-status-monitor.py

# Check if GitHub Pages is configured
if [ -d ".git" ]; then
  echo -e "${BLUE}Checking GitHub repository...${NC}"
  
  # Check if we need to commit changes
  if git status --porcelain | grep -q "mcp-status-site"; then
    echo -e "${BLUE}Changes detected, committing to repository...${NC}"
    git add mcp-status-site/
    git commit -m "Update MCP server status: $(date '+%Y-%m-%d %H:%M:%S')"
    
    # Push if remote exists
    if git remote -v | grep -q "origin"; then
      echo -e "${BLUE}Pushing changes to GitHub...${NC}"
      git push origin
      echo -e "${GREEN}✅ Changes pushed to GitHub${NC}"
      echo -e "${GREEN}✅ Status page will be available at: https://devq-ai.github.io/mcp/${NC}"
    else
      echo -e "${RED}⚠️ No remote repository found. Please push changes manually.${NC}"
    fi
  else
    echo -e "${GREEN}✅ No changes to commit${NC}"
  fi
  
  # Check if GitHub Pages workflow exists
  if [ ! -f ".github/workflows/pages.yml" ]; then
    echo -e "${RED}⚠️ GitHub Pages workflow not found!${NC}"
    echo -e "${BLUE}Creating GitHub Pages workflow...${NC}"
    
    mkdir -p .github/workflows
    cat > .github/workflows/pages.yml << 'EOL'
name: Deploy to GitHub Pages

on:
  push:
    branches: [main]
    paths:
      - 'mcp-status-site/**'
  workflow_dispatch:

permissions:
  contents: read
  pages: write
  id-token: write

concurrency:
  group: "pages"
  cancel-in-progress: false

jobs:
  deploy:
    environment:
      name: github-pages
      url: ${{ steps.deployment.outputs.page_url }}
    runs-on: ubuntu-latest
    steps:
      - name: Checkout
        uses: actions/checkout@v4
      
      - name: Setup Pages
        uses: actions/configure-pages@v4
        
      - name: Upload artifact
        uses: actions/upload-pages-artifact@v2
        with:
          path: './mcp-status-site'
          
      - name: Deploy to GitHub Pages
        id: deployment
        uses: actions/deploy-pages@v4
EOL

    git add .github/workflows/pages.yml
    git commit -m "Add GitHub Pages workflow"
    
    if git remote -v | grep -q "origin"; then
      git push origin
      echo -e "${GREEN}✅ GitHub Pages workflow added and pushed${NC}"
    else
      echo -e "${RED}⚠️ No remote repository found. Please push changes manually.${NC}"
    fi
    
    echo -e "${BLUE}Important: You need to enable GitHub Pages in repository settings:${NC}"
    echo -e "${BLUE}1. Go to your repository on GitHub${NC}"
    echo -e "${BLUE}2. Click Settings > Pages${NC}"
    echo -e "${BLUE}3. Under 'Build and deployment', select 'GitHub Actions'${NC}"
  fi
else
  echo -e "${RED}⚠️ Not a git repository. Cannot push changes to GitHub.${NC}"
fi

echo -e "${GREEN}✅ MCP status update complete!${NC}"
echo -e "${BLUE}To view the status page locally, open:${NC}"
echo -e "${BLUE}$(pwd)/mcp-status-site/index.html${NC}"
