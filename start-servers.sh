#!/bin/bash
# Start individual MCP servers

# Filesystem
npx -y @modelcontextprotocol/server-filesystem .

# Git  
uvx mcp-server-git

# GitHub
npx -y @modelcontextprotocol/server-github

# Fetch
uvx mcp-server-fetch

# Memory
npx -y @modelcontextprotocol/server-memory

# Inspector (debug tool)
npx -y @modelcontextprotocol/inspector

# Bayes MCP
cd /Users/dionedge/devqai/bayes
python bayes_mcp.py