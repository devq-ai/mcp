#!/bin/bash
# Start the MCP API server

# Ensure we're in the repository root
cd "$(dirname "$0")"

# Activate virtual environment if it exists
if [ -d "pydantic_ai_env" ]; then
  source pydantic_ai_env/bin/activate
fi

# Create logs directory if it doesn't exist
mkdir -p logs

# Update status file first
python update-mcp-status.py

# Start the API server
python mcp-server-api.py