#!/bin/bash
# Update MCP server status and copy to GitHub Pages directory

# Ensure we're in the repository root
cd "$(dirname "$0")"

# Run the status monitor
python mcp-status-monitor.py

# Copy the status file to the GitHub Pages directory
mkdir -p mcp-status-site
cp mcp-status.json mcp-status-site/

echo "Status updated and copied to mcp-status-site/"
echo "You can now commit and push the changes:"
echo "git add mcp-status-site/mcp-status.json"
echo "git commit -m \"Update MCP server status\""
echo "git push"