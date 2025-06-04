#!/bin/bash
# Run the ingest process for Crawl4AI MCP server

echo "Starting Crawl4AI ingest process..."

# Change to the script directory
cd "$(dirname "$0")"

# Activate virtual environment
source venv/bin/activate

# Run the ingest process
python run_ingest.py

echo "Ingest process completed."