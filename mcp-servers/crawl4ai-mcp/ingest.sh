#!/bin/bash
# Run the ingest process for Crawl4AI

echo "Starting Crawl4AI ingest process..."

# Activate virtual environment
source venv/bin/activate

# Run the ingest process
python -m src.ingest targets --file /Users/dionedge/devqai/ptolemies/data/crawl_targets.json

echo "Ingest process completed."