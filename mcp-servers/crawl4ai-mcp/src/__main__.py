"""
Crawl4AI MCP Server Main Entry Point

This module serves as the main entry point for the Crawl4AI MCP server.
"""

import sys
from .mcp_server import app, get_app

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "ingest":
        from .ingest import main
        sys.exit(main())
    else:
        import uvicorn
        uvicorn.run(get_app(), host="0.0.0.0", port=8080)