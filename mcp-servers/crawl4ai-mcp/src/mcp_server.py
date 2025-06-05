#!/usr/bin/env python3
"""
Crawl4AI MCP Server

This module implements the Model Context Protocol (MCP) server for
Crawl4AI, providing web crawling services via MCP.
"""

import asyncio
import json
import logging
import os
from datetime import datetime
from typing import Dict, List, Any, Optional

from fastapi import FastAPI, HTTPException, Depends, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from dotenv import load_dotenv

from .crawler import Crawler, ContentProcessor

# Load environment variables
load_dotenv()

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("crawl4ai.mcp_server")

# Configure Logfire if available
logfire_initialized = False
try:
    import logfire
    
    # Initialize Logfire if token is available
    logfire_token = os.getenv("LOGFIRE_TOKEN")
    if logfire_token:
        logfire.configure(
            token=logfire_token,
            service_name=os.getenv("LOGFIRE_SERVICE_NAME", "crawl4ai-mcp"),
            environment=os.getenv("LOGFIRE_ENVIRONMENT", "development")
        )
        logfire_initialized = True
        logger.info("Logfire initialized successfully")
    else:
        logger.info("Logfire token not found, running without Logfire")
except ImportError:
    logger.info("Logfire not installed, running without observability")
    logfire = None

# Create the FastAPI app
app = FastAPI(
    title="Crawl4AI MCP Server",
    description="MCP server for web crawling and content extraction",
    version="0.1.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Instrument FastAPI with Logfire if available
if logfire and logfire_token:
    logfire.instrument_fastapi(app)
    logger.info("FastAPI instrumented with Logfire")

# Load configuration
def load_config():
    config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "config.json")
    with open(config_path, "r") as f:
        return json.load(f)

config = load_config()

# Create crawler instance
crawler = Crawler(
    user_agent=config.get("crawler", {}).get("user_agent", "Crawl4AI/1.0"),
    respect_robots_txt=config.get("crawler", {}).get("respect_robots_txt", True),
    delay_ms=config.get("crawler", {}).get("delay_ms", 1000),
    timeout_s=config.get("crawler", {}).get("timeout_s", 30)
)

# Create content processor
content_processor = ContentProcessor()

# API models
class CrawlRequest(BaseModel):
    tool: str = Field(..., description="Tool name, must be 'crawl4ai'")
    operation: str = Field(..., description="Operation to perform")
    parameters: Dict[str, Any] = Field(..., description="Operation parameters")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Request metadata")

class CrawlResponse(BaseModel):
    result: Optional[Dict[str, Any]] = Field(None, description="Operation result")
    error: Optional[Dict[str, Any]] = Field(None, description="Error information if operation failed")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Response metadata")

# API Key validation
async def validate_api_key(request: Request):
    expected_api_key = config.get("configuration", {}).get("api_key", "your_api_key_here")
    
    # Skip validation if key is the default placeholder
    if expected_api_key == "your_api_key_here":
        return True
        
    api_key = request.headers.get("Authorization")
    if not api_key:
        raise HTTPException(status_code=401, detail="API key missing")
    
    # Extract key from Bearer token
    if api_key.startswith("Bearer "):
        api_key = api_key[7:]
    
    if api_key != expected_api_key:
        raise HTTPException(status_code=401, detail="Invalid API key")
    
    return True

# MCP endpoints
@app.post("/tools/crawl4ai/invoke", response_model=CrawlResponse)
async def invoke_tool(request: CrawlRequest, _: bool = Depends(validate_api_key)):
    """Handle MCP tool invocation requests."""
    start_time = datetime.now()
    
    # Validate tool
    if request.tool != "crawl4ai":
        return CrawlResponse(
            error={"message": "Invalid tool", "code": "invalid_tool"},
            metadata={"timestamp": datetime.now().isoformat()}
        )
    
    # Handle operations
    try:
        if request.operation == "crawl":
            result = await handle_crawl(request.parameters)
        elif request.operation == "process":
            result = await handle_process(request.parameters)
        else:
            return CrawlResponse(
                error={"message": f"Invalid operation: {request.operation}", "code": "invalid_operation"},
                metadata={"timestamp": datetime.now().isoformat()}
            )
        
        # Calculate processing time
        end_time = datetime.now()
        duration_ms = (end_time - start_time).total_seconds() * 1000
        
        # Return the result
        return CrawlResponse(
            result=result,
            metadata={
                "timestamp": datetime.now().isoformat(),
                "processing_time_ms": int(duration_ms),
                "request_id": request.metadata.get("request_id") if request.metadata else None
            }
        )
        
    except Exception as e:
        logger.error(f"Error handling request: {e}")
        return CrawlResponse(
            error={"message": str(e), "code": "processing_error"},
            metadata={"timestamp": datetime.now().isoformat()}
        )

async def handle_crawl(parameters: Dict[str, Any]) -> Dict[str, Any]:
    """Handle a crawl operation."""
    url = parameters.get("url")
    if not url:
        raise ValueError("URL is required")
    
    depth = parameters.get("depth", 2)
    max_pages = parameters.get("max_pages", 100)
    extract_code = parameters.get("extract_code", True)
    extract_tables = parameters.get("extract_tables", True)
    
    logger.info(f"Handling crawl request for {url} with depth {depth}")
    
    if logfire:
        with logfire.span("crawl_operation", 
                         url=url, 
                         depth=depth, 
                         max_pages=max_pages):
            logfire.info("Starting crawl operation",
                        url=url,
                        depth=depth,
                        max_pages=max_pages,
                        extract_code=extract_code,
                        extract_tables=extract_tables)
            
            result = await crawler.crawl(
                url=url,
                depth=depth,
                max_pages=max_pages,
                extract_code=extract_code,
                extract_tables=extract_tables
            )
            
            pages_crawled = len(result.get("pages", []))
            logfire.info("Crawl operation completed",
                        url=url,
                        pages_crawled=pages_crawled,
                        success=True)
            
            return result
    else:
        result = await crawler.crawl(
            url=url,
            depth=depth,
            max_pages=max_pages,
            extract_code=extract_code,
            extract_tables=extract_tables
        )
        
        return result

async def handle_process(parameters: Dict[str, Any]) -> Dict[str, Any]:
    """Handle a content processing operation."""
    pages = parameters.get("pages")
    if not pages:
        raise ValueError("Pages are required")
    
    tags = parameters.get("tags", [])
    category = parameters.get("category")
    
    logger.info(f"Processing {len(pages)} pages")
    
    knowledge_items = await content_processor.process_pages(
        pages=pages,
        tags=tags,
        category=category
    )
    
    return {"knowledge_items": knowledge_items}

# Health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "ok", "timestamp": datetime.now().isoformat()}

# Return the app for ASGI servers
def get_app():
    return app

if __name__ == "__main__":
    import uvicorn
    
    host = config.get("configuration", {}).get("host", "localhost")
    port = config.get("configuration", {}).get("port", 8080)
    
    uvicorn.run(app, host=host, port=port)