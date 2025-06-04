#!/usr/bin/env python3
"""
Crawl4AI Ingest CLI

This script provides a command-line interface for crawling web content
and ingesting it into the Ptolemies Knowledge Base.
"""

import argparse
import asyncio
import json
import logging
import os
import sys
from datetime import datetime
from typing import Dict, List, Any, Optional

from .crawler import Crawler, ContentProcessor
from .ptolemies_integration import PtolemiesIntegration

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("crawl4ai.ingest")

async def ingest_url(url: str, depth: int, tags: List[str], category: str, 
                    extract_code: bool, extract_tables: bool,
                    max_pages: int, respect_robots_txt: bool,
                    delay_ms: int, user_agent: str) -> Dict[str, Any]:
    """Ingest content from a URL into the knowledge base."""
    logger.info(f"Ingesting content from {url} (depth: {depth}, category: {category})")
    
    # Create crawler
    crawler = Crawler(
        user_agent=user_agent,
        respect_robots_txt=respect_robots_txt,
        delay_ms=delay_ms
    )
    
    # Crawl the URL
    crawl_result = await crawler.crawl(
        url=url,
        depth=depth,
        max_pages=max_pages,
        extract_code=extract_code,
        extract_tables=extract_tables
    )
    
    # Process the crawled pages
    content_processor = ContentProcessor()
    knowledge_items = await content_processor.process_pages(
        pages=crawl_result["pages"],
        tags=tags,
        category=category
    )
    
    # Store in Ptolemies Knowledge Base
    ptolemies = PtolemiesIntegration()
    await ptolemies.connect()
    
    item_ids = await ptolemies.store_knowledge_items(knowledge_items)
    
    # Generate embeddings
    await ptolemies.generate_embeddings(item_ids)
    
    return {
        "url": url,
        "crawl_stats": crawl_result["stats"],
        "knowledge_items_created": len(item_ids),
        "knowledge_item_ids": item_ids
    }

async def ingest_from_targets(targets_file: str) -> Dict[str, Any]:
    """Ingest content from targets defined in a JSON file."""
    logger.info(f"Ingesting content from targets in {targets_file}")
    
    # Load targets file
    with open(targets_file, "r") as f:
        data = json.load(f)
    
    targets = data.get("targets", [])
    default_config = data.get("default_config", {})
    
    if not targets:
        logger.warning("No targets found in targets file")
        return {"status": "error", "message": "No targets found"}
    
    logger.info(f"Found {len(targets)} targets to process")
    
    results = []
    
    for target in targets:
        url = target.get("url")
        if not url:
            logger.warning("Target missing URL, skipping")
            continue
        
        try:
            # Combine default config with target-specific settings
            depth = target.get("depth", 2)
            tags = target.get("tags", [])
            category = target.get("category", "Uncategorized")
            priority = target.get("priority", "medium")
            
            extract_code = target.get("extract_code", default_config.get("extract_code", True))
            extract_tables = target.get("extract_tables", default_config.get("extract_tables", True))
            respect_robots_txt = target.get("respect_robots_txt", default_config.get("respect_robots_txt", True))
            user_agent = target.get("user_agent", default_config.get("user_agent", "Crawl4AI/1.0"))
            
            # Adjust max_pages and delay based on priority
            max_pages = {
                "critical": 200,
                "high": 100,
                "medium": 50,
                "low": 20
            }.get(priority, 50)
            
            delay_ms = {
                "critical": 500,
                "high": 1000,
                "medium": 1500,
                "low": 2000
            }.get(priority, 1000)
            
            # Ingest the URL
            result = await ingest_url(
                url=url,
                depth=depth,
                tags=tags,
                category=category,
                extract_code=extract_code,
                extract_tables=extract_tables,
                max_pages=max_pages,
                respect_robots_txt=respect_robots_txt,
                delay_ms=delay_ms,
                user_agent=user_agent
            )
            
            results.append({
                "target_name": target.get("name", url),
                "result": result
            })
            
            logger.info(f"Completed ingestion for {url}")
            
        except Exception as e:
            logger.error(f"Error ingesting {url}: {e}")
            results.append({
                "target_name": target.get("name", url),
                "error": str(e)
            })
    
    return {
        "status": "success",
        "targets_processed": len(results),
        "results": results,
        "timestamp": datetime.now().isoformat()
    }

def main():
    """Main entry point for the ingest CLI."""
    parser = argparse.ArgumentParser(description="Crawl4AI Ingest CLI")
    
    # Command subparsers
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")
    
    # Ingest URL command
    url_parser = subparsers.add_parser("url", help="Ingest content from a URL")
    url_parser.add_argument("url", help="URL to crawl")
    url_parser.add_argument("--depth", type=int, default=2, help="Crawl depth")
    url_parser.add_argument("--tags", nargs="+", default=[], help="Tags to apply")
    url_parser.add_argument("--category", default="Uncategorized", help="Content category")
    url_parser.add_argument("--extract-code", action="store_true", help="Extract code blocks")
    url_parser.add_argument("--extract-tables", action="store_true", help="Extract tables")
    url_parser.add_argument("--max-pages", type=int, default=100, help="Maximum pages to crawl")
    
    # Ingest from targets file command
    targets_parser = subparsers.add_parser("targets", help="Ingest content from targets file")
    targets_parser.add_argument(
        "--file", 
        default="/Users/dionedge/devqai/ptolemies/data/crawl_targets.json",
        help="Path to targets JSON file"
    )
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return 1
    
    # Run the appropriate command
    if args.command == "url":
        result = asyncio.run(ingest_url(
            url=args.url,
            depth=args.depth,
            tags=args.tags,
            category=args.category,
            extract_code=args.extract_code,
            extract_tables=args.extract_tables,
            max_pages=args.max_pages,
            respect_robots_txt=True,
            delay_ms=1000,
            user_agent="Crawl4AI/1.0"
        ))
        print(json.dumps(result, indent=2))
        
    elif args.command == "targets":
        if not os.path.exists(args.file):
            logger.error(f"Targets file not found: {args.file}")
            return 1
            
        result = asyncio.run(ingest_from_targets(args.file))
        print(json.dumps(result, indent=2))
    
    return 0

if __name__ == "__main__":
    sys.exit(main())