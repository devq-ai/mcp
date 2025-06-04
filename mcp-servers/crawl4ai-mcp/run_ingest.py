#!/usr/bin/env python3
"""
Run the ingest process for Crawl4AI
"""

import asyncio
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("crawl4ai.ingest_runner")

# Path to the crawl targets file
TARGETS_FILE = "/Users/dionedge/devqai/ptolemies/data/crawl_targets.json"

async def main():
    """Main entry point for the ingest runner."""
    
    logger.info(f"Starting ingest process using targets from {TARGETS_FILE}")
    
    # Import the necessary modules
    try:
        from src.crawler import Crawler, ContentProcessor
        from src.ptolemies_integration import PtolemiesIntegration
    except ImportError as e:
        logger.error(f"Error importing required modules: {e}")
        return 1
    
    # Check if targets file exists
    if not Path(TARGETS_FILE).exists():
        logger.error(f"Targets file not found: {TARGETS_FILE}")
        return 1
    
    # Load targets file
    try:
        with open(TARGETS_FILE, "r") as f:
            data = json.load(f)
    except Exception as e:
        logger.error(f"Error loading targets file: {e}")
        return 1
    
    targets = data.get("targets", [])
    default_config = data.get("default_config", {})
    
    if not targets:
        logger.warning("No targets found in targets file")
        return 1
    
    logger.info(f"Found {len(targets)} targets to process")
    
    # Create components
    crawler = Crawler(
        user_agent=default_config.get("user_agent", "Ptolemies Knowledge Crawler/1.0"),
        respect_robots_txt=default_config.get("respect_robots_txt", True),
        delay_ms=default_config.get("delay_ms", 1000)
    )
    
    content_processor = ContentProcessor()
    
    # Initialize Ptolemies integration
    ptolemies = PtolemiesIntegration()
    await ptolemies.connect()
    
    # Process each target
    for target in targets:
        url = target.get("url")
        if not url:
            logger.warning("Target missing URL, skipping")
            continue
        
        try:
            # Get target settings
            name = target.get("name", url)
            depth = target.get("depth", 2)
            tags = target.get("tags", [])
            category = target.get("category", "Uncategorized")
            priority = target.get("priority", "medium")
            
            # Adjust settings based on priority
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
            
            logger.info(f"Processing target: {name} ({url})")
            logger.info(f"  Depth: {depth}, Max Pages: {max_pages}, Category: {category}")
            
            # Crawl the URL
            crawl_result = await crawler.crawl(
                url=url,
                depth=depth,
                max_pages=max_pages,
                extract_code=default_config.get("extract_code", True),
                extract_tables=default_config.get("extract_tables", True)
            )
            
            logger.info(f"Crawl completed: {len(crawl_result['pages'])} pages crawled")
            
            # Process the pages
            knowledge_items = await content_processor.process_pages(
                pages=crawl_result["pages"],
                tags=tags,
                category=category
            )
            
            logger.info(f"Processed {len(knowledge_items)} knowledge items")
            
            # Store in Ptolemies
            item_ids = await ptolemies.store_knowledge_items(knowledge_items)
            
            logger.info(f"Stored {len(item_ids)} knowledge items in Ptolemies")
            
            # Generate embeddings
            await ptolemies.generate_embeddings(item_ids)
            
            logger.info(f"Requested embeddings for {len(item_ids)} knowledge items")
            
        except Exception as e:
            logger.error(f"Error processing target {url}: {e}")
    
    logger.info("Ingest process completed")
    return 0

if __name__ == "__main__":
    sys.exit(asyncio.run(main()))