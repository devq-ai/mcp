#!/usr/bin/env python3
"""
Crawl4AI Core Crawler Module

This module provides the core functionality for web crawling,
content extraction, and processing.
"""

import asyncio
import logging
import json
from datetime import datetime
from typing import Dict, List, Any, Optional, Set
from urllib.parse import urljoin, urlparse

import httpx
from bs4 import BeautifulSoup
import trafilatura

logger = logging.getLogger("crawl4ai.crawler")


class Crawler:
    """Core crawler implementation for Crawl4AI."""
    
    def __init__(self, 
                 user_agent: str = "Crawl4AI/1.0",
                 respect_robots_txt: bool = True,
                 delay_ms: int = 1000,
                 timeout_s: int = 30):
        """Initialize the crawler.
        
        Args:
            user_agent: User agent string to use for requests
            respect_robots_txt: Whether to respect robots.txt
            delay_ms: Delay between requests in milliseconds
            timeout_s: Request timeout in seconds
        """
        self.user_agent = user_agent
        self.respect_robots_txt = respect_robots_txt
        self.delay_ms = delay_ms
        self.timeout_s = timeout_s
        self.visited_urls: Set[str] = set()
        self.logger = logger
    
    async def crawl(self, 
                   url: str, 
                   depth: int = 2, 
                   max_pages: int = 100,
                   extract_code: bool = True,
                   extract_tables: bool = True) -> Dict[str, Any]:
        """Crawl a URL and its linked pages.
        
        Args:
            url: The URL to crawl
            depth: How deep to follow links (default: 2)
            max_pages: Maximum pages to crawl (default: 100)
            extract_code: Whether to extract code blocks
            extract_tables: Whether to extract tables
                
        Returns:
            Dictionary with crawl results
        """
        self.visited_urls = set()
        pages = []
        start_time = datetime.now()
        
        self.logger.info(f"Starting crawl of {url} with depth {depth}")
        
        # Create semaphore to limit concurrent requests
        semaphore = asyncio.Semaphore(5)
        
        async def _crawl_url(current_url: str, current_depth: int):
            if current_url in self.visited_urls:
                return
            
            if len(self.visited_urls) >= max_pages:
                return
            
            if current_depth > depth:
                return
            
            # Mark as visited before processing to avoid duplicates
            self.visited_urls.add(current_url)
            
            # Apply rate limiting with semaphore
            async with semaphore:
                try:
                    # Apply delay
                    await asyncio.sleep(self.delay_ms / 1000)
                    
                    self.logger.debug(f"Crawling {current_url} (depth {current_depth})")
                    
                    # Fetch the page
                    async with httpx.AsyncClient(timeout=self.timeout_s) as client:
                        headers = {"User-Agent": self.user_agent}
                        response = await client.get(current_url, headers=headers, follow_redirects=True)
                        response.raise_for_status()
                        
                        # Process the page content
                        page_data = await self._process_page(
                            current_url,
                            response.text,
                            extract_code=extract_code,
                            extract_tables=extract_tables
                        )
                        
                        pages.append(page_data)
                        
                        # If we're at max depth, don't extract links
                        if current_depth >= depth:
                            return
                        
                        # Extract links and crawl them
                        links = self._extract_links(current_url, response.text)
                        tasks = []
                        
                        for link in links:
                            if link not in self.visited_urls and len(self.visited_urls) < max_pages:
                                tasks.append(_crawl_url(link, current_depth + 1))
                        
                        if tasks:
                            await asyncio.gather(*tasks)
                    
                except Exception as e:
                    self.logger.error(f"Error crawling {current_url}: {e}")
        
        # Start crawling from the initial URL
        await _crawl_url(url, 1)
        
        end_time = datetime.now()
        duration_ms = (end_time - start_time).total_seconds() * 1000
        
        # Prepare the result
        result = {
            "pages": pages,
            "stats": {
                "pages_crawled": len(pages),
                "total_content_bytes": sum(len(page.get("content", "")) for page in pages),
                "time_ms": int(duration_ms),
                "start_time": start_time.isoformat(),
                "end_time": end_time.isoformat()
            }
        }
        
        self.logger.info(f"Crawl completed: {len(pages)} pages crawled in {duration_ms:.0f}ms")
        return result
    
    async def _process_page(self, 
                           url: str, 
                           html: str,
                           extract_code: bool = True,
                           extract_tables: bool = True) -> Dict[str, Any]:
        """Process a page and extract its content.
        
        Args:
            url: The URL of the page
            html: The HTML content of the page
            extract_code: Whether to extract code blocks
            extract_tables: Whether to extract tables
            
        Returns:
            Dictionary with processed page data
        """
        # Use trafilatura for main content extraction
        extracted_text = trafilatura.extract(html, include_comments=False, 
                                            include_tables=extract_tables, 
                                            include_images=True,
                                            include_links=True)
        
        # Fallback to BeautifulSoup if trafilatura fails
        if not extracted_text:
            soup = BeautifulSoup(html, 'html.parser')
            
            # Try to get the article or main content
            main_content = None
            for selector in ['article', 'main', '.content', '#content', '.post', '.article']:
                main_content = soup.select_one(selector)
                if main_content:
                    break
            
            if main_content:
                extracted_text = main_content.get_text(separator='\n\n')
            else:
                # Fallback to body
                extracted_text = soup.body.get_text(separator='\n\n') if soup.body else ""
        
        # Extract metadata
        soup = BeautifulSoup(html, 'html.parser')
        title = soup.title.string if soup.title else ""
        
        # Extract meta description
        description = ""
        meta_desc = soup.find('meta', attrs={'name': 'description'})
        if meta_desc:
            description = meta_desc.get('content', '')
        
        # Extract code blocks if requested
        code_blocks = []
        if extract_code:
            for pre in soup.find_all('pre'):
                code = pre.get_text()
                if code and len(code.strip()) > 0:
                    code_blocks.append(code)
        
        # Extract tables if requested
        tables = []
        if extract_tables:
            for table in soup.find_all('table'):
                tables.append(str(table))
        
        # Prepare the page data
        page_data = {
            "url": url,
            "title": title,
            "description": description,
            "content": extracted_text,
            "code_blocks": code_blocks,
            "tables": tables,
            "metadata": {
                "crawl_time": datetime.now().isoformat(),
                "content_type": soup.find('meta', {'http-equiv': 'Content-Type'})['content'] if soup.find('meta', {'http-equiv': 'Content-Type'}) else "text/html"
            }
        }
        
        return page_data
    
    def _extract_links(self, base_url: str, html: str) -> List[str]:
        """Extract links from HTML content.
        
        Args:
            base_url: The base URL for resolving relative links
            html: The HTML content
            
        Returns:
            List of absolute URLs
        """
        soup = BeautifulSoup(html, 'html.parser')
        links = []
        
        base_domain = urlparse(base_url).netloc
        
        for a in soup.find_all('a', href=True):
            href = a['href']
            
            # Skip fragment links
            if href.startswith('#'):
                continue
            
            # Skip JavaScript links
            if href.startswith('javascript:'):
                continue
            
            # Skip mailto links
            if href.startswith('mailto:'):
                continue
            
            # Make relative URLs absolute
            absolute_url = urljoin(base_url, href)
            
            # Only include links to the same domain
            parsed_url = urlparse(absolute_url)
            if parsed_url.netloc == base_domain:
                # Normalize URL (remove fragments)
                normalized_url = parsed_url._replace(fragment='').geturl()
                links.append(normalized_url)
        
        return links


class ContentProcessor:
    """Process crawled content into knowledge items."""
    
    def __init__(self):
        """Initialize the content processor."""
        self.logger = logging.getLogger("crawl4ai.content_processor")
    
    async def process_pages(self, 
                          pages: List[Dict[str, Any]], 
                          tags: Optional[List[str]] = None,
                          category: Optional[str] = None) -> List[Dict[str, Any]]:
        """Process crawled pages into knowledge items.
        
        Args:
            pages: List of crawled pages
            tags: Tags to apply to all knowledge items
            category: Category for all knowledge items
            
        Returns:
            List of knowledge items
        """
        knowledge_items = []
        
        for page in pages:
            try:
                # Create a knowledge item from the page
                item = {
                    "title": page.get("title", "Untitled"),
                    "content": page.get("content", ""),
                    "source": page.get("url", ""),
                    "source_type": "web",
                    "content_type": "text/html",
                    "tags": tags or [],
                    "category": category,
                    "metadata": {
                        "crawl_timestamp": datetime.now().isoformat(),
                        "description": page.get("description", ""),
                        "has_code_blocks": len(page.get("code_blocks", [])) > 0,
                        "has_tables": len(page.get("tables", [])) > 0,
                        "page_metadata": page.get("metadata", {})
                    },
                    "created_at": datetime.now().isoformat(),
                    "updated_at": datetime.now().isoformat()
                }
                
                # Add code blocks if available
                if page.get("code_blocks"):
                    item["code_blocks"] = page.get("code_blocks")
                
                # Add tables if available
                if page.get("tables"):
                    item["tables"] = page.get("tables")
                
                knowledge_items.append(item)
                self.logger.debug(f"Created knowledge item from {page.get('url')}")
                
            except Exception as e:
                self.logger.error(f"Error processing page {page.get('url')}: {e}")
        
        self.logger.info(f"Processed {len(knowledge_items)} knowledge items")
        return knowledge_items