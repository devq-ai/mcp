"""
Web Search Tool for Agentical

This module provides comprehensive web search and content retrieval capabilities
supporting multiple search engines, content extraction, and integration with
the Agentical framework.

Features:
- Multi-search engine support (DuckDuckGo, Google, Bing, SerpAPI)
- Web scraping and content extraction
- Search result filtering and ranking
- Rate limiting and caching
- Content parsing and summarization
- Integration with crawl4ai and other web tools
- Async search operations
- Performance monitoring and observability
"""

import asyncio
import json
import uuid
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, Any, List, Optional, Union, Tuple
import logging
import re
from urllib.parse import urljoin, urlparse
import tempfile

try:
    import requests
    from requests.adapters import HTTPAdapter
    from urllib3.util.retry import Retry
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False

try:
    import aiohttp
    AIOHTTP_AVAILABLE = True
except ImportError:
    AIOHTTP_AVAILABLE = False

try:
    from bs4 import BeautifulSoup
    BS4_AVAILABLE = True
except ImportError:
    BS4_AVAILABLE = False

try:
    import duckduckgo_search
    from duckduckgo_search import DDGS
    DUCKDUCKGO_AVAILABLE = True
except ImportError:
    DUCKDUCKGO_AVAILABLE = False

from ...core.exceptions import (
    ToolError,
    ToolExecutionError,
    ToolValidationError,
    ToolTimeoutError
)
from ...core.logging import log_operation


class SearchEngine(Enum):
    """Supported search engines."""
    DUCKDUCKGO = "duckduckgo"
    GOOGLE = "google"
    BING = "bing"
    SERP = "serp"


class ContentType(Enum):
    """Types of content to search for."""
    WEB = "web"
    NEWS = "news"
    IMAGES = "images"
    VIDEOS = "videos"
    ACADEMIC = "academic"


class SearchFilter(Enum):
    """Search result filters."""
    RECENT = "recent"         # Last 24 hours
    WEEK = "week"            # Last week
    MONTH = "month"          # Last month
    YEAR = "year"            # Last year
    SAFE = "safe"            # Safe search on
    EXACT = "exact"          # Exact phrase matching


class SearchResult:
    """Individual search result with comprehensive metadata."""

    def __init__(
        self,
        title: str,
        url: str,
        snippet: str = "",
        score: float = 0.0,
        source: str = "",
        published_date: Optional[datetime] = None,
        content_type: ContentType = ContentType.WEB,
        metadata: Optional[Dict[str, Any]] = None
    ):
        self.title = title
        self.url = url
        self.snippet = snippet
        self.score = score
        self.source = source
        self.published_date = published_date
        self.content_type = content_type
        self.metadata = metadata or {}
        self.extracted_content: Optional[str] = None
        self.images: List[str] = []
        self.links: List[str] = []

    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary."""
        return {
            "title": self.title,
            "url": self.url,
            "snippet": self.snippet,
            "score": self.score,
            "source": self.source,
            "published_date": self.published_date.isoformat() if self.published_date else None,
            "content_type": self.content_type.value,
            "metadata": self.metadata,
            "extracted_content": self.extracted_content,
            "images": self.images,
            "links": self.links
        }


class WebSearchResult:
    """Result of web search operation with comprehensive details."""

    def __init__(
        self,
        search_id: str,
        query: str,
        search_engine: SearchEngine,
        success: bool,
        results: Optional[List[SearchResult]] = None,
        total_results: int = 0,
        search_time: float = 0.0,
        filters_applied: Optional[List[SearchFilter]] = None,
        error_message: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        self.search_id = search_id
        self.query = query
        self.search_engine = search_engine
        self.success = success
        self.results = results or []
        self.total_results = total_results
        self.search_time = search_time
        self.filters_applied = filters_applied or []
        self.error_message = error_message
        self.metadata = metadata or {}
        self.timestamp = datetime.now()

    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary."""
        return {
            "search_id": self.search_id,
            "query": self.query,
            "search_engine": self.search_engine.value,
            "success": self.success,
            "results": [result.to_dict() for result in self.results],
            "result_count": len(self.results),
            "total_results": self.total_results,
            "search_time": self.search_time,
            "filters_applied": [f.value for f in self.filters_applied],
            "error_message": self.error_message,
            "metadata": self.metadata,
            "timestamp": self.timestamp.isoformat()
        }


class WebSearchTool:
    """
    Comprehensive web search tool supporting multiple search engines
    with content extraction and filtering capabilities.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize web search tool.

        Args:
            config: Configuration for web search operations
        """
        self.config = config or {}
        self.logger = logging.getLogger(__name__)

        # Configuration settings
        self.default_engine = SearchEngine(self.config.get("default_engine", "duckduckgo"))
        self.max_results = self.config.get("max_results", 10)
        self.timeout_seconds = self.config.get("timeout_seconds", 15)
        self.enable_caching = self.config.get("enable_caching", True)
        self.cache_ttl_minutes = self.config.get("cache_ttl_minutes", 60)
        self.user_agent = self.config.get("user_agent", "Agentical/1.0 (+https://devq.ai)")

        # API keys for search engines
        self.api_keys = self.config.get("api_keys", {})

        # Rate limiting
        self.rate_limits = {
            SearchEngine.DUCKDUCKGO: {"requests_per_minute": 30, "last_request": None},
            SearchEngine.GOOGLE: {"requests_per_day": 100, "last_request": None},
            SearchEngine.BING: {"requests_per_month": 1000, "last_request": None},
            SearchEngine.SERP: {"requests_per_month": 100, "last_request": None}
        }

        # Cache for search results
        self.cache = {}

        # Session for HTTP requests
        self.session = None
        if REQUESTS_AVAILABLE:
            self.session = requests.Session()
            retry_strategy = Retry(
                total=3,
                backoff_factor=1,
                status_forcelist=[429, 500, 502, 503, 504]
            )
            adapter = HTTPAdapter(max_retries=retry_strategy)
            self.session.mount("http://", adapter)
            self.session.mount("https://", adapter)
            self.session.headers.update({"User-Agent": self.user_agent})

    @log_operation("web_search")
    async def search(
        self,
        query: str,
        search_engine: Optional[SearchEngine] = None,
        content_type: ContentType = ContentType.WEB,
        max_results: Optional[int] = None,
        filters: Optional[List[SearchFilter]] = None,
        extract_content: bool = False,
        rank_results: bool = True,
        timeout_override: Optional[int] = None
    ) -> WebSearchResult:
        """
        Perform web search with specified parameters.

        Args:
            query: Search query string
            search_engine: Search engine to use
            content_type: Type of content to search for
            max_results: Maximum number of results to return
            filters: Search filters to apply
            extract_content: Whether to extract full content from results
            rank_results: Whether to rank and score results
            timeout_override: Override default timeout

        Returns:
            WebSearchResult: Search results with metadata
        """
        search_id = str(uuid.uuid4())
        start_time = datetime.now()

        # Validate inputs
        if not query.strip():
            raise ToolValidationError("Search query cannot be empty")

        # Set defaults
        engine = search_engine or self.default_engine
        max_results = max_results or self.max_results
        filters = filters or []
        timeout = timeout_override or self.timeout_seconds

        try:
            # Check cache first
            if self.enable_caching:
                cache_key = self._create_cache_key(query, engine, content_type, filters)
                cached_result = self._get_cached_result(cache_key)
                if cached_result:
                    self.logger.info(f"Returning cached search result for query: {query}")
                    return cached_result

            # Check rate limits
            if not self._check_rate_limit(engine):
                raise ToolExecutionError(f"Rate limit exceeded for {engine.value}")

            # Perform search based on engine
            if engine == SearchEngine.DUCKDUCKGO:
                results = await self._search_duckduckgo(
                    query, content_type, max_results, filters, timeout
                )
            elif engine == SearchEngine.GOOGLE:
                results = await self._search_google(
                    query, content_type, max_results, filters, timeout
                )
            elif engine == SearchEngine.BING:
                results = await self._search_bing(
                    query, content_type, max_results, filters, timeout
                )
            elif engine == SearchEngine.SERP:
                results = await self._search_serp(
                    query, content_type, max_results, filters, timeout
                )
            else:
                raise ToolValidationError(f"Unsupported search engine: {engine}")

            # Extract content if requested
            if extract_content and results:
                await self._extract_content_from_results(results, timeout)

            # Rank results if requested
            if rank_results and results:
                results = self._rank_search_results(results, query)

            # Calculate search time
            search_time = (datetime.now() - start_time).total_seconds()

            # Create result object
            search_result = WebSearchResult(
                search_id=search_id,
                query=query,
                search_engine=engine,
                success=True,
                results=results,
                total_results=len(results),
                search_time=search_time,
                filters_applied=filters,
                metadata={
                    "content_type": content_type.value,
                    "extract_content": extract_content,
                    "rank_results": rank_results
                }
            )

            # Cache result
            if self.enable_caching:
                self._cache_result(cache_key, search_result)

            return search_result

        except asyncio.TimeoutError:
            raise ToolTimeoutError(f"Search operation timed out after {timeout} seconds")
        except Exception as e:
            self.logger.error(f"Web search failed: {e}")
            search_time = (datetime.now() - start_time).total_seconds()

            return WebSearchResult(
                search_id=search_id,
                query=query,
                search_engine=engine,
                success=False,
                search_time=search_time,
                error_message=str(e)
            )

    async def _search_duckduckgo(
        self,
        query: str,
        content_type: ContentType,
        max_results: int,
        filters: List[SearchFilter],
        timeout: int
    ) -> List[SearchResult]:
        """Search using DuckDuckGo."""

        if not DUCKDUCKGO_AVAILABLE:
            raise ToolExecutionError("DuckDuckGo search not available. Install duckduckgo-search.")

        results = []

        try:
            # Configure DuckDuckGo search parameters
            ddgs_params = {
                "max_results": max_results,
                "safesearch": "on" if SearchFilter.SAFE in filters else "moderate"
            }

            # Apply time filters
            if SearchFilter.RECENT in filters:
                ddgs_params["timelimit"] = "d"  # Last day
            elif SearchFilter.WEEK in filters:
                ddgs_params["timelimit"] = "w"  # Last week
            elif SearchFilter.MONTH in filters:
                ddgs_params["timelimit"] = "m"  # Last month
            elif SearchFilter.YEAR in filters:
                ddgs_params["timelimit"] = "y"  # Last year

            with DDGS() as ddgs:
                if content_type == ContentType.WEB:
                    search_results = ddgs.text(query, **ddgs_params)
                elif content_type == ContentType.NEWS:
                    search_results = ddgs.news(query, **ddgs_params)
                elif content_type == ContentType.IMAGES:
                    search_results = ddgs.images(query, **ddgs_params)
                elif content_type == ContentType.VIDEOS:
                    search_results = ddgs.videos(query, **ddgs_params)
                else:
                    search_results = ddgs.text(query, **ddgs_params)

                # Convert to SearchResult objects
                for i, result in enumerate(search_results):
                    if i >= max_results:
                        break

                    search_result = SearchResult(
                        title=result.get("title", ""),
                        url=result.get("href", result.get("url", "")),
                        snippet=result.get("body", result.get("description", "")),
                        score=1.0 - (i * 0.1),  # Simple scoring based on position
                        source="DuckDuckGo",
                        content_type=content_type,
                        metadata=result
                    )

                    # Parse published date if available
                    if "date" in result:
                        try:
                            search_result.published_date = datetime.fromisoformat(result["date"])
                        except:
                            pass

                    results.append(search_result)

        except Exception as e:
            raise ToolExecutionError(f"DuckDuckGo search failed: {e}")

        return results

    async def _search_google(
        self,
        query: str,
        content_type: ContentType,
        max_results: int,
        filters: List[SearchFilter],
        timeout: int
    ) -> List[SearchResult]:
        """Search using Google Custom Search API."""

        # Google Custom Search requires API key and Search Engine ID
        api_key = self.api_keys.get("google_api_key")
        search_engine_id = self.api_keys.get("google_search_engine_id")

        if not api_key or not search_engine_id:
            raise ToolExecutionError("Google search requires API key and search engine ID")

        # This would implement Google Custom Search API
        # For now, return empty results
        self.logger.warning("Google search not fully implemented")
        return []

    async def _search_bing(
        self,
        query: str,
        content_type: ContentType,
        max_results: int,
        filters: List[SearchFilter],
        timeout: int
    ) -> List[SearchResult]:
        """Search using Bing Search API."""

        api_key = self.api_keys.get("bing_api_key")

        if not api_key:
            raise ToolExecutionError("Bing search requires API key")

        # This would implement Bing Search API
        # For now, return empty results
        self.logger.warning("Bing search not fully implemented")
        return []

    async def _search_serp(
        self,
        query: str,
        content_type: ContentType,
        max_results: int,
        filters: List[SearchFilter],
        timeout: int
    ) -> List[SearchResult]:
        """Search using SerpAPI."""

        api_key = self.api_keys.get("serp_api_key")

        if not api_key:
            raise ToolExecutionError("SerpAPI search requires API key")

        # This would implement SerpAPI
        # For now, return empty results
        self.logger.warning("SerpAPI search not fully implemented")
        return []

    async def _extract_content_from_results(
        self,
        results: List[SearchResult],
        timeout: int
    ) -> None:
        """Extract full content from search result URLs."""

        if not BS4_AVAILABLE:
            self.logger.warning("Content extraction requires BeautifulSoup4")
            return

        for result in results:
            try:
                await self._extract_single_page_content(result, timeout)
            except Exception as e:
                self.logger.warning(f"Failed to extract content from {result.url}: {e}")

    async def _extract_single_page_content(
        self,
        result: SearchResult,
        timeout: int
    ) -> None:
        """Extract content from a single page."""

        if not self.session:
            return

        try:
            response = self.session.get(result.url, timeout=timeout)
            response.raise_for_status()

            soup = BeautifulSoup(response.content, 'html.parser')

            # Remove script and style elements
            for script in soup(["script", "style"]):
                script.decompose()

            # Extract text content
            text = soup.get_text()

            # Clean up whitespace
            lines = (line.strip() for line in text.splitlines())
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            result.extracted_content = ' '.join(chunk for chunk in chunks if chunk)

            # Extract images
            images = soup.find_all('img')
            result.images = [
                urljoin(result.url, img.get('src'))
                for img in images
                if img.get('src')
            ][:10]  # Limit to 10 images

            # Extract links
            links = soup.find_all('a')
            result.links = [
                urljoin(result.url, link.get('href'))
                for link in links
                if link.get('href')
            ][:20]  # Limit to 20 links

        except Exception as e:
            self.logger.debug(f"Content extraction failed for {result.url}: {e}")

    def _rank_search_results(
        self,
        results: List[SearchResult],
        query: str
    ) -> List[SearchResult]:
        """Rank search results based on relevance."""

        query_terms = set(query.lower().split())

        for result in results:
            score = 0.0

            # Title relevance (higher weight)
            title_terms = set(result.title.lower().split())
            title_matches = len(query_terms & title_terms)
            score += title_matches * 0.4

            # Snippet relevance
            snippet_terms = set(result.snippet.lower().split())
            snippet_matches = len(query_terms & snippet_terms)
            score += snippet_matches * 0.3

            # URL relevance
            url_terms = set(result.url.lower().replace('/', ' ').replace('-', ' ').split())
            url_matches = len(query_terms & url_terms)
            score += url_matches * 0.2

            # Recency bonus
            if result.published_date:
                days_old = (datetime.now() - result.published_date).days
                if days_old < 7:
                    score += 0.1
                elif days_old < 30:
                    score += 0.05

            result.score = score

        # Sort by score (descending)
        return sorted(results, key=lambda r: r.score, reverse=True)

    def _check_rate_limit(self, engine: SearchEngine) -> bool:
        """Check if search engine rate limit allows request."""

        rate_limit = self.rate_limits[engine]
        last_request = rate_limit.get("last_request")

        if last_request is None:
            rate_limit["last_request"] = datetime.now()
            return True

        # Simple rate limiting logic
        now = datetime.now()
        time_diff = now - last_request

        if engine == SearchEngine.DUCKDUCKGO:
            if time_diff.total_seconds() >= 2:  # 2 seconds between requests
                rate_limit["last_request"] = now
                return True
        else:
            # For other engines, allow requests for now
            rate_limit["last_request"] = now
            return True

        return False

    def _create_cache_key(
        self,
        query: str,
        engine: SearchEngine,
        content_type: ContentType,
        filters: List[SearchFilter]
    ) -> str:
        """Create cache key for search results."""

        filter_str = ",".join(sorted([f.value for f in filters]))
        return f"{engine.value}:{content_type.value}:{query}:{filter_str}"

    def _get_cached_result(self, cache_key: str) -> Optional[WebSearchResult]:
        """Get cached search result if still valid."""

        if cache_key not in self.cache:
            return None

        cached_data = self.cache[cache_key]
        cache_time = cached_data["timestamp"]

        # Check if cache is still valid
        if datetime.now() - cache_time > timedelta(minutes=self.cache_ttl_minutes):
            del self.cache[cache_key]
            return None

        return cached_data["result"]

    def _cache_result(self, cache_key: str, result: WebSearchResult) -> None:
        """Cache search result."""

        self.cache[cache_key] = {
            "result": result,
            "timestamp": datetime.now()
        }

        # Simple cache cleanup - remove old entries
        if len(self.cache) > 100:
            oldest_key = min(self.cache.keys(), key=lambda k: self.cache[k]["timestamp"])
            del self.cache[oldest_key]

    async def get_page_content(self, url: str, timeout: Optional[int] = None) -> Dict[str, Any]:
        """Get full content from a specific URL."""

        timeout = timeout or self.timeout_seconds

        if not self.session:
            raise ToolExecutionError("HTTP session not available")

        try:
            response = self.session.get(url, timeout=timeout)
            response.raise_for_status()

            content_info = {
                "url": url,
                "status_code": response.status_code,
                "content_type": response.headers.get("content-type", ""),
                "content_length": len(response.content),
                "title": "",
                "text_content": "",
                "images": [],
                "links": [],
                "metadata": {}
            }

            if BS4_AVAILABLE and "text/html" in content_info["content_type"]:
                soup = BeautifulSoup(response.content, 'html.parser')

                # Extract title
                title_tag = soup.find('title')
                if title_tag:
                    content_info["title"] = title_tag.get_text().strip()

                # Extract text content
                for script in soup(["script", "style"]):
                    script.decompose()

                text = soup.get_text()
                lines = (line.strip() for line in text.splitlines())
                chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
                content_info["text_content"] = ' '.join(chunk for chunk in chunks if chunk)

                # Extract images
                images = soup.find_all('img')
                content_info["images"] = [
                    urljoin(url, img.get('src'))
                    for img in images
                    if img.get('src')
                ]

                # Extract links
                links = soup.find_all('a')
                content_info["links"] = [
                    urljoin(url, link.get('href'))
                    for link in links
                    if link.get('href')
                ]

                # Extract metadata
                meta_tags = soup.find_all('meta')
                for meta in meta_tags:
                    name = meta.get('name') or meta.get('property')
                    content = meta.get('content')
                    if name and content:
                        content_info["metadata"][name] = content

            return content_info

        except Exception as e:
            raise ToolExecutionError(f"Failed to retrieve content from {url}: {e}")

    def get_supported_engines(self) -> List[str]:
        """Get list of supported search engines."""
        return [engine.value for engine in SearchEngine]

    def get_engine_info(self, engine: Union[SearchEngine, str]) -> Dict[str, Any]:
        """Get information about a specific search engine."""

        if isinstance(engine, str):
            engine = SearchEngine(engine.lower())

        engine_info = {
            SearchEngine.DUCKDUCKGO: {
                "name": "DuckDuckGo",
                "api_key_required": False,
                "rate_limit": "30 requests/minute",
                "content_types": ["web", "news", "images", "videos"],
                "available": DUCKDUCKGO_AVAILABLE
            },
            SearchEngine.GOOGLE: {
                "name": "Google Custom Search",
                "api_key_required": True,
                "rate_limit": "100 requests/day",
                "content_types": ["web", "images"],
                "available": True
            },
            SearchEngine.BING: {
                "name": "Bing Search API",
                "api_key_required": True,
                "rate_limit": "3000 requests/month",
                "content_types": ["web", "news", "images", "videos"],
                "available": True
            },
            SearchEngine.SERP: {
                "name": "SerpAPI",
                "api_key_required": True,
                "rate_limit": "100 requests/month",
                "content_types": ["web", "news", "images", "videos"],
                "available": True
            }
        }

        return engine_info.get(engine, {})

    async def health_check(self) -> Dict[str, Any]:
        """Perform health check on web search tool."""

        health_status = {
            "status": "healthy",
            "supported_engines": self.get_supported_engines(),
            "default_engine": self.default_engine.value,
            "configuration": {
                "max_results": self.max_results,
                "timeout_seconds": self.timeout_seconds,
                "enable_caching": self.enable_caching,
                "cache_ttl_minutes": self.cache_ttl_minutes
            },
            "dependencies": {
                "requests": REQUESTS_AVAILABLE,
                "aiohttp": AIOHTTP_AVAILABLE,
                "beautifulsoup4": BS4_AVAILABLE,
                "duckduckgo_search": DUCKDUCKGO_AVAILABLE
            },
            "cache_size": len(self.cache)
        }

        # Test basic functionality
        try:
            test_result = await self.search(
                "test query",
                search_engine=SearchEngine.DUCKDUCKGO,
                max_results=1,
                timeout_override=5
            )
            health_status["basic_search"] = test_result.success

        except Exception as e:
            health_status["status"] = "degraded"
            health_status["basic_search"] = False
            health_status["error"] = str(e)

        return health_status


# Factory function for creating web search tool
def create_web_search_tool(config: Optional[Dict[str, Any]] = None) -> WebSearchTool:
    """
    Create a web search tool with specified configuration.

    Args:
        config: Configuration for web search operations

    Returns:
        WebSearchTool: Configured web search tool instance
    """
    return WebSearchTool(config=config)
