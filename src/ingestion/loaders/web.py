"""Web document loader using Firecrawl."""

import os
from typing import Dict, Any, List, Optional, Union
from pathlib import Path
import logging
from urllib.parse import urlparse
import time

logger = logging.getLogger(__name__)


class FirecrawlLoader:
    """Load web content using Firecrawl API."""
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        formats: List[str] = ["markdown", "html"],
        timeout: int = 30000,
        wait_for_selector: Optional[str] = None,
        screenshot: bool = False,
        only_main_content: bool = True,
        include_links: bool = False
    ):
        """
        Initialize Firecrawl loader.
        
        Args:
            api_key: Firecrawl API key (can also be set via FIRECRAWL_API_KEY env var)
            formats: List of formats to extract ["markdown", "html", "rawHtml", "screenshot"]
            timeout: Request timeout in milliseconds
            wait_for_selector: CSS selector to wait for before extraction
            screenshot: Whether to capture screenshots
            only_main_content: Extract only main content (remove headers, footers, etc.)
            include_links: Include links in the extracted content
        """
        self.api_key = api_key or os.getenv("FIRECRAWL_API_KEY")
        if not self.api_key:
            raise ValueError(
                "Firecrawl API key is required. "
                "Set it via FIRECRAWL_API_KEY environment variable or pass it to the constructor."
            )
        
        self.formats = formats
        self.timeout = timeout
        self.wait_for_selector = wait_for_selector
        self.screenshot = screenshot
        self.only_main_content = only_main_content
        self.include_links = include_links
        
        # Initialize Firecrawl
        try:
            from firecrawl import FirecrawlApp
            self.app = FirecrawlApp(api_key=self.api_key)
        except ImportError:
            raise ImportError(
                "Firecrawl package not found. "
                "Install it with: pip install firecrawl-py"
            )
    
    def load(self, url: str) -> Dict[str, Any]:
        """Load a single URL."""
        try:
            # Parse URL to extract domain for metadata
            parsed_url = urlparse(url)
            domain = parsed_url.netloc
            
            # Scrape the URL
            logger.info(f"Scraping URL: {url}")
            
            scrape_options = {
                "formats": self.formats,
                "timeout": self.timeout,
                "onlyMainContent": self.only_main_content,
                "includeLinks": self.include_links
            }
            
            if self.wait_for_selector:
                scrape_options["waitFor"] = self.wait_for_selector
            
            if self.screenshot:
                scrape_options["screenshot"] = True
                if "screenshot" not in self.formats:
                    self.formats.append("screenshot")
            
            result = self.app.scrape_url(url, **scrape_options)
            
            # Extract content based on format preference
            content = ""
            if "markdown" in self.formats and result.get("markdown"):
                content = result["markdown"]
            elif "html" in self.formats and result.get("html"):
                content = result["html"]
            elif "rawHtml" in self.formats and result.get("rawHtml"):
                content = result["rawHtml"]
            
            # Build metadata
            metadata = {
                "source": url,
                "domain": domain,
                "file_type": "web",
                "title": result.get("title", ""),
                "description": result.get("description", ""),
                "language": result.get("language", ""),
                "status_code": result.get("statusCode", 200),
                "scraped_at": time.time()
            }
            
            # Add screenshot if available
            if self.screenshot and result.get("screenshot"):
                metadata["screenshot_url"] = result["screenshot"]
            
            # Add links if extracted
            if self.include_links and result.get("links"):
                metadata["links"] = result["links"]
            
            # Add any custom metadata from Firecrawl
            if result.get("metadata"):
                metadata.update(result["metadata"])
            
            return {
                "content": content,
                "metadata": metadata
            }
            
        except Exception as e:
            logger.error(f"Failed to scrape URL {url}: {e}")
            raise
    
    def load_multiple(self, urls: List[str]) -> List[Dict[str, Any]]:
        """Load multiple URLs."""
        documents = []
        
        for url in urls:
            try:
                doc = self.load(url)
                documents.append(doc)
            except Exception as e:
                logger.error(f"Failed to load URL {url}: {e}")
        
        return documents
    
    def crawl(
        self,
        url: str,
        max_depth: Optional[int] = None,
        limit: int = 10,
        exclude_patterns: Optional[List[str]] = None,
        include_patterns: Optional[List[str]] = None,
        allow_backward_links: bool = False
    ) -> List[Dict[str, Any]]:
        """
        Crawl a website starting from the given URL.
        
        Args:
            url: Starting URL for the crawl
            max_depth: Maximum crawl depth
            limit: Maximum number of pages to crawl
            exclude_patterns: URL patterns to exclude
            include_patterns: URL patterns to include
            allow_backward_links: Whether to follow links to parent directories
            
        Returns:
            List of documents from crawled pages
        """
        try:
            logger.info(f"Starting crawl from: {url}")
            
            crawl_options = {
                "limit": limit,
                "scrapeOptions": {
                    "formats": self.formats,
                    "onlyMainContent": self.only_main_content,
                    "includeLinks": self.include_links
                }
            }
            
            if max_depth is not None:
                crawl_options["maxDepth"] = max_depth
            
            if exclude_patterns:
                crawl_options["excludePatterns"] = exclude_patterns
            
            if include_patterns:
                crawl_options["includePatterns"] = include_patterns
            
            if allow_backward_links:
                crawl_options["allowBackwardLinks"] = allow_backward_links
            
            # Start the crawl
            crawl_result = self.app.crawl_url(url, **crawl_options)
            
            documents = []
            
            # Process crawled pages
            if crawl_result and crawl_result.get("data"):
                for page_data in crawl_result["data"]:
                    # Extract content
                    content = ""
                    if "markdown" in self.formats and page_data.get("markdown"):
                        content = page_data["markdown"]
                    elif "html" in self.formats and page_data.get("html"):
                        content = page_data["html"]
                    
                    # Build metadata
                    page_url = page_data.get("url", url)
                    parsed_url = urlparse(page_url)
                    
                    metadata = {
                        "source": page_url,
                        "domain": parsed_url.netloc,
                        "file_type": "web",
                        "title": page_data.get("title", ""),
                        "description": page_data.get("description", ""),
                        "crawled_from": url,
                        "crawled_at": time.time()
                    }
                    
                    if page_data.get("metadata"):
                        metadata.update(page_data["metadata"])
                    
                    documents.append({
                        "content": content,
                        "metadata": metadata
                    })
            
            logger.info(f"Crawled {len(documents)} pages from {url}")
            return documents
            
        except Exception as e:
            logger.error(f"Failed to crawl website {url}: {e}")
            raise
    
    def map_website(self, url: str, limit: int = 5000) -> List[str]:
        """
        Map all URLs in a website.
        
        Args:
            url: Website URL to map
            limit: Maximum number of URLs to map
            
        Returns:
            List of discovered URLs
        """
        try:
            logger.info(f"Mapping website: {url}")
            
            result = self.app.map_url(url, limit=limit)
            
            if result and result.get("urls"):
                logger.info(f"Found {len(result['urls'])} URLs in {url}")
                return result["urls"]
            
            return []
            
        except Exception as e:
            logger.error(f"Failed to map website {url}: {e}")
            raise


class WebLoader:
    """Generic web loader that supports multiple backends."""
    
    def __init__(
        self,
        backend: str = "firecrawl",
        **kwargs
    ):
        """
        Initialize web loader.
        
        Args:
            backend: Backend to use ("firecrawl", "beautifulsoup", "selenium")
            **kwargs: Backend-specific arguments
        """
        self.backend = backend
        
        if backend == "firecrawl":
            self.loader = FirecrawlLoader(**kwargs)
        else:
            raise ValueError(f"Unsupported backend: {backend}")
    
    def load(self, url: str) -> Dict[str, Any]:
        """Load a single URL."""
        return self.loader.load(url)
    
    def load_multiple(self, urls: List[str]) -> List[Dict[str, Any]]:
        """Load multiple URLs."""
        return self.loader.load_multiple(urls)
    
    def crawl(self, url: str, **kwargs) -> List[Dict[str, Any]]:
        """Crawl a website."""
        if hasattr(self.loader, 'crawl'):
            return self.loader.crawl(url, **kwargs)
        else:
            raise NotImplementedError(f"Crawling not supported by {self.backend} backend")