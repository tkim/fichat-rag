"""Utilities for LLM performance enhancements."""

import hashlib
import time
from typing import Dict, Any, Optional, Tuple
from collections import OrderedDict
from threading import Lock
import requests
from urllib3.util.retry import Retry
from requests.adapters import HTTPAdapter


class LRUCache:
    """Simple LRU cache implementation for response caching."""
    
    def __init__(self, max_size: int = 100, ttl_seconds: int = 3600):
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self.cache: OrderedDict[str, Tuple[Any, float]] = OrderedDict()
        self.lock = Lock()
    
    def _make_key(self, prompt: str, **kwargs) -> str:
        """Create a cache key from prompt and parameters."""
        key_data = {
            "prompt": prompt,
            **kwargs
        }
        key_str = str(sorted(key_data.items()))
        return hashlib.md5(key_str.encode()).hexdigest()
    
    def get(self, prompt: str, **kwargs) -> Optional[Any]:
        """Get a cached response if available and not expired."""
        key = self._make_key(prompt, **kwargs)
        
        with self.lock:
            if key in self.cache:
                value, timestamp = self.cache[key]
                
                # Check if expired
                if time.time() - timestamp > self.ttl_seconds:
                    del self.cache[key]
                    return None
                
                # Move to end (most recently used)
                self.cache.move_to_end(key)
                return value
        
        return None
    
    def set(self, prompt: str, value: Any, **kwargs):
        """Cache a response."""
        key = self._make_key(prompt, **kwargs)
        
        with self.lock:
            # Remove oldest item if at capacity
            if len(self.cache) >= self.max_size:
                self.cache.popitem(last=False)
            
            self.cache[key] = (value, time.time())
    
    def clear(self):
        """Clear the cache."""
        with self.lock:
            self.cache.clear()


class ConnectionPool:
    """HTTP connection pool with retry logic."""
    
    def __init__(
        self,
        pool_connections: int = 10,
        pool_maxsize: int = 10,
        max_retries: int = 3,
        backoff_factor: float = 0.3
    ):
        self.session = requests.Session()
        
        # Configure retry strategy
        retry_strategy = Retry(
            total=max_retries,
            backoff_factor=backoff_factor,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["HEAD", "GET", "PUT", "DELETE", "OPTIONS", "TRACE", "POST"]
        )
        
        # Configure adapter with connection pooling
        adapter = HTTPAdapter(
            pool_connections=pool_connections,
            pool_maxsize=pool_maxsize,
            max_retries=retry_strategy
        )
        
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)
    
    def get(self, url: str, **kwargs) -> requests.Response:
        """Make a GET request using the connection pool."""
        return self.session.get(url, **kwargs)
    
    def post(self, url: str, **kwargs) -> requests.Response:
        """Make a POST request using the connection pool."""
        return self.session.post(url, **kwargs)
    
    def delete(self, url: str, **kwargs) -> requests.Response:
        """Make a DELETE request using the connection pool."""
        return self.session.delete(url, **kwargs)
    
    def close(self):
        """Close the session."""
        self.session.close()


class PromptOptimizer:
    """Optimize prompts for specific models."""
    
    # Model-specific prompt templates
    PROMPT_TEMPLATES = {
        "llama2": """<s>[INST] <<SYS>>
{system_prompt}
<</SYS>>

{user_prompt} [/INST]""",
        
        "mistral": """{user_prompt}""",
        
        "codellama": """<PRE> {user_prompt} </PRE>""",
        
        "phi": """Instruct: {user_prompt}
Output:""",
    }
    
    @classmethod
    def optimize_prompt(
        cls,
        prompt: str,
        model: str,
        system_prompt: Optional[str] = None
    ) -> str:
        """Optimize prompt format for specific model."""
        model_base = model.split(":")[0]
        
        if model_base in cls.PROMPT_TEMPLATES:
            template = cls.PROMPT_TEMPLATES[model_base]
            
            if "{system_prompt}" in template and system_prompt:
                return template.format(
                    system_prompt=system_prompt,
                    user_prompt=prompt
                )
            else:
                return template.format(user_prompt=prompt)
        
        return prompt
    
    @classmethod
    def truncate_to_context_window(
        cls,
        prompt: str,
        context_window: int,
        reserve_tokens: int = 500
    ) -> str:
        """Truncate prompt to fit within context window."""
        # Simple approximation: 1 token ≈ 4 characters
        max_chars = (context_window - reserve_tokens) * 4
        
        if len(prompt) > max_chars:
            # Truncate from the middle to preserve context
            half_size = max_chars // 2
            return prompt[:half_size] + "\n...[truncated]...\n" + prompt[-half_size:]
        
        return prompt