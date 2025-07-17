"""LLM interfaces and implementations."""

import os
import json
import requests
from abc import ABC, abstractmethod
from typing import Optional, Iterator, List, Dict, Any, Union
import asyncio
import aiohttp
from dataclasses import dataclass
from datetime import datetime
import logging
from .llm_utils import LRUCache, ConnectionPool, PromptOptimizer

logger = logging.getLogger(__name__)


@dataclass
class LLMResponse:
    """Standard response from LLM."""
    content: str
    model: str
    usage: Optional[Dict[str, int]] = None
    metadata: Optional[Dict[str, Any]] = None


class BaseLLM(ABC):
    """Base class for all LLM implementations."""
    
    @abstractmethod
    def generate(self, prompt: str, **kwargs) -> LLMResponse:
        """Generate a response from the LLM."""
        pass
    
    @abstractmethod
    def stream_generate(self, prompt: str, **kwargs) -> Iterator[str]:
        """Stream generate a response from the LLM."""
        pass
    
    @abstractmethod
    async def agenerate(self, prompt: str, **kwargs) -> LLMResponse:
        """Async generate a response from the LLM."""
        pass
    
    def chat(self, messages: List[Dict[str, str]], **kwargs) -> LLMResponse:
        """Chat completion interface."""
        # Default implementation converts to single prompt
        prompt = "\n".join([f"{msg['role']}: {msg['content']}" for msg in messages])
        return self.generate(prompt, **kwargs)


class OllamaLLM(BaseLLM):
    """Enhanced Ollama LLM implementation with model management features."""
    
    # Model-specific context window sizes
    MODEL_CONTEXT_WINDOWS = {
        "llama2": 4096,
        "llama2:7b": 4096,
        "llama2:13b": 4096,
        "llama2:70b": 4096,
        "mistral": 8192,
        "mixtral": 32768,
        "codellama": 16384,
        "phi": 2048,
        "phi-2": 2048,
        "neural-chat": 4096,
        "starling-lm": 8192,
        "vicuna": 2048,
        "orca-mini": 2048,
        "deepseek-coder": 16384,
        "qwen": 8192,
    }
    
    # Model-specific optimal parameters
    MODEL_OPTIMAL_PARAMS = {
        "llama2": {"temperature": 0.7, "top_p": 0.9, "repeat_penalty": 1.1},
        "mistral": {"temperature": 0.7, "top_p": 0.95, "repeat_penalty": 1.15},
        "codellama": {"temperature": 0.1, "top_p": 0.95, "repeat_penalty": 1.0},
        "phi": {"temperature": 0.8, "top_p": 0.9, "repeat_penalty": 1.05},
    }
    
    def __init__(
        self,
        model: str = "llama2",
        base_url: str = "http://localhost:11434",
        temperature: float = 0.7,
        max_tokens: int = 2000,
        context_window: Optional[int] = None,
        enable_cache: bool = True,
        cache_ttl: int = 3600,
        enable_connection_pool: bool = True,
        **kwargs
    ):
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.temperature = temperature
        self.max_tokens = max_tokens
        
        # Set context window based on model or use provided value
        model_base = model.split(":")[0]
        self.context_window = context_window or self.MODEL_CONTEXT_WINDOWS.get(
            model, self.MODEL_CONTEXT_WINDOWS.get(model_base, 4096)
        )
        
        # Apply model-specific optimal parameters
        optimal_params = self.MODEL_OPTIMAL_PARAMS.get(
            model, self.MODEL_OPTIMAL_PARAMS.get(model_base, {})
        )
        self.default_params = {**optimal_params, **kwargs}
        
        # Initialize performance enhancements
        self.cache = LRUCache(ttl_seconds=cache_ttl) if enable_cache else None
        self.connection_pool = ConnectionPool() if enable_connection_pool else None
    
    def generate(self, prompt: str, **kwargs) -> LLMResponse:
        """Generate a response using Ollama with caching and optimizations."""
        # Check cache first
        if self.cache and not kwargs.get("no_cache", False):
            cache_key_params = {
                "model": self.model,
                "temperature": kwargs.get("temperature", self.temperature),
                "max_tokens": kwargs.get("max_tokens", self.max_tokens)
            }
            cached_response = self.cache.get(prompt, **cache_key_params)
            if cached_response:
                logger.debug(f"Cache hit for prompt: {prompt[:50]}...")
                return cached_response
        
        # Optimize prompt for model
        optimized_prompt = PromptOptimizer.optimize_prompt(
            prompt, self.model, kwargs.get("system_prompt")
        )
        
        # Ensure prompt fits in context window
        optimized_prompt = PromptOptimizer.truncate_to_context_window(
            optimized_prompt, self.context_window, self.max_tokens
        )
        
        url = f"{self.base_url}/api/generate"
        
        params = {
            "model": self.model,
            "prompt": optimized_prompt,
            "stream": False,
            "options": {
                "temperature": kwargs.get("temperature", self.temperature),
                "num_predict": kwargs.get("max_tokens", self.max_tokens),
                **self.default_params
            }
        }
        
        try:
            # Use connection pool if available
            if self.connection_pool:
                response = self.connection_pool.post(url, json=params)
            else:
                response = requests.post(url, json=params)
                
            response.raise_for_status()
            
            data = response.json()
            result = LLMResponse(
                content=data["response"],
                model=self.model,
                usage={
                    "prompt_tokens": data.get("prompt_eval_count", 0),
                    "completion_tokens": data.get("eval_count", 0),
                    "total_tokens": data.get("prompt_eval_count", 0) + data.get("eval_count", 0)
                },
                metadata={
                    "eval_duration": data.get("eval_duration"),
                    "total_duration": data.get("total_duration")
                }
            )
            
            # Cache the result
            if self.cache and not kwargs.get("no_cache", False):
                self.cache.set(prompt, result, **cache_key_params)
            
            return result
            
        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"Failed to generate response from Ollama: {e}")
    
    def stream_generate(self, prompt: str, **kwargs) -> Iterator[str]:
        """Stream generate a response using Ollama."""
        url = f"{self.base_url}/api/generate"
        
        params = {
            "model": self.model,
            "prompt": prompt,
            "stream": True,
            "options": {
                "temperature": kwargs.get("temperature", self.temperature),
                "num_predict": kwargs.get("max_tokens", self.max_tokens),
                **self.default_params
            }
        }
        
        try:
            with requests.post(url, json=params, stream=True) as response:
                response.raise_for_status()
                
                for line in response.iter_lines():
                    if line:
                        data = json.loads(line)
                        if "response" in data:
                            yield data["response"]
                        if data.get("done", False):
                            break
        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"Failed to stream response from Ollama: {e}")
    
    async def agenerate(self, prompt: str, **kwargs) -> LLMResponse:
        """Async generate a response using Ollama."""
        url = f"{self.base_url}/api/generate"
        
        params = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": kwargs.get("temperature", self.temperature),
                "num_predict": kwargs.get("max_tokens", self.max_tokens),
                **self.default_params
            }
        }
        
        async with aiohttp.ClientSession() as session:
            try:
                async with session.post(url, json=params) as response:
                    response.raise_for_status()
                    data = await response.json()
                    
                    return LLMResponse(
                        content=data["response"],
                        model=self.model,
                        usage={
                            "prompt_tokens": data.get("prompt_eval_count", 0),
                            "completion_tokens": data.get("eval_count", 0),
                            "total_tokens": data.get("prompt_eval_count", 0) + data.get("eval_count", 0)
                        },
                        metadata={
                            "eval_duration": data.get("eval_duration"),
                            "total_duration": data.get("total_duration")
                        }
                    )
            except aiohttp.ClientError as e:
                raise RuntimeError(f"Failed to async generate response from Ollama: {e}")
    
    def chat(self, messages: List[Dict[str, str]], **kwargs) -> LLMResponse:
        """Chat completion using Ollama."""
        url = f"{self.base_url}/api/chat"
        
        params = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": kwargs.get("temperature", self.temperature),
                "num_predict": kwargs.get("max_tokens", self.max_tokens),
                **self.default_params
            }
        }
        
        try:
            response = requests.post(url, json=params)
            response.raise_for_status()
            
            data = response.json()
            return LLMResponse(
                content=data["message"]["content"],
                model=self.model,
                usage={
                    "prompt_tokens": data.get("prompt_eval_count", 0),
                    "completion_tokens": data.get("eval_count", 0),
                    "total_tokens": data.get("prompt_eval_count", 0) + data.get("eval_count", 0)
                },
                metadata={
                    "eval_duration": data.get("eval_duration"),
                    "total_duration": data.get("total_duration")
                }
            )
        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"Failed to chat with Ollama: {e}")
    
    def list_models(self) -> List[Dict[str, Any]]:
        """List available Ollama models."""
        url = f"{self.base_url}/api/tags"
        
        try:
            if self.connection_pool:
                response = self.connection_pool.get(url)
            else:
                response = requests.get(url)
            response.raise_for_status()
            data = response.json()
            
            models = []
            for model in data.get("models", []):
                models.append({
                    "name": model.get("name"),
                    "size": model.get("size"),
                    "digest": model.get("digest"),
                    "modified_at": model.get("modified_at"),
                    "details": model.get("details", {})
                })
            
            return models
        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"Failed to list models: {e}")
    
    def pull_model(self, model_name: str, stream_progress: bool = True) -> Dict[str, Any]:
        """Pull a model from Ollama registry."""
        url = f"{self.base_url}/api/pull"
        
        params = {
            "name": model_name,
            "stream": stream_progress
        }
        
        try:
            if stream_progress:
                with requests.post(url, json=params, stream=True) as response:
                    response.raise_for_status()
                    
                    for line in response.iter_lines():
                        if line:
                            data = json.loads(line)
                            status = data.get("status", "")
                            
                            if "pulling" in status or "downloading" in status:
                                logger.info(f"Pulling {model_name}: {status}")
                            elif data.get("completed"):
                                logger.info(f"Download progress: {data.get('completed')}/{data.get('total')}")
                            
                            if status == "success":
                                return {"status": "success", "model": model_name}
            else:
                response = requests.post(url, json=params)
                response.raise_for_status()
                return response.json()
                
        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"Failed to pull model {model_name}: {e}")
    
    def model_info(self, model_name: Optional[str] = None) -> Dict[str, Any]:
        """Get detailed information about a model."""
        url = f"{self.base_url}/api/show"
        model_to_check = model_name or self.model
        
        params = {"name": model_to_check}
        
        try:
            response = requests.post(url, json=params)
            response.raise_for_status()
            data = response.json()
            
            return {
                "name": model_to_check,
                "model_info": data.get("modelinfo", {}),
                "parameters": data.get("parameters", ""),
                "template": data.get("template", ""),
                "details": data.get("details", {}),
                "license": data.get("license", "")
            }
        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"Failed to get model info for {model_to_check}: {e}")
    
    def delete_model(self, model_name: str) -> Dict[str, str]:
        """Delete a model from local storage."""
        url = f"{self.base_url}/api/delete"
        
        params = {"name": model_name}
        
        try:
            response = requests.delete(url, json=params)
            response.raise_for_status()
            return {"status": "success", "message": f"Model {model_name} deleted"}
        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"Failed to delete model {model_name}: {e}")
    
    def copy_model(self, source: str, destination: str) -> Dict[str, str]:
        """Copy a model to a new name."""
        url = f"{self.base_url}/api/copy"
        
        params = {
            "source": source,
            "destination": destination
        }
        
        try:
            response = requests.post(url, json=params)
            response.raise_for_status()
            return {"status": "success", "message": f"Model copied from {source} to {destination}"}
        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"Failed to copy model: {e}")
    
    def ensure_model_loaded(self, model_name: Optional[str] = None) -> bool:
        """Ensure a model is loaded and ready to use."""
        model_to_check = model_name or self.model
        
        try:
            # Check if model exists locally
            models = self.list_models()
            model_names = [m["name"] for m in models]
            
            if model_to_check not in model_names:
                logger.info(f"Model {model_to_check} not found locally. Pulling...")
                self.pull_model(model_to_check)
                return True
            
            # Try a simple generation to ensure model is loaded
            test_response = requests.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": model_to_check,
                    "prompt": "test",
                    "stream": False,
                    "options": {"num_predict": 1}
                }
            )
            
            return test_response.status_code == 200
            
        except Exception as e:
            logger.error(f"Failed to ensure model {model_to_check} is loaded: {e}")
            return False
    
    def get_context_window(self, model_name: Optional[str] = None) -> int:
        """Get the context window size for a model."""
        model_to_check = model_name or self.model
        model_base = model_to_check.split(":")[0]
        
        # First check our known models
        if model_to_check in self.MODEL_CONTEXT_WINDOWS:
            return self.MODEL_CONTEXT_WINDOWS[model_to_check]
        elif model_base in self.MODEL_CONTEXT_WINDOWS:
            return self.MODEL_CONTEXT_WINDOWS[model_base]
        
        # Try to get from model info
        try:
            info = self.model_info(model_to_check)
            # Parse model info for context window information
            # This is model-specific and may need adjustment
            return self.context_window
        except:
            return self.context_window


class OpenAILLM(BaseLLM):
    """OpenAI LLM implementation."""
    
    def __init__(
        self,
        model: str = "gpt-3.5-turbo",
        api_key: Optional[str] = None,
        base_url: str = "https://api.openai.com/v1",
        temperature: float = 0.7,
        max_tokens: int = 2000,
        **kwargs
    ):
        self.model = model
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key is required")
        
        self.base_url = base_url.rstrip("/")
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.default_params = kwargs
        
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
    
    def generate(self, prompt: str, **kwargs) -> LLMResponse:
        """Generate a response using OpenAI."""
        messages = [{"role": "user", "content": prompt}]
        return self.chat(messages, **kwargs)
    
    def stream_generate(self, prompt: str, **kwargs) -> Iterator[str]:
        """Stream generate a response using OpenAI."""
        url = f"{self.base_url}/chat/completions"
        messages = [{"role": "user", "content": prompt}]
        
        params = {
            "model": self.model,
            "messages": messages,
            "temperature": kwargs.get("temperature", self.temperature),
            "max_tokens": kwargs.get("max_tokens", self.max_tokens),
            "stream": True,
            **self.default_params
        }
        
        try:
            with requests.post(url, json=params, headers=self.headers, stream=True) as response:
                response.raise_for_status()
                
                for line in response.iter_lines():
                    if line:
                        line = line.decode('utf-8')
                        if line.startswith("data: "):
                            if line.strip() == "data: [DONE]":
                                break
                            try:
                                data = json.loads(line[6:])
                                if "choices" in data and data["choices"]:
                                    delta = data["choices"][0].get("delta", {})
                                    if "content" in delta:
                                        yield delta["content"]
                            except json.JSONDecodeError:
                                continue
        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"Failed to stream response from OpenAI: {e}")
    
    async def agenerate(self, prompt: str, **kwargs) -> LLMResponse:
        """Async generate a response using OpenAI."""
        messages = [{"role": "user", "content": prompt}]
        url = f"{self.base_url}/chat/completions"
        
        params = {
            "model": self.model,
            "messages": messages,
            "temperature": kwargs.get("temperature", self.temperature),
            "max_tokens": kwargs.get("max_tokens", self.max_tokens),
            **self.default_params
        }
        
        async with aiohttp.ClientSession() as session:
            try:
                async with session.post(url, json=params, headers=self.headers) as response:
                    response.raise_for_status()
                    data = await response.json()
                    
                    return LLMResponse(
                        content=data["choices"][0]["message"]["content"],
                        model=data["model"],
                        usage=data.get("usage", {})
                    )
            except aiohttp.ClientError as e:
                raise RuntimeError(f"Failed to async generate response from OpenAI: {e}")
    
    def chat(self, messages: List[Dict[str, str]], **kwargs) -> LLMResponse:
        """Chat completion using OpenAI."""
        url = f"{self.base_url}/chat/completions"
        
        params = {
            "model": self.model,
            "messages": messages,
            "temperature": kwargs.get("temperature", self.temperature),
            "max_tokens": kwargs.get("max_tokens", self.max_tokens),
            **self.default_params
        }
        
        try:
            response = requests.post(url, json=params, headers=self.headers)
            response.raise_for_status()
            
            data = response.json()
            return LLMResponse(
                content=data["choices"][0]["message"]["content"],
                model=data["model"],
                usage=data.get("usage", {})
            )
        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"Failed to chat with OpenAI: {e}")