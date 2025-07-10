"""LLM interfaces and implementations."""

import os
import json
import requests
from abc import ABC, abstractmethod
from typing import Optional, Iterator, List, Dict, Any
import asyncio
import aiohttp
from dataclasses import dataclass


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
    """Ollama LLM implementation."""
    
    def __init__(
        self,
        model: str = "llama2",
        base_url: str = "http://localhost:11434",
        temperature: float = 0.7,
        max_tokens: int = 2000,
        **kwargs
    ):
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.default_params = kwargs
    
    def generate(self, prompt: str, **kwargs) -> LLMResponse:
        """Generate a response using Ollama."""
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
        
        try:
            response = requests.post(url, json=params)
            response.raise_for_status()
            
            data = response.json()
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