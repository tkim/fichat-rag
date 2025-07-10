"""Embedding models for vector generation."""

import os
from abc import ABC, abstractmethod
from typing import List, Union, Optional
import numpy as np
from sentence_transformers import SentenceTransformer
import torch
import asyncio
from concurrent.futures import ThreadPoolExecutor


class BaseEmbeddings(ABC):
    """Base class for all embedding implementations."""
    
    @abstractmethod
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of documents."""
        pass
    
    @abstractmethod
    def embed_query(self, text: str) -> List[float]:
        """Embed a single query."""
        pass
    
    @abstractmethod
    async def aembed_documents(self, texts: List[str]) -> List[List[float]]:
        """Async embed a list of documents."""
        pass
    
    @abstractmethod
    async def aembed_query(self, text: str) -> List[float]:
        """Async embed a single query."""
        pass
    
    @property
    @abstractmethod
    def embedding_dim(self) -> int:
        """Return the dimension of embeddings."""
        pass


class SentenceTransformerEmbeddings(BaseEmbeddings):
    """Sentence Transformer embedding implementation."""
    
    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        device: Optional[str] = None,
        batch_size: int = 32,
        normalize_embeddings: bool = True,
        **kwargs
    ):
        self.model_name = model_name
        self.batch_size = batch_size
        self.normalize_embeddings = normalize_embeddings
        
        # Set device
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        
        # Load model
        self.model = SentenceTransformer(model_name, device=self.device, **kwargs)
        
        # Thread pool for async operations
        self._executor = ThreadPoolExecutor(max_workers=1)
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of documents."""
        if not texts:
            return []
        
        embeddings = self.model.encode(
            texts,
            batch_size=self.batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=self.normalize_embeddings
        )
        
        return embeddings.tolist()
    
    def embed_query(self, text: str) -> List[float]:
        """Embed a single query."""
        embedding = self.model.encode(
            text,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=self.normalize_embeddings
        )
        
        return embedding.tolist()
    
    async def aembed_documents(self, texts: List[str]) -> List[List[float]]:
        """Async embed a list of documents."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self._executor, self.embed_documents, texts)
    
    async def aembed_query(self, text: str) -> List[float]:
        """Async embed a single query."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self._executor, self.embed_query, text)
    
    @property
    def embedding_dim(self) -> int:
        """Return the dimension of embeddings."""
        return self.model.get_sentence_embedding_dimension()
    
    def __del__(self):
        """Cleanup thread pool."""
        if hasattr(self, '_executor'):
            self._executor.shutdown(wait=False)


class OpenAIEmbeddings(BaseEmbeddings):
    """OpenAI embedding implementation."""
    
    def __init__(
        self,
        model: str = "text-embedding-ada-002",
        api_key: Optional[str] = None,
        base_url: str = "https://api.openai.com/v1",
        batch_size: int = 100,
        **kwargs
    ):
        self.model = model
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key is required")
        
        self.base_url = base_url.rstrip("/")
        self.batch_size = batch_size
        
        # Set embedding dimension based on model
        self._embedding_dims = {
            "text-embedding-ada-002": 1536,
            "text-embedding-3-small": 1536,
            "text-embedding-3-large": 3072,
        }
        
        import requests
        self.session = requests.Session()
        self.session.headers.update({
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        })
    
    def _get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Get embeddings from OpenAI API."""
        url = f"{self.base_url}/embeddings"
        
        all_embeddings = []
        
        # Process in batches
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]
            
            data = {
                "model": self.model,
                "input": batch
            }
            
            try:
                response = self.session.post(url, json=data)
                response.raise_for_status()
                
                result = response.json()
                embeddings = [item["embedding"] for item in result["data"]]
                all_embeddings.extend(embeddings)
                
            except Exception as e:
                raise RuntimeError(f"Failed to get embeddings from OpenAI: {e}")
        
        return all_embeddings
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of documents."""
        if not texts:
            return []
        return self._get_embeddings(texts)
    
    def embed_query(self, text: str) -> List[float]:
        """Embed a single query."""
        return self._get_embeddings([text])[0]
    
    async def aembed_documents(self, texts: List[str]) -> List[List[float]]:
        """Async embed a list of documents."""
        import aiohttp
        
        if not texts:
            return []
        
        url = f"{self.base_url}/embeddings"
        all_embeddings = []
        
        async with aiohttp.ClientSession() as session:
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            
            # Process in batches
            for i in range(0, len(texts), self.batch_size):
                batch = texts[i:i + self.batch_size]
                
                data = {
                    "model": self.model,
                    "input": batch
                }
                
                try:
                    async with session.post(url, json=data, headers=headers) as response:
                        response.raise_for_status()
                        result = await response.json()
                        embeddings = [item["embedding"] for item in result["data"]]
                        all_embeddings.extend(embeddings)
                        
                except Exception as e:
                    raise RuntimeError(f"Failed to async get embeddings from OpenAI: {e}")
        
        return all_embeddings
    
    async def aembed_query(self, text: str) -> List[float]:
        """Async embed a single query."""
        embeddings = await self.aembed_documents([text])
        return embeddings[0]
    
    @property
    def embedding_dim(self) -> int:
        """Return the dimension of embeddings."""
        return self._embedding_dims.get(self.model, 1536)