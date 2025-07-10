"""fiChat RAG Framework - A production-ready RAG solution."""

from .core import RAG, Config
from .core.llm import BaseLLM, OllamaLLM, OpenAILLM
from .core.embeddings import BaseEmbeddings, SentenceTransformerEmbeddings

__version__ = "0.1.0"
__all__ = [
    "RAG",
    "Config", 
    "BaseLLM",
    "OllamaLLM",
    "OpenAILLM",
    "BaseEmbeddings",
    "SentenceTransformerEmbeddings",
]