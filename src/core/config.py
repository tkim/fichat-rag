"""Configuration management for fiChat RAG."""

import os
from typing import Optional, Dict, Any
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Config:
    """Main configuration class for fiChat RAG."""
    
    # LLM settings
    llm_provider: str = "ollama"
    llm_model: str = "llama2"
    llm_temperature: float = 0.7
    llm_max_tokens: int = 2000
    llm_base_url: Optional[str] = None
    
    # Embedding settings
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_device: str = "cpu"
    embedding_batch_size: int = 32
    
    # Vector store settings
    vector_store_type: str = "postgres"
    vector_store_config: Dict[str, Any] = field(default_factory=dict)
    
    # Retrieval settings
    retrieval_top_k: int = 10
    retrieval_search_type: str = "hybrid"  # "vector", "keyword", "hybrid"
    retrieval_rerank: bool = True
    retrieval_rerank_top_k: int = 5
    
    # Chunking settings
    chunk_size: int = 500
    chunk_overlap: int = 50
    chunk_strategy: str = "semantic"  # "semantic", "fixed", "sentence"
    
    # Generation settings
    generation_prompt_template: Optional[str] = None
    generation_system_prompt: Optional[str] = None
    
    # Other settings
    cache_enabled: bool = True
    cache_ttl: int = 3600  # seconds
    log_level: str = "INFO"
    data_dir: Path = Path("./data")
    
    def __post_init__(self):
        """Post-initialization processing."""
        # Create data directory if it doesn't exist
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # Set default vector store config if not provided
        if not self.vector_store_config:
            if self.vector_store_type == "postgres":
                self.vector_store_config = {
                    "host": os.getenv("POSTGRES_HOST", "localhost"),
                    "port": int(os.getenv("POSTGRES_PORT", "5432")),
                    "database": os.getenv("POSTGRES_DB", "fichat_rag"),
                    "user": os.getenv("POSTGRES_USER", "rag_user"),
                    "password": os.getenv("POSTGRES_PASSWORD", "rag_pass"),
                }
            elif self.vector_store_type == "chromadb":
                self.vector_store_config = {
                    "persist_directory": str(self.data_dir / "chromadb"),
                }
        
        # Set default LLM base URLs
        if not self.llm_base_url:
            if self.llm_provider == "ollama":
                self.llm_base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
            elif self.llm_provider == "openai":
                self.llm_base_url = "https://api.openai.com/v1"
    
    @classmethod
    def from_env(cls) -> "Config":
        """Create config from environment variables."""
        return cls(
            llm_provider=os.getenv("LLM_PROVIDER", "ollama"),
            llm_model=os.getenv("LLM_MODEL", "llama2"),
            llm_temperature=float(os.getenv("LLM_TEMPERATURE", "0.7")),
            llm_max_tokens=int(os.getenv("LLM_MAX_TOKENS", "2000")),
            embedding_model=os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"),
            embedding_device=os.getenv("EMBEDDING_DEVICE", "cpu"),
            vector_store_type=os.getenv("VECTOR_STORE_TYPE", "postgres"),
            retrieval_top_k=int(os.getenv("RETRIEVAL_TOP_K", "10")),
            retrieval_search_type=os.getenv("RETRIEVAL_SEARCH_TYPE", "hybrid"),
            retrieval_rerank=os.getenv("RETRIEVAL_RERANK", "true").lower() == "true",
            chunk_size=int(os.getenv("CHUNK_SIZE", "500")),
            chunk_overlap=int(os.getenv("CHUNK_OVERLAP", "50")),
            cache_enabled=os.getenv("CACHE_ENABLED", "true").lower() == "true",
            log_level=os.getenv("LOG_LEVEL", "INFO"),
        )