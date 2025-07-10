"""Base classes for vector storage."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
import uuid


@dataclass
class Document:
    """Document class for storing text and metadata."""
    
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    embedding: Optional[List[float]] = None
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    
    def __post_init__(self):
        """Ensure metadata has created_at timestamp."""
        if "created_at" not in self.metadata:
            self.metadata["created_at"] = datetime.utcnow().isoformat()


@dataclass
class SearchResult:
    """Result from vector search."""
    
    document: Document
    score: float
    rank: Optional[int] = None


class BaseVectorStore(ABC):
    """Base class for all vector store implementations."""
    
    @abstractmethod
    def add_documents(
        self,
        documents: List[Document],
        embeddings: Optional[List[List[float]]] = None,
        **kwargs
    ) -> List[str]:
        """Add documents to the vector store."""
        pass
    
    @abstractmethod
    def search(
        self,
        query_embedding: List[float],
        k: int = 10,
        filter: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> List[SearchResult]:
        """Search for similar documents using vector similarity."""
        pass
    
    @abstractmethod
    def hybrid_search(
        self,
        query: str,
        query_embedding: List[float],
        k: int = 10,
        filter: Optional[Dict[str, Any]] = None,
        vector_weight: float = 0.7,
        **kwargs
    ) -> List[SearchResult]:
        """Perform hybrid search combining vector and keyword search."""
        pass
    
    @abstractmethod
    def delete(self, ids: List[str]) -> bool:
        """Delete documents by IDs."""
        pass
    
    @abstractmethod
    def get(self, ids: List[str]) -> List[Document]:
        """Get documents by IDs."""
        pass
    
    @abstractmethod
    def update(self, documents: List[Document]) -> bool:
        """Update existing documents."""
        pass
    
    @abstractmethod
    def count(self, filter: Optional[Dict[str, Any]] = None) -> int:
        """Count documents in the store."""
        pass
    
    @abstractmethod
    def clear(self) -> bool:
        """Clear all documents from the store."""
        pass
    
    @abstractmethod
    async def asearch(
        self,
        query_embedding: List[float],
        k: int = 10,
        filter: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> List[SearchResult]:
        """Async search for similar documents."""
        pass
    
    def similarity_search_with_score(
        self,
        query_embedding: List[float],
        k: int = 10,
        filter: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> List[Tuple[Document, float]]:
        """Search and return documents with similarity scores."""
        results = self.search(query_embedding, k=k, filter=filter, **kwargs)
        return [(r.document, r.score) for r in results]