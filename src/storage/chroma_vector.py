"""ChromaDB vector store implementation for local persistence."""

import os
import uuid
from typing import List, Dict, Any, Optional
import logging
from pathlib import Path

from .base import BaseVectorStore, Document

logger = logging.getLogger(__name__)

# Import ChromaDB with error handling
try:
    import chromadb
    from chromadb.config import Settings
    from chromadb.utils import embedding_functions
    CHROMADB_AVAILABLE = True
except ImportError:
    CHROMADB_AVAILABLE = False
    logger.warning(
        "ChromaDB not installed. Install with: pip install chromadb"
    )


class ChromaVectorStore(BaseVectorStore):
    """ChromaDB vector store with local persistence.
    
    ChromaDB is a purpose-built embedding database that provides
    excellent performance and features for RAG applications.
    """
    
    def __init__(
        self,
        persist_directory: str = "./chroma_db",
        collection_name: str = "fichat",
        embedding_dim: int = 384,
        distance_metric: str = "cosine"
    ):
        """Initialize ChromaDB vector store.
        
        Args:
            persist_directory: Directory to persist the database
            collection_name: Name of the collection
            embedding_dim: Dimension of embeddings
            distance_metric: Distance metric (cosine, l2, ip)
        """
        if not CHROMADB_AVAILABLE:
            raise ImportError(
                "ChromaDB is not installed. Install with: pip install chromadb"
            )
        
        self.persist_directory = persist_directory
        self.collection_name = collection_name
        self.embedding_dim = embedding_dim
        self.distance_metric = distance_metric
        
        # Ensure directory exists
        Path(persist_directory).mkdir(parents=True, exist_ok=True)
        
        # Initialize ChromaDB client
        self._init_client()
    
    def _init_client(self):
        """Initialize ChromaDB client and collection."""
        # Configure ChromaDB for local persistence
        settings = Settings(
            chroma_db_impl="duckdb+parquet",
            persist_directory=self.persist_directory,
            anonymized_telemetry=False
        )
        
        # Create persistent client
        self.client = chromadb.PersistentClient(
            path=self.persist_directory,
            settings=settings
        )
        
        # Get or create collection
        try:
            self.collection = self.client.get_collection(
                name=self.collection_name
            )
            logger.info(f"Loaded existing collection: {self.collection_name}")
        except:
            # Collection doesn't exist, create it
            metadata = {
                "embedding_dim": self.embedding_dim,
                "distance_metric": self.distance_metric
            }
            
            # Map distance metric to ChromaDB format
            distance_map = {
                "cosine": "cosine",
                "l2": "l2",
                "ip": "ip",  # inner product
                "euclidean": "l2"
            }
            
            self.collection = self.client.create_collection(
                name=self.collection_name,
                metadata=metadata,
                embedding_function=None,  # We provide embeddings
                metric=distance_map.get(self.distance_metric, "cosine")
            )
            logger.info(f"Created new collection: {self.collection_name}")
    
    def add(
        self,
        texts: List[str],
        embeddings: List[List[float]],
        metadatas: Optional[List[Dict[str, Any]]] = None,
        ids: Optional[List[str]] = None
    ) -> List[str]:
        """Add documents to the store."""
        if not texts:
            return []
        
        # Generate IDs if not provided
        if ids is None:
            ids = [str(uuid.uuid4()) for _ in texts]
        
        # Ensure metadatas list
        if metadatas is None:
            metadatas = [{}] * len(texts)
        
        # ChromaDB expects specific format
        try:
            self.collection.add(
                ids=ids,
                embeddings=embeddings,
                documents=texts,
                metadatas=metadatas
            )
            
            logger.info(f"Added {len(texts)} documents to ChromaDB")
            return ids
            
        except Exception as e:
            logger.error(f"Failed to add documents to ChromaDB: {e}")
            raise
    
    def search(
        self,
        query_embedding: List[float],
        k: int = 5,
        filter: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """Search for similar documents."""
        try:
            # Build where clause from filter
            where = self._build_where_clause(filter) if filter else None
            
            # Query the collection
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=k,
                where=where,
                include=["documents", "metadatas", "distances"]
            )
            
            # Format results
            formatted_results = []
            
            if results["ids"] and results["ids"][0]:
                for i in range(len(results["ids"][0])):
                    # Convert distance to similarity score
                    distance = results["distances"][0][i]
                    
                    if self.distance_metric == "cosine":
                        # Cosine distance to similarity
                        score = 1 - distance
                    elif self.distance_metric in ["l2", "euclidean"]:
                        # L2 distance to similarity (inverse)
                        score = 1 / (1 + distance)
                    else:  # inner product
                        score = distance
                    
                    formatted_results.append({
                        "id": results["ids"][0][i],
                        "content": results["documents"][0][i],
                        "metadata": results["metadatas"][0][i] or {},
                        "score": float(score)
                    })
            
            return formatted_results
            
        except Exception as e:
            logger.error(f"Search failed: {e}")
            raise
    
    def hybrid_search(
        self,
        query_text: str,
        query_embedding: List[float],
        k: int = 5,
        alpha: float = 0.5,
        filter: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """Hybrid search combining vector and keyword search.
        
        ChromaDB doesn't have native keyword search, so we'll use
        vector search with text filtering as an approximation.
        """
        # Get vector search results
        vector_results = self.search(query_embedding, k * 2, filter)
        
        # Filter results by keyword relevance
        query_terms = set(query_text.lower().split())
        
        scored_results = []
        for result in vector_results:
            content_lower = result["content"].lower()
            
            # Calculate keyword score
            keyword_score = 0
            for term in query_terms:
                if term in content_lower:
                    # Count occurrences
                    keyword_score += content_lower.count(term) / len(content_lower.split())
            
            # Normalize keyword score
            if query_terms:
                keyword_score = keyword_score / len(query_terms)
            
            # Combine scores
            combined_score = (
                alpha * result["score"] + 
                (1 - alpha) * keyword_score
            )
            
            result["score"] = combined_score
            scored_results.append(result)
        
        # Sort and return top k
        scored_results.sort(key=lambda x: x["score"], reverse=True)
        return scored_results[:k]
    
    def get(self, ids: List[str]) -> List[Optional[Document]]:
        """Get documents by IDs."""
        try:
            results = self.collection.get(
                ids=ids,
                include=["documents", "metadatas"]
            )
            
            # Create document mapping
            doc_map = {}
            if results["ids"]:
                for i, doc_id in enumerate(results["ids"]):
                    doc_map[doc_id] = Document(
                        id=doc_id,
                        content=results["documents"][i],
                        metadata=results["metadatas"][i] or {}
                    )
            
            # Return in order of requested IDs
            return [doc_map.get(doc_id) for doc_id in ids]
            
        except Exception as e:
            logger.error(f"Failed to get documents: {e}")
            raise
    
    def update(
        self,
        ids: List[str],
        texts: Optional[List[str]] = None,
        embeddings: Optional[List[List[float]]] = None,
        metadatas: Optional[List[Dict[str, Any]]] = None
    ):
        """Update existing documents."""
        try:
            update_params = {"ids": ids}
            
            if texts is not None:
                update_params["documents"] = texts
            
            if embeddings is not None:
                update_params["embeddings"] = embeddings
            
            if metadatas is not None:
                update_params["metadatas"] = metadatas
            
            self.collection.update(**update_params)
            logger.info(f"Updated {len(ids)} documents")
            
        except Exception as e:
            logger.error(f"Failed to update documents: {e}")
            raise
    
    def delete(self, ids: List[str]):
        """Delete documents by IDs."""
        try:
            self.collection.delete(ids=ids)
            logger.info(f"Deleted {len(ids)} documents")
            
        except Exception as e:
            logger.error(f"Failed to delete documents: {e}")
            raise
    
    def count(self, filter: Optional[Dict[str, Any]] = None) -> int:
        """Count documents in the store."""
        try:
            if filter:
                # ChromaDB doesn't have a direct count with filter
                # We'll do a query and count results
                where = self._build_where_clause(filter)
                
                # Query with large limit to get all matching docs
                results = self.collection.query(
                    query_embeddings=[[0.0] * self.embedding_dim],
                    n_results=1000000,  # Large number
                    where=where,
                    include=[]  # Don't include data, just IDs
                )
                
                return len(results["ids"][0]) if results["ids"] else 0
            else:
                # Get total count
                return self.collection.count()
                
        except Exception as e:
            logger.error(f"Failed to count documents: {e}")
            raise
    
    def clear(self):
        """Clear all documents from the store."""
        try:
            # Delete the collection and recreate it
            self.client.delete_collection(name=self.collection_name)
            self._init_client()
            logger.info("Cleared all documents from ChromaDB")
            
        except Exception as e:
            logger.error(f"Failed to clear store: {e}")
            raise
    
    def _build_where_clause(self, filter: Dict[str, Any]) -> Dict[str, Any]:
        """Build ChromaDB where clause from filter dict.
        
        ChromaDB supports various operators:
        - $eq, $ne, $gt, $gte, $lt, $lte
        - $in, $nin
        - $and, $or
        """
        where = {}
        
        for key, value in filter.items():
            if isinstance(value, dict):
                # Already in operator format
                where[key] = value
            elif isinstance(value, list):
                # Use $in operator for lists
                where[key] = {"$in": value}
            else:
                # Simple equality
                where[key] = {"$eq": value}
        
        return where
    
    def persist(self):
        """Persist the database to disk."""
        # ChromaDB with PersistentClient automatically persists
        logger.info("ChromaDB automatically persists changes")
    
    def get_collection_info(self) -> Dict[str, Any]:
        """Get information about the collection."""
        try:
            return {
                "name": self.collection_name,
                "count": self.collection.count(),
                "metadata": self.collection.metadata,
                "persist_directory": self.persist_directory
            }
        except Exception as e:
            logger.error(f"Failed to get collection info: {e}")
            return {}
    
    def optimize(self):
        """Optimize the collection for better performance."""
        # ChromaDB handles optimization internally
        logger.info("ChromaDB handles optimization automatically")
    
    def __del__(self):
        """Cleanup on deletion."""
        # ChromaDB client handles cleanup
        pass