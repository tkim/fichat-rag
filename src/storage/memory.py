"""In-memory vector store for testing and development."""

import numpy as np
from typing import List, Dict, Any, Optional
import asyncio
from collections import defaultdict
import re

from .base import BaseVectorStore, Document, SearchResult


class InMemoryVectorStore(BaseVectorStore):
    """Simple in-memory vector store implementation."""
    
    def __init__(self, **kwargs):
        self.documents: Dict[str, Document] = {}
        self.embeddings: Dict[str, np.ndarray] = {}
        self.metadata_index: Dict[str, Dict[str, List[str]]] = defaultdict(lambda: defaultdict(list))
    
    def add_documents(
        self,
        documents: List[Document],
        embeddings: Optional[List[List[float]]] = None,
        **kwargs
    ) -> List[str]:
        """Add documents to the vector store."""
        if not documents:
            return []
        
        if embeddings and len(embeddings) != len(documents):
            raise ValueError("Number of embeddings must match number of documents")
        
        ids = []
        for i, doc in enumerate(documents):
            embedding = embeddings[i] if embeddings else doc.embedding
            if not embedding:
                raise ValueError(f"No embedding provided for document {i}")
            
            # Store document and embedding
            self.documents[doc.id] = doc
            self.embeddings[doc.id] = np.array(embedding)
            
            # Update metadata index
            for key, value in doc.metadata.items():
                self.metadata_index[key][str(value)].append(doc.id)
            
            ids.append(doc.id)
        
        return ids
    
    def search(
        self,
        query_embedding: List[float],
        k: int = 10,
        filter: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> List[SearchResult]:
        """Search for similar documents using vector similarity."""
        query_vec = np.array(query_embedding)
        
        # Get candidate IDs based on filter
        candidate_ids = self._get_filtered_ids(filter)
        
        # Calculate similarities
        scores = []
        for doc_id in candidate_ids:
            if doc_id in self.embeddings:
                # Cosine similarity
                doc_vec = self.embeddings[doc_id]
                similarity = np.dot(query_vec, doc_vec) / (np.linalg.norm(query_vec) * np.linalg.norm(doc_vec))
                scores.append((doc_id, float(similarity)))
        
        # Sort by score and get top k
        scores.sort(key=lambda x: x[1], reverse=True)
        scores = scores[:k]
        
        # Create results
        results = []
        for i, (doc_id, score) in enumerate(scores):
            results.append(SearchResult(
                document=self.documents[doc_id],
                score=score,
                rank=i + 1
            ))
        
        return results
    
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
        # Get vector search results
        vector_results = self.search(query_embedding, k=k*2, filter=filter)
        vector_scores = {r.document.id: r.score for r in vector_results}
        
        # Get keyword search results
        keyword_results = self._keyword_search(query, k=k*2, filter=filter)
        keyword_scores = {r.document.id: r.score for r in keyword_results}
        
        # Combine scores
        all_ids = set(vector_scores.keys()) | set(keyword_scores.keys())
        combined_scores = []
        
        for doc_id in all_ids:
            vector_score = vector_scores.get(doc_id, 0.0)
            keyword_score = keyword_scores.get(doc_id, 0.0)
            combined_score = (vector_score * vector_weight) + (keyword_score * (1 - vector_weight))
            combined_scores.append((doc_id, combined_score))
        
        # Sort and get top k
        combined_scores.sort(key=lambda x: x[1], reverse=True)
        combined_scores = combined_scores[:k]
        
        # Create results
        results = []
        for i, (doc_id, score) in enumerate(combined_scores):
            results.append(SearchResult(
                document=self.documents[doc_id],
                score=score,
                rank=i + 1
            ))
        
        return results
    
    def delete(self, ids: List[str]) -> bool:
        """Delete documents by IDs."""
        if not ids:
            return True
        
        deleted = False
        for doc_id in ids:
            if doc_id in self.documents:
                # Remove from metadata index
                doc = self.documents[doc_id]
                for key, value in doc.metadata.items():
                    if str(value) in self.metadata_index[key]:
                        self.metadata_index[key][str(value)].remove(doc_id)
                
                # Remove document and embedding
                del self.documents[doc_id]
                if doc_id in self.embeddings:
                    del self.embeddings[doc_id]
                
                deleted = True
        
        return deleted
    
    def get(self, ids: List[str]) -> List[Document]:
        """Get documents by IDs."""
        return [self.documents[doc_id] for doc_id in ids if doc_id in self.documents]
    
    def update(self, documents: List[Document]) -> bool:
        """Update existing documents."""
        if not documents:
            return True
        
        updated = False
        for doc in documents:
            if doc.id in self.documents:
                # Update metadata index
                old_doc = self.documents[doc.id]
                for key, value in old_doc.metadata.items():
                    if str(value) in self.metadata_index[key]:
                        self.metadata_index[key][str(value)].remove(doc.id)
                
                for key, value in doc.metadata.items():
                    self.metadata_index[key][str(value)].append(doc.id)
                
                # Update document
                self.documents[doc.id] = doc
                updated = True
        
        return updated
    
    def count(self, filter: Optional[Dict[str, Any]] = None) -> int:
        """Count documents in the store."""
        if not filter:
            return len(self.documents)
        
        return len(self._get_filtered_ids(filter))
    
    def clear(self) -> bool:
        """Clear all documents from the store."""
        self.documents.clear()
        self.embeddings.clear()
        self.metadata_index.clear()
        return True
    
    async def asearch(
        self,
        query_embedding: List[float],
        k: int = 10,
        filter: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> List[SearchResult]:
        """Async search for similar documents."""
        # For in-memory store, just wrap sync search
        return self.search(query_embedding, k=k, filter=filter, **kwargs)
    
    def _get_filtered_ids(self, filter: Optional[Dict[str, Any]]) -> List[str]:
        """Get document IDs that match the filter."""
        if not filter:
            return list(self.documents.keys())
        
        # Start with all documents
        candidate_ids = set(self.documents.keys())
        
        # Apply filters
        for key, value in filter.items():
            matching_ids = set(self.metadata_index[key].get(str(value), []))
            candidate_ids &= matching_ids
        
        return list(candidate_ids)
    
    def _keyword_search(
        self,
        query: str,
        k: int = 10,
        filter: Optional[Dict[str, Any]] = None
    ) -> List[SearchResult]:
        """Simple keyword search implementation."""
        # Get candidate IDs
        candidate_ids = self._get_filtered_ids(filter)
        
        # Tokenize query
        query_tokens = set(re.findall(r'\w+', query.lower()))
        
        # Score documents
        scores = []
        for doc_id in candidate_ids:
            doc = self.documents[doc_id]
            doc_tokens = set(re.findall(r'\w+', doc.content.lower()))
            
            # Simple Jaccard similarity
            intersection = len(query_tokens & doc_tokens)
            union = len(query_tokens | doc_tokens)
            score = intersection / union if union > 0 else 0
            
            if score > 0:
                scores.append((doc_id, score))
        
        # Sort and get top k
        scores.sort(key=lambda x: x[1], reverse=True)
        scores = scores[:k]
        
        # Create results
        results = []
        for i, (doc_id, score) in enumerate(scores):
            results.append(SearchResult(
                document=self.documents[doc_id],
                score=score,
                rank=i + 1
            ))
        
        return results