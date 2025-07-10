"""Retrieval implementations for RAG."""

from typing import List, Dict, Any, Optional, Tuple
import asyncio
from abc import ABC, abstractmethod
import logging

from ..storage.base import BaseVectorStore, Document, SearchResult
from ..core.embeddings import BaseEmbeddings

logger = logging.getLogger(__name__)


class BaseRetriever(ABC):
    """Base class for retrievers."""
    
    @abstractmethod
    def retrieve(
        self,
        query: str,
        k: int = 10,
        filter: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> List[Document]:
        """Retrieve relevant documents for a query."""
        pass
    
    @abstractmethod
    async def aretrieve(
        self,
        query: str,
        k: int = 10,
        filter: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> List[Document]:
        """Async retrieve relevant documents."""
        pass


class Retriever(BaseRetriever):
    """Standard vector similarity retriever."""
    
    def __init__(
        self,
        vector_store: BaseVectorStore,
        embeddings: BaseEmbeddings,
        search_type: str = "similarity",  # "similarity" or "mmr"
        fetch_k: int = 20,  # For MMR
        lambda_mult: float = 0.5  # For MMR diversity
    ):
        self.vector_store = vector_store
        self.embeddings = embeddings
        self.search_type = search_type
        self.fetch_k = fetch_k
        self.lambda_mult = lambda_mult
    
    def retrieve(
        self,
        query: str,
        k: int = 10,
        filter: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> List[Document]:
        """Retrieve documents using vector similarity."""
        # Embed query
        query_embedding = self.embeddings.embed_query(query)
        
        if self.search_type == "similarity":
            # Standard similarity search
            results = self.vector_store.search(
                query_embedding=query_embedding,
                k=k,
                filter=filter,
                **kwargs
            )
        elif self.search_type == "mmr":
            # Maximal Marginal Relevance
            results = self._mmr_search(
                query_embedding=query_embedding,
                k=k,
                fetch_k=self.fetch_k,
                lambda_mult=self.lambda_mult,
                filter=filter,
                **kwargs
            )
        else:
            raise ValueError(f"Unknown search type: {self.search_type}")
        
        return [r.document for r in results]
    
    async def aretrieve(
        self,
        query: str,
        k: int = 10,
        filter: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> List[Document]:
        """Async retrieve documents."""
        # Embed query
        query_embedding = await self.embeddings.aembed_query(query)
        
        # Search
        results = await self.vector_store.asearch(
            query_embedding=query_embedding,
            k=k,
            filter=filter,
            **kwargs
        )
        
        return [r.document for r in results]
    
    def _mmr_search(
        self,
        query_embedding: List[float],
        k: int,
        fetch_k: int,
        lambda_mult: float,
        filter: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> List[SearchResult]:
        """Maximal Marginal Relevance search."""
        import numpy as np
        
        # Fetch more candidates than needed
        candidates = self.vector_store.search(
            query_embedding=query_embedding,
            k=fetch_k,
            filter=filter,
            **kwargs
        )
        
        if len(candidates) <= k:
            return candidates
        
        # Get embeddings for candidates
        candidate_embeddings = []
        for candidate in candidates:
            if candidate.document.embedding:
                candidate_embeddings.append(candidate.document.embedding)
            else:
                # Need to embed the document
                embedding = self.embeddings.embed_query(candidate.document.content)
                candidate_embeddings.append(embedding)
        
        # Convert to numpy arrays
        query_vec = np.array(query_embedding)
        candidate_vecs = np.array(candidate_embeddings)
        
        # Calculate similarity to query
        query_similarities = np.dot(candidate_vecs, query_vec) / (
            np.linalg.norm(candidate_vecs, axis=1) * np.linalg.norm(query_vec)
        )
        
        # MMR selection
        selected_indices = []
        selected_indices.append(0)  # Start with most similar
        
        while len(selected_indices) < k:
            # Calculate similarity to already selected documents
            selected_vecs = candidate_vecs[selected_indices]
            
            remaining_indices = [i for i in range(len(candidates)) if i not in selected_indices]
            remaining_vecs = candidate_vecs[remaining_indices]
            
            # Calculate max similarity to selected documents
            max_similarities = np.max(
                np.dot(remaining_vecs, selected_vecs.T) / (
                    np.linalg.norm(remaining_vecs, axis=1)[:, np.newaxis] *
                    np.linalg.norm(selected_vecs, axis=1)
                ),
                axis=1
            )
            
            # MMR score
            remaining_query_sims = query_similarities[remaining_indices]
            mmr_scores = lambda_mult * remaining_query_sims - (1 - lambda_mult) * max_similarities
            
            # Select next document
            next_idx = remaining_indices[np.argmax(mmr_scores)]
            selected_indices.append(next_idx)
        
        # Return selected documents
        return [candidates[i] for i in selected_indices]


class HybridRetriever(BaseRetriever):
    """Hybrid retriever combining vector and keyword search."""
    
    def __init__(
        self,
        vector_store: BaseVectorStore,
        embeddings: BaseEmbeddings,
        vector_weight: float = 0.7,
        normalize_scores: bool = True
    ):
        self.vector_store = vector_store
        self.embeddings = embeddings
        self.vector_weight = vector_weight
        self.normalize_scores = normalize_scores
    
    def retrieve(
        self,
        query: str,
        k: int = 10,
        filter: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> List[Document]:
        """Retrieve using hybrid search."""
        # Embed query
        query_embedding = self.embeddings.embed_query(query)
        
        # Perform hybrid search
        results = self.vector_store.hybrid_search(
            query=query,
            query_embedding=query_embedding,
            k=k,
            filter=filter,
            vector_weight=self.vector_weight,
            **kwargs
        )
        
        return [r.document for r in results]
    
    async def aretrieve(
        self,
        query: str,
        k: int = 10,
        filter: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> List[Document]:
        """Async hybrid retrieve."""
        # For now, run sync version in executor
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            self.retrieve,
            query,
            k,
            filter
        )


class MultiRetriever(BaseRetriever):
    """Retriever that queries multiple sources."""
    
    def __init__(
        self,
        retrievers: List[Tuple[BaseRetriever, float]],
        merge_strategy: str = "weighted"  # "weighted", "reciprocal_rank"
    ):
        self.retrievers = retrievers
        self.merge_strategy = merge_strategy
    
    def retrieve(
        self,
        query: str,
        k: int = 10,
        filter: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> List[Document]:
        """Retrieve from multiple sources and merge results."""
        all_results = []
        
        # Retrieve from each source
        for retriever, weight in self.retrievers:
            results = retriever.retrieve(query, k=k*2, filter=filter, **kwargs)
            all_results.append((results, weight))
        
        # Merge results
        if self.merge_strategy == "weighted":
            return self._weighted_merge(all_results, k)
        elif self.merge_strategy == "reciprocal_rank":
            return self._reciprocal_rank_merge(all_results, k)
        else:
            raise ValueError(f"Unknown merge strategy: {self.merge_strategy}")
    
    async def aretrieve(
        self,
        query: str,
        k: int = 10,
        filter: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> List[Document]:
        """Async retrieve from multiple sources."""
        # Create tasks for all retrievers
        tasks = [
            retriever.aretrieve(query, k=k*2, filter=filter, **kwargs)
            for retriever, _ in self.retrievers
        ]
        
        # Wait for all results
        results = await asyncio.gather(*tasks)
        
        # Combine with weights
        all_results = [(results[i], self.retrievers[i][1]) for i in range(len(results))]
        
        # Merge results
        if self.merge_strategy == "weighted":
            return self._weighted_merge(all_results, k)
        elif self.merge_strategy == "reciprocal_rank":
            return self._reciprocal_rank_merge(all_results, k)
        else:
            raise ValueError(f"Unknown merge strategy: {self.merge_strategy}")
    
    def _weighted_merge(
        self,
        all_results: List[Tuple[List[Document], float]],
        k: int
    ) -> List[Document]:
        """Merge results using weighted scores."""
        doc_scores = {}
        
        for results, weight in all_results:
            for i, doc in enumerate(results):
                # Score based on rank
                score = (len(results) - i) / len(results) * weight
                
                if doc.id in doc_scores:
                    doc_scores[doc.id] = (doc_scores[doc.id][0], doc_scores[doc.id][1] + score)
                else:
                    doc_scores[doc.id] = (doc, score)
        
        # Sort by combined score
        sorted_docs = sorted(doc_scores.values(), key=lambda x: x[1], reverse=True)
        
        return [doc for doc, _ in sorted_docs[:k]]
    
    def _reciprocal_rank_merge(
        self,
        all_results: List[Tuple[List[Document], float]],
        k: int
    ) -> List[Document]:
        """Merge using reciprocal rank fusion."""
        doc_scores = {}
        
        for results, weight in all_results:
            for i, doc in enumerate(results):
                # Reciprocal rank score
                score = weight / (i + 1)
                
                if doc.id in doc_scores:
                    doc_scores[doc.id] = (doc_scores[doc.id][0], doc_scores[doc.id][1] + score)
                else:
                    doc_scores[doc.id] = (doc, score)
        
        # Sort by combined score
        sorted_docs = sorted(doc_scores.values(), key=lambda x: x[1], reverse=True)
        
        return [doc for doc, _ in sorted_docs[:k]]