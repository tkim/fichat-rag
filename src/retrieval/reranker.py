"""Reranking implementations for improving retrieval quality."""

from typing import List, Tuple, Optional
import numpy as np
from abc import ABC, abstractmethod
import logging
from sentence_transformers import CrossEncoder

from ..storage.base import Document
from ..core.embeddings import BaseEmbeddings

logger = logging.getLogger(__name__)


class BaseReranker(ABC):
    """Base class for rerankers."""
    
    @abstractmethod
    def rerank(
        self,
        query: str,
        documents: List[Document],
        top_k: Optional[int] = None
    ) -> List[Tuple[Document, float]]:
        """Rerank documents based on relevance to query."""
        pass


class CrossEncoderReranker(BaseReranker):
    """Reranker using cross-encoder models."""
    
    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        device: Optional[str] = None,
        max_length: int = 512
    ):
        self.model_name = model_name
        self.max_length = max_length
        
        try:
            self.model = CrossEncoder(model_name, device=device, max_length=max_length)
        except Exception as e:
            logger.error(f"Failed to load cross-encoder model: {e}")
            raise
    
    def rerank(
        self,
        query: str,
        documents: List[Document],
        top_k: Optional[int] = None
    ) -> List[Tuple[Document, float]]:
        """Rerank documents using cross-encoder."""
        if not documents:
            return []
        
        # Prepare pairs for cross-encoder
        pairs = [[query, doc.content] for doc in documents]
        
        # Get scores
        scores = self.model.predict(pairs)
        
        # Combine documents with scores
        doc_scores = list(zip(documents, scores))
        
        # Sort by score (descending)
        doc_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Return top_k if specified
        if top_k is not None:
            doc_scores = doc_scores[:top_k]
        
        return doc_scores


class EmbeddingReranker(BaseReranker):
    """Reranker using embedding similarity."""
    
    def __init__(
        self,
        embeddings: BaseEmbeddings,
        similarity_metric: str = "cosine"  # "cosine", "euclidean", "dot"
    ):
        self.embeddings = embeddings
        self.similarity_metric = similarity_metric
    
    def rerank(
        self,
        query: str,
        documents: List[Document],
        top_k: Optional[int] = None
    ) -> List[Tuple[Document, float]]:
        """Rerank using embedding similarity."""
        if not documents:
            return []
        
        # Embed query
        query_embedding = np.array(self.embeddings.embed_query(query))
        
        # Embed documents if needed
        doc_embeddings = []
        for doc in documents:
            if doc.embedding:
                doc_embeddings.append(np.array(doc.embedding))
            else:
                embedding = self.embeddings.embed_query(doc.content)
                doc_embeddings.append(np.array(embedding))
        
        doc_embeddings = np.array(doc_embeddings)
        
        # Calculate similarities
        if self.similarity_metric == "cosine":
            # Normalize vectors
            query_norm = query_embedding / np.linalg.norm(query_embedding)
            doc_norms = doc_embeddings / np.linalg.norm(doc_embeddings, axis=1, keepdims=True)
            scores = np.dot(doc_norms, query_norm)
        elif self.similarity_metric == "euclidean":
            # Negative euclidean distance (closer = higher score)
            scores = -np.linalg.norm(doc_embeddings - query_embedding, axis=1)
        elif self.similarity_metric == "dot":
            scores = np.dot(doc_embeddings, query_embedding)
        else:
            raise ValueError(f"Unknown similarity metric: {self.similarity_metric}")
        
        # Combine documents with scores
        doc_scores = list(zip(documents, scores))
        
        # Sort by score (descending)
        doc_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Return top_k if specified
        if top_k is not None:
            doc_scores = doc_scores[:top_k]
        
        return doc_scores


class LLMReranker(BaseReranker):
    """Reranker using LLM for relevance scoring."""
    
    def __init__(self, llm, score_prompt_template: Optional[str] = None):
        self.llm = llm
        self.score_prompt_template = score_prompt_template or """
Rate the relevance of the following document to the query on a scale of 0-10.
Query: {query}
Document: {document}
Relevance score (0-10):"""
    
    def rerank(
        self,
        query: str,
        documents: List[Document],
        top_k: Optional[int] = None
    ) -> List[Tuple[Document, float]]:
        """Rerank using LLM relevance scoring."""
        if not documents:
            return []
        
        doc_scores = []
        
        for doc in documents:
            # Create scoring prompt
            prompt = self.score_prompt_template.format(
                query=query,
                document=doc.content[:1000]  # Limit document length
            )
            
            # Get LLM response
            response = self.llm.generate(prompt, max_tokens=10)
            
            # Extract score
            try:
                score = float(response.content.strip().split()[0])
                score = max(0, min(10, score))  # Clamp to 0-10
            except (ValueError, IndexError):
                logger.warning(f"Failed to parse score from LLM response: {response.content}")
                score = 5.0  # Default middle score
            
            doc_scores.append((doc, score))
        
        # Sort by score (descending)
        doc_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Return top_k if specified
        if top_k is not None:
            doc_scores = doc_scores[:top_k]
        
        return doc_scores


class HybridReranker(BaseReranker):
    """Combines multiple reranking strategies."""
    
    def __init__(
        self,
        rerankers: List[Tuple[BaseReranker, float]],
        normalize_scores: bool = True
    ):
        self.rerankers = rerankers
        self.normalize_scores = normalize_scores
    
    def rerank(
        self,
        query: str,
        documents: List[Document],
        top_k: Optional[int] = None
    ) -> List[Tuple[Document, float]]:
        """Rerank using multiple rerankers."""
        if not documents:
            return []
        
        # Get scores from each reranker
        all_scores = {}
        
        for reranker, weight in self.rerankers:
            ranked_docs = reranker.rerank(query, documents)
            
            # Normalize scores if needed
            if self.normalize_scores and ranked_docs:
                scores = [score for _, score in ranked_docs]
                min_score = min(scores)
                max_score = max(scores)
                
                if max_score > min_score:
                    for doc, score in ranked_docs:
                        normalized_score = (score - min_score) / (max_score - min_score)
                        
                        if doc.id in all_scores:
                            all_scores[doc.id] = (
                                all_scores[doc.id][0],
                                all_scores[doc.id][1] + normalized_score * weight
                            )
                        else:
                            all_scores[doc.id] = (doc, normalized_score * weight)
            else:
                for doc, score in ranked_docs:
                    if doc.id in all_scores:
                        all_scores[doc.id] = (
                            all_scores[doc.id][0],
                            all_scores[doc.id][1] + score * weight
                        )
                    else:
                        all_scores[doc.id] = (doc, score * weight)
        
        # Sort by combined score
        doc_scores = sorted(all_scores.values(), key=lambda x: x[1], reverse=True)
        
        # Return top_k if specified
        if top_k is not None:
            doc_scores = doc_scores[:top_k]
        
        return doc_scores