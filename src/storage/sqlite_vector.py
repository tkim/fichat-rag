"""SQLite vector store implementation using sqlite-vss extension."""

import os
import json
import sqlite3
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import logging

from .base import BaseVectorStore, Document

logger = logging.getLogger(__name__)


class SQLiteVectorStore(BaseVectorStore):
    """SQLite with sqlite-vss extension for vector similarity search.
    
    This implementation provides a lightweight, file-based vector store
    perfect for development and small to medium-sized deployments.
    """
    
    def __init__(
        self,
        db_path: str = "fichat.db",
        embedding_dim: int = 384,
        table_name: str = "documents",
        vss_version: str = "v0.1.2"
    ):
        """Initialize SQLite vector store.
        
        Args:
            db_path: Path to SQLite database file
            embedding_dim: Dimension of embeddings
            table_name: Name of the documents table
            vss_version: Version of sqlite-vss to use
        """
        self.db_path = db_path
        self.embedding_dim = embedding_dim
        self.table_name = table_name
        self.vss_table_name = f"vss_{table_name}"
        
        # Ensure directory exists
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        
        # Initialize database
        self._init_db()
    
    def _init_db(self):
        """Initialize database with required tables and extensions."""
        try:
            self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
            
            # Try to load sqlite-vss extension
            try:
                self.conn.enable_load_extension(True)
                
                # Try different possible paths for the extension
                extension_paths = [
                    "vector0",  # Default name
                    "vss0",     # VSS extension
                    "./vector0.so",  # Local directory
                    "./vss0.so",
                    "/usr/local/lib/vector0.so",  # Common installation paths
                    "/usr/local/lib/vss0.so",
                ]
                
                loaded = False
                for ext_path in extension_paths:
                    try:
                        self.conn.load_extension(ext_path)
                        loaded = True
                        logger.info(f"Loaded SQLite extension: {ext_path}")
                        break
                    except sqlite3.OperationalError:
                        continue
                
                if not loaded:
                    logger.warning(
                        "sqlite-vss extension not found. "
                        "Install with: pip install sqlite-vss"
                    )
                    # Continue without vector extension
                    self._use_basic_similarity = True
                else:
                    self._use_basic_similarity = False
                    
            except sqlite3.OperationalError as e:
                logger.warning(f"Cannot load extensions: {e}")
                self._use_basic_similarity = True
            
            # Create main documents table
            self.conn.execute(f"""
                CREATE TABLE IF NOT EXISTS {self.table_name} (
                    id TEXT PRIMARY KEY,
                    content TEXT NOT NULL,
                    metadata TEXT,
                    embedding BLOB
                )
            """)
            
            # Create indexes
            self.conn.execute(f"""
                CREATE INDEX IF NOT EXISTS idx_{self.table_name}_id 
                ON {self.table_name}(id)
            """)
            
            # Create FTS table for keyword search
            self.conn.execute(f"""
                CREATE VIRTUAL TABLE IF NOT EXISTS {self.table_name}_fts 
                USING fts5(id, content)
            """)
            
            if not self._use_basic_similarity:
                # Create VSS virtual table for vector similarity
                try:
                    self.conn.execute(f"""
                        CREATE VIRTUAL TABLE IF NOT EXISTS {self.vss_table_name}
                        USING vss0(embedding({self.embedding_dim}))
                    """)
                except sqlite3.OperationalError:
                    logger.warning("Could not create VSS table, falling back to basic similarity")
                    self._use_basic_similarity = True
            
            self.conn.commit()
            
        except Exception as e:
            logger.error(f"Failed to initialize SQLite database: {e}")
            raise
    
    def _serialize_embedding(self, embedding: List[float]) -> bytes:
        """Serialize embedding to bytes for storage."""
        return np.array(embedding, dtype=np.float32).tobytes()
    
    def _deserialize_embedding(self, embedding_bytes: bytes) -> List[float]:
        """Deserialize embedding from bytes."""
        return np.frombuffer(embedding_bytes, dtype=np.float32).tolist()
    
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
            ids = [self._generate_id() for _ in texts]
        
        # Ensure metadatas list
        if metadatas is None:
            metadatas = [{}] * len(texts)
        
        cursor = self.conn.cursor()
        
        try:
            # Insert documents
            for i, (doc_id, text, embedding, metadata) in enumerate(
                zip(ids, texts, embeddings, metadatas)
            ):
                # Serialize embedding
                embedding_bytes = self._serialize_embedding(embedding)
                metadata_json = json.dumps(metadata)
                
                # Insert into main table
                cursor.execute(f"""
                    INSERT OR REPLACE INTO {self.table_name} 
                    (id, content, metadata, embedding)
                    VALUES (?, ?, ?, ?)
                """, (doc_id, text, metadata_json, embedding_bytes))
                
                # Insert into FTS table
                cursor.execute(f"""
                    INSERT OR REPLACE INTO {self.table_name}_fts 
                    (id, content)
                    VALUES (?, ?)
                """, (doc_id, text))
                
                # Insert into VSS table if available
                if not self._use_basic_similarity:
                    try:
                        cursor.execute(f"""
                            INSERT OR REPLACE INTO {self.vss_table_name}
                            (rowid, embedding)
                            VALUES (?, ?)
                        """, (i + 1, embedding_bytes))
                    except sqlite3.OperationalError:
                        pass
            
            self.conn.commit()
            logger.info(f"Added {len(texts)} documents to SQLite vector store")
            
            return ids
            
        except Exception as e:
            self.conn.rollback()
            logger.error(f"Failed to add documents: {e}")
            raise
    
    def search(
        self,
        query_embedding: List[float],
        k: int = 5,
        filter: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """Search for similar documents."""
        cursor = self.conn.cursor()
        
        try:
            if self._use_basic_similarity:
                # Fallback to basic cosine similarity
                return self._search_basic(query_embedding, k, filter)
            
            # Use VSS for vector search
            query_bytes = self._serialize_embedding(query_embedding)
            
            # Build query with optional metadata filter
            base_query = f"""
                SELECT 
                    d.id,
                    d.content,
                    d.metadata,
                    d.embedding,
                    vss_distance_l2(d.embedding, ?) as distance
                FROM {self.table_name} d
            """
            
            params = [query_bytes]
            
            if filter:
                # Add metadata filters
                filter_conditions = []
                for key, value in filter.items():
                    filter_conditions.append(
                        f"json_extract(d.metadata, '$.{key}') = ?"
                    )
                    params.append(value)
                
                if filter_conditions:
                    base_query += " WHERE " + " AND ".join(filter_conditions)
            
            base_query += f" ORDER BY distance LIMIT {k}"
            
            cursor.execute(base_query, params)
            results = cursor.fetchall()
            
            # Format results
            formatted_results = []
            for row in results:
                doc_id, content, metadata_json, embedding_bytes, distance = row
                
                # Calculate similarity score (1 - normalized distance)
                score = max(0, 1 - (distance / 2))  # Normalize L2 distance
                
                formatted_results.append({
                    "id": doc_id,
                    "content": content,
                    "metadata": json.loads(metadata_json),
                    "score": score
                })
            
            return formatted_results
            
        except Exception as e:
            logger.error(f"Search failed: {e}")
            return self._search_basic(query_embedding, k, filter)
    
    def _search_basic(
        self,
        query_embedding: List[float],
        k: int = 5,
        filter: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Basic search using cosine similarity (fallback)."""
        cursor = self.conn.cursor()
        
        # Build query
        query = f"SELECT id, content, metadata, embedding FROM {self.table_name}"
        params = []
        
        if filter:
            filter_conditions = []
            for key, value in filter.items():
                filter_conditions.append(
                    f"json_extract(metadata, '$.{key}') = ?"
                )
                params.append(value)
            
            if filter_conditions:
                query += " WHERE " + " AND ".join(filter_conditions)
        
        cursor.execute(query, params)
        
        # Calculate similarities
        query_np = np.array(query_embedding, dtype=np.float32)
        results_with_scores = []
        
        for row in cursor.fetchall():
            doc_id, content, metadata_json, embedding_bytes = row
            
            # Deserialize embedding
            doc_embedding = np.frombuffer(embedding_bytes, dtype=np.float32)
            
            # Calculate cosine similarity
            similarity = np.dot(query_np, doc_embedding) / (
                np.linalg.norm(query_np) * np.linalg.norm(doc_embedding)
            )
            
            results_with_scores.append({
                "id": doc_id,
                "content": content,
                "metadata": json.loads(metadata_json),
                "score": float(similarity)
            })
        
        # Sort by score and return top k
        results_with_scores.sort(key=lambda x: x["score"], reverse=True)
        return results_with_scores[:k]
    
    def hybrid_search(
        self,
        query_text: str,
        query_embedding: List[float],
        k: int = 5,
        alpha: float = 0.5,
        filter: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """Hybrid search combining vector and keyword search."""
        # Get vector search results
        vector_results = self.search(query_embedding, k * 2, filter)
        
        # Get keyword search results
        keyword_results = self._keyword_search(query_text, k * 2, filter)
        
        # Combine and rerank
        combined_scores = {}
        all_ids = set()
        
        # Add vector search scores
        for result in vector_results:
            doc_id = result["id"]
            all_ids.add(doc_id)
            combined_scores[doc_id] = {
                "vector_score": result["score"],
                "keyword_score": 0,
                "content": result["content"],
                "metadata": result["metadata"]
            }
        
        # Add keyword search scores
        for result in keyword_results:
            doc_id = result["id"]
            all_ids.add(doc_id)
            
            if doc_id in combined_scores:
                combined_scores[doc_id]["keyword_score"] = result["score"]
            else:
                combined_scores[doc_id] = {
                    "vector_score": 0,
                    "keyword_score": result["score"],
                    "content": result["content"],
                    "metadata": result["metadata"]
                }
        
        # Calculate combined scores
        final_results = []
        for doc_id, scores in combined_scores.items():
            combined_score = (
                alpha * scores["vector_score"] + 
                (1 - alpha) * scores["keyword_score"]
            )
            
            final_results.append({
                "id": doc_id,
                "content": scores["content"],
                "metadata": scores["metadata"],
                "score": combined_score
            })
        
        # Sort and return top k
        final_results.sort(key=lambda x: x["score"], reverse=True)
        return final_results[:k]
    
    def _keyword_search(
        self,
        query_text: str,
        k: int = 5,
        filter: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Perform keyword search using FTS5."""
        cursor = self.conn.cursor()
        
        # Build FTS query
        fts_query = f"""
            SELECT 
                f.id,
                d.content,
                d.metadata,
                rank
            FROM {self.table_name}_fts f
            JOIN {self.table_name} d ON f.id = d.id
            WHERE {self.table_name}_fts MATCH ?
        """
        
        params = [query_text]
        
        if filter:
            filter_conditions = []
            for key, value in filter.items():
                filter_conditions.append(
                    f"json_extract(d.metadata, '$.{key}') = ?"
                )
                params.append(value)
            
            if filter_conditions:
                fts_query += " AND " + " AND ".join(filter_conditions)
        
        fts_query += " ORDER BY rank LIMIT ?"
        params.append(k)
        
        cursor.execute(fts_query, params)
        
        results = []
        for row in cursor.fetchall():
            doc_id, content, metadata_json, rank = row
            
            # Convert rank to score (FTS5 rank is negative)
            score = 1.0 / (1.0 + abs(rank))
            
            results.append({
                "id": doc_id,
                "content": content,
                "metadata": json.loads(metadata_json),
                "score": score
            })
        
        return results
    
    def get(self, ids: List[str]) -> List[Optional[Document]]:
        """Get documents by IDs."""
        cursor = self.conn.cursor()
        
        placeholders = ",".join(["?" for _ in ids])
        query = f"""
            SELECT id, content, metadata 
            FROM {self.table_name}
            WHERE id IN ({placeholders})
        """
        
        cursor.execute(query, ids)
        
        # Create mapping of results
        results_map = {}
        for row in cursor.fetchall():
            doc_id, content, metadata_json = row
            results_map[doc_id] = Document(
                id=doc_id,
                content=content,
                metadata=json.loads(metadata_json)
            )
        
        # Return in order of requested IDs
        return [results_map.get(doc_id) for doc_id in ids]
    
    def update(
        self,
        ids: List[str],
        texts: Optional[List[str]] = None,
        embeddings: Optional[List[List[float]]] = None,
        metadatas: Optional[List[Dict[str, Any]]] = None
    ):
        """Update existing documents."""
        cursor = self.conn.cursor()
        
        try:
            for i, doc_id in enumerate(ids):
                updates = []
                params = []
                
                if texts and i < len(texts):
                    updates.append("content = ?")
                    params.append(texts[i])
                
                if embeddings and i < len(embeddings):
                    updates.append("embedding = ?")
                    params.append(self._serialize_embedding(embeddings[i]))
                
                if metadatas and i < len(metadatas):
                    updates.append("metadata = ?")
                    params.append(json.dumps(metadatas[i]))
                
                if updates:
                    params.append(doc_id)
                    query = f"""
                        UPDATE {self.table_name}
                        SET {", ".join(updates)}
                        WHERE id = ?
                    """
                    cursor.execute(query, params)
                    
                    # Update FTS if content changed
                    if texts and i < len(texts):
                        cursor.execute(f"""
                            UPDATE {self.table_name}_fts
                            SET content = ?
                            WHERE id = ?
                        """, (texts[i], doc_id))
            
            self.conn.commit()
            
        except Exception as e:
            self.conn.rollback()
            logger.error(f"Failed to update documents: {e}")
            raise
    
    def delete(self, ids: List[str]):
        """Delete documents by IDs."""
        cursor = self.conn.cursor()
        
        try:
            placeholders = ",".join(["?" for _ in ids])
            
            # Delete from main table
            cursor.execute(f"""
                DELETE FROM {self.table_name}
                WHERE id IN ({placeholders})
            """, ids)
            
            # Delete from FTS table
            cursor.execute(f"""
                DELETE FROM {self.table_name}_fts
                WHERE id IN ({placeholders})
            """, ids)
            
            self.conn.commit()
            
        except Exception as e:
            self.conn.rollback()
            logger.error(f"Failed to delete documents: {e}")
            raise
    
    def count(self, filter: Optional[Dict[str, Any]] = None) -> int:
        """Count documents in the store."""
        cursor = self.conn.cursor()
        
        query = f"SELECT COUNT(*) FROM {self.table_name}"
        params = []
        
        if filter:
            filter_conditions = []
            for key, value in filter.items():
                filter_conditions.append(
                    f"json_extract(metadata, '$.{key}') = ?"
                )
                params.append(value)
            
            if filter_conditions:
                query += " WHERE " + " AND ".join(filter_conditions)
        
        cursor.execute(query, params)
        return cursor.fetchone()[0]
    
    def clear(self):
        """Clear all documents from the store."""
        cursor = self.conn.cursor()
        
        try:
            cursor.execute(f"DELETE FROM {self.table_name}")
            cursor.execute(f"DELETE FROM {self.table_name}_fts")
            
            if not self._use_basic_similarity:
                try:
                    cursor.execute(f"DELETE FROM {self.vss_table_name}")
                except sqlite3.OperationalError:
                    pass
            
            self.conn.commit()
            
        except Exception as e:
            self.conn.rollback()
            logger.error(f"Failed to clear store: {e}")
            raise
    
    def close(self):
        """Close the database connection."""
        if hasattr(self, "conn"):
            self.conn.close()
    
    def __del__(self):
        """Cleanup on deletion."""
        self.close()