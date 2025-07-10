"""PostgreSQL with pgvector implementation."""

import json
import asyncio
import asyncpg
import psycopg2
from psycopg2.extras import RealDictCursor, execute_batch
from pgvector.psycopg2 import register_vector
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from contextlib import contextmanager
import logging

from .base import BaseVectorStore, Document, SearchResult

logger = logging.getLogger(__name__)


class PostgresVectorStore(BaseVectorStore):
    """PostgreSQL with pgvector vector store implementation."""
    
    def __init__(
        self,
        connection_string: Optional[str] = None,
        host: str = "localhost",
        port: int = 5432,
        database: str = "fichat_rag",
        user: str = "rag_user",
        password: str = "rag_pass",
        table_name: str = "documents",
        embedding_dim: int = 384,
        pool_size: int = 10,
        **kwargs
    ):
        # Build connection string if not provided
        if connection_string:
            self.connection_string = connection_string
        else:
            self.connection_string = (
                f"postgresql://{user}:{password}@{host}:{port}/{database}"
            )
        
        self.table_name = table_name
        self.embedding_dim = embedding_dim
        self.pool_size = pool_size
        
        # Initialize database
        self._init_db()
    
    def _init_db(self):
        """Initialize database with pgvector extension and tables."""
        with self._get_connection() as conn:
            with conn.cursor() as cur:
                # Create pgvector extension
                cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
                
                # Create documents table
                cur.execute(f"""
                    CREATE TABLE IF NOT EXISTS {self.table_name} (
                        id TEXT PRIMARY KEY,
                        content TEXT NOT NULL,
                        metadata JSONB,
                        embedding vector({self.embedding_dim}),
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                
                # Create indexes
                cur.execute(f"""
                    CREATE INDEX IF NOT EXISTS idx_{self.table_name}_embedding 
                    ON {self.table_name} USING ivfflat (embedding vector_l2_ops)
                    WITH (lists = 100)
                """)
                
                cur.execute(f"""
                    CREATE INDEX IF NOT EXISTS idx_{self.table_name}_metadata 
                    ON {self.table_name} USING GIN (metadata)
                """)
                
                # Create full-text search index
                cur.execute(f"""
                    ALTER TABLE {self.table_name} 
                    ADD COLUMN IF NOT EXISTS content_tsvector tsvector
                    GENERATED ALWAYS AS (to_tsvector('english', content)) STORED
                """)
                
                cur.execute(f"""
                    CREATE INDEX IF NOT EXISTS idx_{self.table_name}_content_fts 
                    ON {self.table_name} USING GIN (content_tsvector)
                """)
                
                conn.commit()
    
    @contextmanager
    def _get_connection(self):
        """Get a database connection."""
        conn = psycopg2.connect(self.connection_string)
        register_vector(conn)
        try:
            yield conn
        finally:
            conn.close()
    
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
        
        with self._get_connection() as conn:
            with conn.cursor() as cur:
                # Prepare data for insertion
                values = []
                ids = []
                
                for i, doc in enumerate(documents):
                    embedding = embeddings[i] if embeddings else doc.embedding
                    if not embedding:
                        raise ValueError(f"No embedding provided for document {i}")
                    
                    values.append((
                        doc.id,
                        doc.content,
                        json.dumps(doc.metadata),
                        embedding
                    ))
                    ids.append(doc.id)
                
                # Batch insert
                execute_batch(
                    cur,
                    f"""
                    INSERT INTO {self.table_name} (id, content, metadata, embedding)
                    VALUES (%s, %s, %s, %s)
                    ON CONFLICT (id) DO UPDATE SET
                        content = EXCLUDED.content,
                        metadata = EXCLUDED.metadata,
                        embedding = EXCLUDED.embedding,
                        updated_at = CURRENT_TIMESTAMP
                    """,
                    values,
                    page_size=100
                )
                
                conn.commit()
                
        return ids
    
    def search(
        self,
        query_embedding: List[float],
        k: int = 10,
        filter: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> List[SearchResult]:
        """Search for similar documents using vector similarity."""
        with self._get_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                # Build query with optional metadata filter
                where_clause = ""
                params = [query_embedding, k]
                
                if filter:
                    where_conditions = []
                    for key, value in filter.items():
                        where_conditions.append(f"metadata->>{self._quote_literal(key)} = %s")
                        params.append(str(value))
                    where_clause = "WHERE " + " AND ".join(where_conditions)
                
                query = f"""
                    SELECT 
                        id,
                        content,
                        metadata,
                        embedding <-> %s AS distance,
                        1 - (embedding <-> %s) AS score
                    FROM {self.table_name}
                    {where_clause}
                    ORDER BY distance
                    LIMIT %s
                """
                
                # Add query_embedding twice for distance and score calculation
                params.insert(1, query_embedding)
                
                cur.execute(query, params)
                rows = cur.fetchall()
                
                results = []
                for i, row in enumerate(rows):
                    doc = Document(
                        id=row["id"],
                        content=row["content"],
                        metadata=row["metadata"] or {},
                        embedding=None  # Don't return embeddings by default
                    )
                    results.append(SearchResult(
                        document=doc,
                        score=float(row["score"]),
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
        with self._get_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                # Build filter clause
                filter_clause = ""
                params = []
                
                if filter:
                    where_conditions = []
                    for key, value in filter.items():
                        where_conditions.append(f"metadata->>{self._quote_literal(key)} = %s")
                        params.append(str(value))
                    filter_clause = "AND " + " AND ".join(where_conditions)
                
                # Combined query with both vector and text search
                query_sql = f"""
                    WITH vector_search AS (
                        SELECT 
                            id,
                            1 - (embedding <-> %s) AS vector_score
                        FROM {self.table_name}
                        WHERE 1=1 {filter_clause}
                        ORDER BY embedding <-> %s
                        LIMIT %s
                    ),
                    text_search AS (
                        SELECT 
                            id,
                            ts_rank_cd(content_tsvector, plainto_tsquery('english', %s)) AS text_score
                        FROM {self.table_name}
                        WHERE content_tsvector @@ plainto_tsquery('english', %s)
                        {filter_clause}
                        ORDER BY text_score DESC
                        LIMIT %s
                    ),
                    combined AS (
                        SELECT 
                            COALESCE(v.id, t.id) AS id,
                            COALESCE(v.vector_score, 0) * %s AS weighted_vector_score,
                            COALESCE(t.text_score, 0) * %s AS weighted_text_score
                        FROM vector_search v
                        FULL OUTER JOIN text_search t ON v.id = t.id
                    )
                    SELECT 
                        d.id,
                        d.content,
                        d.metadata,
                        c.weighted_vector_score + c.weighted_text_score AS score
                    FROM combined c
                    JOIN {self.table_name} d ON c.id = d.id
                    ORDER BY score DESC
                    LIMIT %s
                """
                
                # Build parameters
                all_params = [
                    query_embedding,  # for vector similarity
                    query_embedding,  # for vector ordering
                    k * 2,           # vector limit
                    *params,         # filter params for vector
                    query,           # for text search
                    query,           # for text matching
                    *params,         # filter params for text
                    k * 2,           # text limit
                    vector_weight,   # vector weight
                    1 - vector_weight,  # text weight
                    k               # final limit
                ]
                
                cur.execute(query_sql, all_params)
                rows = cur.fetchall()
                
                results = []
                for i, row in enumerate(rows):
                    doc = Document(
                        id=row["id"],
                        content=row["content"],
                        metadata=row["metadata"] or {}
                    )
                    results.append(SearchResult(
                        document=doc,
                        score=float(row["score"]),
                        rank=i + 1
                    ))
                
                return results
    
    def delete(self, ids: List[str]) -> bool:
        """Delete documents by IDs."""
        if not ids:
            return True
        
        with self._get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    f"DELETE FROM {self.table_name} WHERE id = ANY(%s)",
                    (ids,)
                )
                conn.commit()
                return cur.rowcount > 0
    
    def get(self, ids: List[str]) -> List[Document]:
        """Get documents by IDs."""
        if not ids:
            return []
        
        with self._get_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(
                    f"SELECT id, content, metadata FROM {self.table_name} WHERE id = ANY(%s)",
                    (ids,)
                )
                rows = cur.fetchall()
                
                return [
                    Document(
                        id=row["id"],
                        content=row["content"],
                        metadata=row["metadata"] or {}
                    )
                    for row in rows
                ]
    
    def update(self, documents: List[Document]) -> bool:
        """Update existing documents."""
        if not documents:
            return True
        
        with self._get_connection() as conn:
            with conn.cursor() as cur:
                values = [
                    (doc.content, json.dumps(doc.metadata), doc.id)
                    for doc in documents
                ]
                
                execute_batch(
                    cur,
                    f"""
                    UPDATE {self.table_name}
                    SET content = %s, metadata = %s, updated_at = CURRENT_TIMESTAMP
                    WHERE id = %s
                    """,
                    values
                )
                
                conn.commit()
                return cur.rowcount > 0
    
    def count(self, filter: Optional[Dict[str, Any]] = None) -> int:
        """Count documents in the store."""
        with self._get_connection() as conn:
            with conn.cursor() as cur:
                where_clause = ""
                params = []
                
                if filter:
                    where_conditions = []
                    for key, value in filter.items():
                        where_conditions.append(f"metadata->>{self._quote_literal(key)} = %s")
                        params.append(str(value))
                    where_clause = "WHERE " + " AND ".join(where_conditions)
                
                cur.execute(
                    f"SELECT COUNT(*) FROM {self.table_name} {where_clause}",
                    params
                )
                
                return cur.fetchone()[0]
    
    def clear(self) -> bool:
        """Clear all documents from the store."""
        with self._get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(f"TRUNCATE TABLE {self.table_name}")
                conn.commit()
                return True
    
    async def asearch(
        self,
        query_embedding: List[float],
        k: int = 10,
        filter: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> List[SearchResult]:
        """Async search for similar documents."""
        # For now, run sync search in executor
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            self.search,
            query_embedding,
            k,
            filter
        )
    
    def _quote_literal(self, value: str) -> str:
        """Safely quote a literal value for SQL."""
        return f"'{value.replace(\"'\", \"''\")}'"