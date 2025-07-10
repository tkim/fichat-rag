"""Main RAG implementation."""

from typing import List, Dict, Any, Optional, Union
import logging
from pathlib import Path

from .config import Config
from .llm import BaseLLM, OllamaLLM, OpenAILLM
from .embeddings import BaseEmbeddings, SentenceTransformerEmbeddings
from ..storage.base import BaseVectorStore, Document
from ..storage.postgres_vector import PostgresVectorStore
from ..storage.memory import InMemoryVectorStore
from ..ingestion.chunkers import SemanticChunker, BaseChunker
from ..ingestion.processors import DocumentProcessor
from ..ingestion.loaders import PDFLoader, TextLoader
from ..retrieval.retriever import HybridRetriever, BaseRetriever
from ..retrieval.reranker import CrossEncoderReranker, BaseReranker
from ..generation.generator import ResponseGenerator

logger = logging.getLogger(__name__)


class RAG:
    """Main RAG class orchestrating all components."""
    
    def __init__(
        self,
        config: Optional[Config] = None,
        llm: Optional[BaseLLM] = None,
        embeddings: Optional[BaseEmbeddings] = None,
        vector_store: Optional[BaseVectorStore] = None,
        chunker: Optional[BaseChunker] = None,
        retriever: Optional[BaseRetriever] = None,
        reranker: Optional[BaseReranker] = None,
        **kwargs
    ):
        # Use provided config or create from environment
        self.config = config or Config.from_env()
        
        # Initialize components
        self.llm = llm or self._create_llm()
        self.embeddings = embeddings or self._create_embeddings()
        self.vector_store = vector_store or self._create_vector_store()
        self.chunker = chunker or self._create_chunker()
        self.retriever = retriever or self._create_retriever()
        self.reranker = reranker or self._create_reranker()
        
        # Document processor
        self.doc_processor = DocumentProcessor()
        
        # Response generator
        self.response_generator = ResponseGenerator(
            llm=self.llm,
            prompt_template=self.config.generation_prompt_template,
            system_prompt=self.config.generation_system_prompt
        )
        
        # Document loaders
        self.loaders = {
            'pdf': PDFLoader(),
            'txt': TextLoader(),
            'text': TextLoader(),
            'md': TextLoader(),
            'markdown': TextLoader(),
        }
    
    def _create_llm(self) -> BaseLLM:
        """Create LLM based on config."""
        if self.config.llm_provider == "ollama":
            return OllamaLLM(
                model=self.config.llm_model,
                base_url=self.config.llm_base_url,
                temperature=self.config.llm_temperature,
                max_tokens=self.config.llm_max_tokens
            )
        elif self.config.llm_provider == "openai":
            return OpenAILLM(
                model=self.config.llm_model,
                base_url=self.config.llm_base_url,
                temperature=self.config.llm_temperature,
                max_tokens=self.config.llm_max_tokens
            )
        else:
            raise ValueError(f"Unknown LLM provider: {self.config.llm_provider}")
    
    def _create_embeddings(self) -> BaseEmbeddings:
        """Create embeddings based on config."""
        return SentenceTransformerEmbeddings(
            model_name=self.config.embedding_model,
            device=self.config.embedding_device,
            batch_size=self.config.embedding_batch_size
        )
    
    def _create_vector_store(self) -> BaseVectorStore:
        """Create vector store based on config."""
        if self.config.vector_store_type == "postgres":
            return PostgresVectorStore(
                embedding_dim=self.embeddings.embedding_dim,
                **self.config.vector_store_config
            )
        elif self.config.vector_store_type == "memory":
            return InMemoryVectorStore()
        else:
            raise ValueError(f"Unknown vector store type: {self.config.vector_store_type}")
    
    def _create_chunker(self) -> BaseChunker:
        """Create chunker based on config."""
        return SemanticChunker(
            chunk_size=self.config.chunk_size,
            chunk_overlap=self.config.chunk_overlap
        )
    
    def _create_retriever(self) -> BaseRetriever:
        """Create retriever based on config."""
        return HybridRetriever(
            vector_store=self.vector_store,
            embeddings=self.embeddings,
            vector_weight=0.7
        )
    
    def _create_reranker(self) -> Optional[BaseReranker]:
        """Create reranker if enabled."""
        if self.config.retrieval_rerank:
            try:
                return CrossEncoderReranker()
            except Exception as e:
                logger.warning(f"Failed to create reranker: {e}")
                return None
        return None
    
    def add_documents(
        self,
        sources: Union[str, List[str], List[Document]],
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> List[str]:
        """
        Add documents to the RAG system.
        
        Args:
            sources: File path(s) or Document objects
            metadata: Additional metadata to add to all documents
            
        Returns:
            List of document IDs
        """
        documents = []
        
        # Handle different input types
        if isinstance(sources, str):
            sources = [sources]
        
        for source in sources:
            if isinstance(source, Document):
                documents.append(source)
            elif isinstance(source, (str, Path)):
                # Load document from file
                loaded_docs = self._load_documents(str(source), metadata)
                documents.extend(loaded_docs)
            else:
                raise ValueError(f"Unsupported source type: {type(source)}")
        
        # Process and chunk documents
        all_chunks = []
        
        for doc in documents:
            # Process document
            processed_content = self.doc_processor.process(doc.content, doc.metadata)
            
            # Chunk document
            chunks = self.chunker.chunk(processed_content)
            
            # Create chunk documents
            for i, (chunk_text, start, end) in enumerate(chunks):
                chunk_metadata = {
                    **doc.metadata,
                    "chunk_index": i,
                    "chunk_start": start,
                    "chunk_end": end,
                    "parent_doc_id": doc.id
                }
                
                chunk_doc = Document(
                    content=chunk_text,
                    metadata=chunk_metadata
                )
                all_chunks.append(chunk_doc)
        
        # Embed chunks
        logger.info(f"Embedding {len(all_chunks)} chunks...")
        texts = [chunk.content for chunk in all_chunks]
        embeddings = self.embeddings.embed_documents(texts)
        
        # Add to vector store
        logger.info(f"Adding chunks to vector store...")
        ids = self.vector_store.add_documents(all_chunks, embeddings=embeddings)
        
        logger.info(f"Successfully added {len(ids)} chunks from {len(documents)} documents")
        return ids
    
    def _load_documents(
        self,
        file_path: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> List[Document]:
        """Load documents from file."""
        path = Path(file_path)
        
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        # Get file extension
        ext = path.suffix.lower()[1:]  # Remove the dot
        
        # Get appropriate loader
        loader = self.loaders.get(ext)
        if not loader:
            # Default to text loader
            loader = self.loaders['txt']
            logger.warning(f"No specific loader for {ext}, using text loader")
        
        # Load document
        doc_data = loader.load(file_path)
        
        # Merge metadata
        doc_metadata = {**doc_data['metadata'], **(metadata or {})}
        
        # Create document
        document = Document(
            content=doc_data['content'],
            metadata=doc_metadata
        )
        
        return [document]
    
    def query(
        self,
        query: str,
        k: int = None,
        filter: Optional[Dict[str, Any]] = None,
        include_sources: bool = True,
        stream: bool = False,
        **kwargs
    ) -> Union[str, Dict[str, Any]]:
        """
        Query the RAG system.
        
        Args:
            query: The question to answer
            k: Number of documents to retrieve
            filter: Metadata filter for retrieval
            include_sources: Whether to include source documents in response
            stream: Whether to stream the response
            
        Returns:
            Answer string or dict with answer and sources
        """
        k = k or self.config.retrieval_top_k
        
        # Retrieve relevant documents
        logger.info(f"Retrieving {k} documents for query: {query[:100]}...")
        retrieved_docs = self.retriever.retrieve(query, k=k, filter=filter)
        
        # Rerank if enabled
        if self.reranker and retrieved_docs:
            logger.info("Reranking documents...")
            reranked = self.reranker.rerank(
                query,
                retrieved_docs,
                top_k=self.config.retrieval_rerank_top_k
            )
            retrieved_docs = [doc for doc, _ in reranked]
        
        # Generate response
        logger.info("Generating response...")
        
        if stream:
            # Stream response
            return self.response_generator.stream_generate(
                query=query,
                documents=retrieved_docs,
                **kwargs
            )
        else:
            response = self.response_generator.generate(
                query=query,
                documents=retrieved_docs,
                **kwargs
            )
            
            if include_sources:
                return {
                    "answer": response,
                    "sources": [
                        {
                            "content": doc.content,
                            "metadata": doc.metadata
                        }
                        for doc in retrieved_docs
                    ]
                }
            else:
                return response
    
    async def aquery(
        self,
        query: str,
        k: int = None,
        filter: Optional[Dict[str, Any]] = None,
        include_sources: bool = True,
        **kwargs
    ) -> Union[str, Dict[str, Any]]:
        """Async query the RAG system."""
        k = k or self.config.retrieval_top_k
        
        # Retrieve relevant documents
        retrieved_docs = await self.retriever.aretrieve(query, k=k, filter=filter)
        
        # Rerank if enabled (sync for now)
        if self.reranker and retrieved_docs:
            reranked = self.reranker.rerank(
                query,
                retrieved_docs,
                top_k=self.config.retrieval_rerank_top_k
            )
            retrieved_docs = [doc for doc, _ in reranked]
        
        # Generate response
        response = await self.response_generator.agenerate(
            query=query,
            documents=retrieved_docs,
            **kwargs
        )
        
        if include_sources:
            return {
                "answer": response,
                "sources": [
                    {
                        "content": doc.content,
                        "metadata": doc.metadata
                    }
                    for doc in retrieved_docs
                ]
            }
        else:
            return response
    
    def clear(self) -> bool:
        """Clear all documents from the vector store."""
        return self.vector_store.clear()
    
    def count(self, filter: Optional[Dict[str, Any]] = None) -> int:
        """Count documents in the vector store."""
        return self.vector_store.count(filter=filter)