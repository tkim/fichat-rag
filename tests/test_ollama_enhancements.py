"""Tests for enhanced Ollama integration features."""

import pytest
from unittest.mock import Mock, patch, MagicMock
import json

from src.core.llm import OllamaLLM, LLMResponse
from src.core.llm_utils import LRUCache, ConnectionPool, PromptOptimizer


class TestOllamaEnhancements:
    """Test enhanced Ollama features."""
    
    def test_model_specific_parameters(self):
        """Test model-specific parameter initialization."""
        # Test Llama2 parameters
        llm = OllamaLLM(model="llama2")
        assert llm.context_window == 4096
        assert "temperature" in llm.default_params
        assert llm.default_params["temperature"] == 0.7
        
        # Test Mistral parameters
        llm = OllamaLLM(model="mistral")
        assert llm.context_window == 8192
        assert llm.default_params.get("top_p") == 0.95
        
        # Test CodeLlama parameters
        llm = OllamaLLM(model="codellama")
        assert llm.context_window == 16384
        assert llm.default_params["temperature"] == 0.1
    
    @patch('requests.get')
    def test_list_models(self, mock_get):
        """Test listing available models."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "models": [
                {
                    "name": "llama2",
                    "size": 3826793472,
                    "digest": "abc123",
                    "modified_at": "2024-01-01T00:00:00Z"
                },
                {
                    "name": "mistral",
                    "size": 4113993472,
                    "digest": "def456",
                    "modified_at": "2024-01-02T00:00:00Z"
                }
            ]
        }
        mock_get.return_value = mock_response
        
        llm = OllamaLLM()
        models = llm.list_models()
        
        assert len(models) == 2
        assert models[0]["name"] == "llama2"
        assert models[1]["name"] == "mistral"
        assert "size" in models[0]
        assert "digest" in models[0]
    
    @patch('requests.post')
    def test_pull_model(self, mock_post):
        """Test pulling a model."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.iter_lines.return_value = [
            b'{"status": "pulling manifest"}',
            b'{"status": "downloading", "completed": 1000, "total": 5000}',
            b'{"status": "success"}'
        ]
        mock_post.return_value.__enter__ = Mock(return_value=mock_response)
        mock_post.return_value.__exit__ = Mock(return_value=None)
        
        llm = OllamaLLM()
        result = llm.pull_model("llama2", stream_progress=True)
        
        assert result["status"] == "success"
        assert result["model"] == "llama2"
    
    @patch('requests.post')
    def test_model_info(self, mock_post):
        """Test getting model information."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "modelinfo": {
                "general.architecture": "llama",
                "general.parameter_size": "7B"
            },
            "parameters": "temperature 0.7",
            "template": "{{ .Prompt }}",
            "details": {"format": "gguf"},
            "license": "LLAMA 2 LICENSE"
        }
        mock_post.return_value = mock_response
        
        llm = OllamaLLM()
        info = llm.model_info("llama2")
        
        assert info["name"] == "llama2"
        assert "model_info" in info
        assert "parameters" in info
        assert "license" in info
    
    def test_context_window_management(self):
        """Test context window detection and management."""
        # Test known model
        llm = OllamaLLM(model="mixtral")
        assert llm.get_context_window() == 32768
        
        # Test model variant
        llm = OllamaLLM(model="llama2:13b")
        assert llm.get_context_window() == 4096
        
        # Test unknown model with custom context
        llm = OllamaLLM(model="custom-model", context_window=8192)
        assert llm.get_context_window() == 8192


class TestLLMUtils:
    """Test LLM utility classes."""
    
    def test_lru_cache(self):
        """Test LRU cache functionality."""
        cache = LRUCache(max_size=3, ttl_seconds=3600)
        
        # Test set and get
        cache.set("prompt1", "response1", temperature=0.7)
        assert cache.get("prompt1", temperature=0.7) == "response1"
        
        # Test cache miss with different params
        assert cache.get("prompt1", temperature=0.9) is None
        
        # Test LRU eviction
        cache.set("prompt2", "response2")
        cache.set("prompt3", "response3")
        cache.set("prompt4", "response4")  # Should evict prompt1
        
        assert cache.get("prompt1", temperature=0.7) is None
        assert cache.get("prompt4") == "response4"
    
    def test_prompt_optimizer(self):
        """Test prompt optimization for different models."""
        # Test Llama2 format
        optimized = PromptOptimizer.optimize_prompt(
            "What is RAG?",
            "llama2",
            system_prompt="You are a helpful assistant."
        )
        assert "[INST]" in optimized
        assert "<<SYS>>" in optimized
        
        # Test Phi format
        optimized = PromptOptimizer.optimize_prompt(
            "What is RAG?",
            "phi"
        )
        assert "Instruct:" in optimized
        assert "Output:" in optimized
        
        # Test context window truncation
        long_prompt = "x" * 10000
        truncated = PromptOptimizer.truncate_to_context_window(
            long_prompt,
            context_window=2048,
            reserve_tokens=500
        )
        assert len(truncated) < len(long_prompt)
        assert "...[truncated]..." in truncated
    
    @patch('requests.Session')
    def test_connection_pool(self, mock_session):
        """Test connection pooling."""
        pool = ConnectionPool(pool_connections=5, pool_maxsize=5)
        
        # Test that session is configured correctly
        assert mock_session.called
        
        # Test request methods
        pool.get("http://example.com")
        pool.post("http://example.com", json={"test": "data"})
        
        assert pool.session.get.called
        assert pool.session.post.called


class TestOllamaWithCache:
    """Test Ollama with caching enabled."""
    
    @patch('requests.post')
    def test_cache_hit(self, mock_post):
        """Test cache hit scenario."""
        # First request
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "response": "RAG is Retrieval-Augmented Generation",
            "model": "llama2",
            "prompt_eval_count": 10,
            "eval_count": 20
        }
        mock_post.return_value = mock_response
        
        llm = OllamaLLM(enable_cache=True)
        
        # First call - should hit API
        response1 = llm.generate("What is RAG?")
        assert mock_post.call_count == 1
        assert response1.content == "RAG is Retrieval-Augmented Generation"
        
        # Second call - should hit cache
        response2 = llm.generate("What is RAG?")
        assert mock_post.call_count == 1  # No additional API call
        assert response2.content == response1.content
        
        # Call with no_cache - should hit API again
        response3 = llm.generate("What is RAG?", no_cache=True)
        assert mock_post.call_count == 2


class TestVectorStores:
    """Test new vector store implementations."""
    
    def test_sqlite_vector_store_init(self):
        """Test SQLite vector store initialization."""
        from src.storage.sqlite_vector import SQLiteVectorStore
        
        # Test with in-memory database
        store = SQLiteVectorStore(db_path=":memory:", embedding_dim=384)
        
        # Test basic operations
        ids = store.add(
            texts=["Test document"],
            embeddings=[[0.1] * 384],
            metadatas=[{"source": "test"}]
        )
        
        assert len(ids) == 1
        
        # Test search
        results = store.search([0.1] * 384, k=1)
        assert len(results) == 1
        assert results[0]["content"] == "Test document"
    
    @patch('chromadb.PersistentClient')
    def test_chroma_vector_store_init(self, mock_client):
        """Test ChromaDB vector store initialization."""
        from src.storage.chroma_vector import ChromaVectorStore
        
        # Mock collection
        mock_collection = Mock()
        mock_client.return_value.get_collection.return_value = mock_collection
        
        store = ChromaVectorStore(
            persist_directory="./test_chroma",
            collection_name="test"
        )
        
        # Test add operation
        mock_collection.add.return_value = None
        ids = store.add(
            texts=["Test document"],
            embeddings=[[0.1] * 384],
            metadatas=[{"source": "test"}]
        )
        
        assert mock_collection.add.called
        assert len(ids) == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])