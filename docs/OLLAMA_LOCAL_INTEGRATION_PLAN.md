# fiChat-RAG: Ollama and Local Database Integration Plan

## Executive Summary

fiChat-RAG already supports Ollama as the default LLM provider with excellent abstraction layers. This document outlines a comprehensive plan to enhance the existing integration and add more local database options for fully offline RAG deployments.

## Table of Contents

1. [Current State Analysis](#current-state-analysis)
2. [Architecture Overview](#architecture-overview)
3. [Implementation Phases](#implementation-phases)
4. [Configuration Examples](#configuration-examples)
5. [Testing Strategy](#testing-strategy)
6. [Migration Guide](#migration-guide)
7. [Monitoring & Optimization](#monitoring--optimization)

## Current State Analysis

### ✅ Already Implemented

- **Full Ollama Integration**: Default LLM provider with complete API support
- **PostgreSQL with pgvector**: Production-ready vector storage
- **In-memory Vector Store**: For development and testing
- **Docker Compose Setup**: Including Ollama service configuration
- **Abstraction Layers**: Clean interfaces for LLM, embeddings, and vector stores

### 🔄 Enhancement Opportunities

- Add more local vector database options (SQLite, ChromaDB, DuckDB, Qdrant)
- Optimize for different Ollama models (Llama 2, Mistral, CodeLlama, Phi-2)
- Add model management features
- Improve local deployment options

## Architecture Overview

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│   Ollama Models │     │  Local Embeddings│     │ Local Databases │
├─────────────────┤     ├──────────────────┤     ├─────────────────┤
│ • llama2        │     │ • MiniLM-L6      │     │ • SQLite-VSS    │
│ • mistral       │     │ • all-mpnet-base │     │ • ChromaDB      │
│ • codellama     │     │ • e5-small       │     │ • DuckDB        │
│ • phi-2         │     └──────────────────┘     │ • Qdrant Local  │
└─────────────────┘                              │ • PostgreSQL    │
         │                      │                 └─────────────────┘
         │                      │                          │
         └──────────────────────┴──────────────────────────┘
                                │
                    ┌───────────────────────┐
                    │   fiChat-RAG Core     │
                    ├───────────────────────┤
                    │ • BaseLLM Interface   │
                    │ • BaseVectorStore     │
                    │ • BaseEmbeddings      │
                    │ • RAG Orchestrator    │
                    └───────────────────────┘
```

## Implementation Phases

### Phase 1: Enhanced Ollama Integration (Week 1)

#### 1.1 Model Management Features

```python
# Enhanced src/core/llm.py
class OllamaLLM(BaseLLM):
    def list_models(self) -> List[str]:
        """List available Ollama models"""
        url = f"{self.base_url}/api/tags"
        response = requests.get(url)
        return [model["name"] for model in response.json()["models"]]
        
    def pull_model(self, model_name: str):
        """Pull a model from Ollama registry"""
        url = f"{self.base_url}/api/pull"
        requests.post(url, json={"name": model_name})
        
    def model_info(self, model_name: str) -> Dict:
        """Get model information"""
        url = f"{self.base_url}/api/show"
        response = requests.post(url, json={"name": model_name})
        return response.json()
```

#### 1.2 Model-Specific Optimizations

- **Llama 2**: Optimized for instruction following
- **Mistral**: Best for general-purpose tasks
- **CodeLlama**: Specialized for code generation
- **Phi-2**: Lightweight option for resource-constrained environments

#### 1.3 Performance Enhancements

- Connection pooling for concurrent requests
- Response caching for repeated queries
- Batch processing support
- Context window management

### Phase 2: Local Database Implementations (Week 2-3)

#### 2.1 SQLite with Vector Extensions

```python
# src/storage/sqlite_vector.py
class SQLiteVectorStore(BaseVectorStore):
    """SQLite with sqlite-vss for vector similarity search"""
    
    def __init__(self, db_path: str = "fichat.db", embedding_dim: int = 384):
        self.db_path = db_path
        self.embedding_dim = embedding_dim
        self._init_db()
    
    def _init_db(self):
        self.conn = sqlite3.connect(self.db_path)
        self.conn.enable_load_extension(True)
        self.conn.load_extension("vector0")
        self.conn.load_extension("vss0")
        
        # Create tables
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS documents (
                id TEXT PRIMARY KEY,
                content TEXT,
                metadata TEXT,
                embedding BLOB
            )
        """)
        
        # Create vector index
        self.conn.execute("""
            CREATE VIRTUAL TABLE IF NOT EXISTS vss_documents USING vss0(
                embedding(384)
            )
        """)
```

**Pros:**
- Single file deployment
- No server required
- Good for small-medium datasets (<1M vectors)
- Minimal dependencies

**Cons:**
- Limited concurrent write performance
- No built-in sharding

#### 2.2 ChromaDB Local Implementation

```python
# src/storage/chroma_vector.py
class ChromaVectorStore(BaseVectorStore):
    """ChromaDB in persistent local mode"""
    
    def __init__(self, persist_directory: str = "./chroma_db"):
        import chromadb
        self.client = chromadb.PersistentClient(path=persist_directory)
        self.collection = self.client.get_or_create_collection(
            name="fichat",
            metadata={"hnsw:space": "cosine"}
        )
```

**Pros:**
- Purpose-built for embeddings
- Rich metadata filtering
- Built-in persistence
- Good documentation

**Cons:**
- Larger storage footprint
- Python-only implementation

#### 2.3 DuckDB Integration

```python
# src/storage/duckdb_vector.py
class DuckDBVectorStore(BaseVectorStore):
    """DuckDB with vector similarity extensions"""
    
    def __init__(self, db_path: str = "fichat.duckdb"):
        import duckdb
        self.conn = duckdb.connect(db_path)
        self.conn.execute("INSTALL vss; LOAD vss;")
        
        # Create tables with vector column
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS documents (
                id VARCHAR PRIMARY KEY,
                content TEXT,
                metadata JSON,
                embedding FLOAT[384]
            )
        """)
```

**Pros:**
- Excellent analytical capabilities
- Columnar storage (efficient)
- SIMD optimizations
- Good for hybrid workloads

**Cons:**
- Vector support is newer
- Less mature for pure vector search

#### 2.4 Qdrant Local Mode

```python
# src/storage/qdrant_vector.py
class QdrantVectorStore(BaseVectorStore):
    """Qdrant in local/file mode"""
    
    def __init__(self, path: str = "./qdrant_data"):
        from qdrant_client import QdrantClient
        from qdrant_client.models import Distance, VectorParams
        
        self.client = QdrantClient(path=path)  # Local file mode
        
        # Create collection if not exists
        try:
            self.client.create_collection(
                collection_name="fichat",
                vectors_config=VectorParams(
                    size=384,
                    distance=Distance.COSINE
                )
            )
        except:
            pass  # Collection already exists
```

**Pros:**
- Production-ready performance
- Advanced filtering capabilities
- Rust-based (fast)
- Excellent scaling characteristics

**Cons:**
- Larger binary size
- More complex setup

### Phase 3: Deployment Configurations (Week 4)

#### 3.1 Minimal Single-File Deployment

```yaml
# docker-compose.local-minimal.yml
version: '3.8'

services:
  fichat-minimal:
    build: .
    container_name: fichat-minimal
    ports:
      - "8000:8000"
    environment:
      - LLM_PROVIDER=ollama
      - LLM_MODEL=phi-2
      - OLLAMA_BASE_URL=http://host.docker.internal:11434
      - VECTOR_STORE_TYPE=sqlite
      - EMBEDDING_MODEL=all-MiniLM-L6-v2
      - EMBEDDING_DEVICE=cpu
    volumes:
      - ./data:/app/data
      - ./documents:/app/documents
```

#### 3.2 Full-Featured Local Setup

```yaml
# docker-compose.local-full.yml
version: '3.8'

services:
  ollama:
    image: ollama/ollama:latest
    container_name: fichat-ollama
    ports:
      - "11434:11434"
    volumes:
      - ollama_data:/root/.ollama
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu]
              
  chromadb:
    image: chromadb/chroma:latest
    container_name: fichat-chromadb
    ports:
      - "8001:8000"
    volumes:
      - chroma_data:/chroma/chroma
      
  qdrant:
    image: qdrant/qdrant:latest
    container_name: fichat-qdrant
    ports:
      - "6333:6333"
    volumes:
      - qdrant_data:/qdrant/storage
      
  fichat-rag:
    build: .
    container_name: fichat-rag
    ports:
      - "8000:8000"
    environment:
      - LLM_PROVIDER=ollama
      - LLM_MODEL=${OLLAMA_MODEL:-llama2}
      - OLLAMA_BASE_URL=http://ollama:11434
      - VECTOR_STORE_TYPE=${DB_TYPE:-chromadb}
      - CHROMADB_HOST=chromadb
      - CHROMADB_PORT=8000
      - QDRANT_HOST=qdrant
      - QDRANT_PORT=6333
    depends_on:
      - ollama
      - chromadb
      - qdrant

volumes:
  ollama_data:
  chroma_data:
  qdrant_data:
```

### Phase 4: Model & Database Selection UI (Week 5)

#### 4.1 Interactive Configuration Script

```python
# scripts/configure_local.py
import os
import subprocess
from typing import Dict, List

class LocalSetupWizard:
    """Interactive setup wizard for local deployment"""
    
    def __init__(self):
        self.config = {}
        
    def detect_hardware(self) -> Dict:
        """Detect available hardware resources"""
        return {
            "gpu_available": self._check_gpu(),
            "ram_gb": self._get_ram(),
            "cpu_cores": os.cpu_count(),
            "disk_space_gb": self._get_disk_space()
        }
    
    def recommend_config(self, hardware: Dict) -> Dict:
        """Recommend configuration based on hardware"""
        if hardware["gpu_available"]:
            return {
                "llm_model": "llama2:13b",
                "vector_db": "qdrant",
                "embedding_model": "e5-base",
                "embedding_device": "cuda"
            }
        elif hardware["ram_gb"] >= 16:
            return {
                "llm_model": "mistral",
                "vector_db": "chromadb",
                "embedding_model": "all-mpnet-base-v2",
                "embedding_device": "cpu"
            }
        else:
            return {
                "llm_model": "phi-2",
                "vector_db": "sqlite",
                "embedding_model": "all-MiniLM-L6-v2",
                "embedding_device": "cpu"
            }
    
    def run(self):
        """Run the configuration wizard"""
        print("🚀 fiChat-RAG Local Setup Wizard")
        print("="*50)
        
        # Detect hardware
        hardware = self.detect_hardware()
        print(f"\n📊 Detected Hardware:")
        print(f"  - GPU: {'Yes' if hardware['gpu_available'] else 'No'}")
        print(f"  - RAM: {hardware['ram_gb']}GB")
        print(f"  - CPU Cores: {hardware['cpu_cores']}")
        
        # Get recommendations
        recommended = self.recommend_config(hardware)
        print(f"\n💡 Recommended Configuration:")
        for key, value in recommended.items():
            print(f"  - {key}: {value}")
        
        # Generate config files
        self._generate_env_file(recommended)
        self._generate_docker_compose(recommended)
        
        print("\n✅ Configuration complete!")
        print("Run 'docker-compose -f docker-compose.local.yml up' to start")
```

## Configuration Examples

### 1. Minimal Local Setup (Low Resources)

```bash
# .env.local-minimal
LLM_PROVIDER=ollama
LLM_MODEL=phi-2
LLM_TEMPERATURE=0.7
LLM_MAX_TOKENS=1000

VECTOR_STORE_TYPE=sqlite
SQLITE_DB_PATH=./data/fichat.db

EMBEDDING_MODEL=all-MiniLM-L6-v2
EMBEDDING_DEVICE=cpu

CHUNK_SIZE=500
CHUNK_OVERLAP=50
```

### 2. Balanced Setup (Medium Resources)

```bash
# .env.local-balanced
LLM_PROVIDER=ollama
LLM_MODEL=mistral
LLM_TEMPERATURE=0.7
LLM_MAX_TOKENS=2000

VECTOR_STORE_TYPE=chromadb
CHROMADB_PERSIST_DIR=./data/chroma

EMBEDDING_MODEL=all-mpnet-base-v2
EMBEDDING_DEVICE=cpu

CHUNK_SIZE=1000
CHUNK_OVERLAP=100
```

### 3. Performance Setup (High Resources)

```bash
# .env.local-performance
LLM_PROVIDER=ollama
LLM_MODEL=llama2:13b
LLM_TEMPERATURE=0.7
LLM_MAX_TOKENS=4000

VECTOR_STORE_TYPE=qdrant
QDRANT_PATH=./data/qdrant

EMBEDDING_MODEL=e5-large
EMBEDDING_DEVICE=cuda

CHUNK_SIZE=1500
CHUNK_OVERLAP=150
```

## Testing Strategy

### 1. Unit Tests

```python
# tests/test_vector_stores.py
import pytest
from src.storage import SQLiteVectorStore, ChromaVectorStore, DuckDBVectorStore

@pytest.mark.parametrize("store_class", [
    SQLiteVectorStore,
    ChromaVectorStore,
    DuckDBVectorStore
])
def test_vector_store_operations(store_class, tmp_path):
    """Test basic CRUD operations for each vector store"""
    store = store_class(path=str(tmp_path / "test_db"))
    
    # Test add
    doc_id = store.add(
        texts=["Hello world"],
        embeddings=[[0.1] * 384],
        metadatas=[{"source": "test"}]
    )[0]
    
    # Test search
    results = store.search([0.1] * 384, k=1)
    assert len(results) == 1
    assert results[0]["id"] == doc_id
```

### 2. Integration Tests

```python
# tests/test_ollama_integration.py
def test_ollama_models():
    """Test different Ollama models"""
    models = ["phi-2", "mistral", "llama2"]
    
    for model in models:
        llm = OllamaLLM(model=model)
        response = llm.generate("Hello, world!")
        assert response.content
        assert response.model == model
```

### 3. Performance Benchmarks

```python
# benchmarks/vector_store_performance.py
import time
import numpy as np

def benchmark_vector_stores(n_documents=10000, embedding_dim=384):
    """Benchmark different vector stores"""
    results = {}
    
    for store_name, store_class in VECTOR_STORES.items():
        store = store_class()
        
        # Generate random data
        texts = [f"Document {i}" for i in range(n_documents)]
        embeddings = np.random.rand(n_documents, embedding_dim)
        
        # Benchmark insertion
        start = time.time()
        store.add(texts=texts, embeddings=embeddings)
        insert_time = time.time() - start
        
        # Benchmark search
        query_embedding = np.random.rand(embedding_dim)
        start = time.time()
        results = store.search(query_embedding, k=10)
        search_time = time.time() - start
        
        results[store_name] = {
            "insert_time": insert_time,
            "search_time": search_time,
            "docs_per_second": n_documents / insert_time
        }
    
    return results
```

## Migration Guide

### From Cloud to Local

#### 1. Export Data from Cloud Services

```python
# scripts/export_from_cloud.py
def export_from_openai():
    """Export embeddings from OpenAI"""
    # Implementation for exporting data
    
def export_from_pinecone():
    """Export vectors from Pinecone"""
    # Implementation for exporting data
```

#### 2. Import to Local Database

```python
# scripts/import_to_local.py
def import_to_local(export_file: str, target_db: str):
    """Import exported data to local database"""
    data = load_export(export_file)
    
    if target_db == "sqlite":
        store = SQLiteVectorStore()
    elif target_db == "chromadb":
        store = ChromaVectorStore()
    # ... other databases
    
    store.add(
        texts=data["texts"],
        embeddings=data["embeddings"],
        metadatas=data["metadatas"]
    )
```

#### 3. Configuration Update

```bash
# Update .env file
sed -i 's/LLM_PROVIDER=openai/LLM_PROVIDER=ollama/g' .env
sed -i 's/VECTOR_STORE_TYPE=pinecone/VECTOR_STORE_TYPE=chromadb/g' .env
```

## Monitoring & Optimization

### 1. Resource Monitoring Dashboard

```python
# src/monitoring/dashboard.py
import psutil
import GPUtil

class ResourceMonitor:
    """Monitor system resources"""
    
    def get_metrics(self):
        return {
            "cpu_percent": psutil.cpu_percent(),
            "memory_percent": psutil.virtual_memory().percent,
            "disk_usage": psutil.disk_usage('/').percent,
            "gpu_usage": self._get_gpu_usage()
        }
    
    def _get_gpu_usage(self):
        try:
            gpus = GPUtil.getGPUs()
            return [{"id": gpu.id, "usage": gpu.load * 100} for gpu in gpus]
        except:
            return []
```

### 2. Performance Optimization Tips

#### Query Optimization
- Use appropriate chunk sizes based on model context window
- Implement query caching for repeated questions
- Use hybrid search (vector + keyword) for better results

#### Resource Optimization
- Enable GPU acceleration when available
- Use quantized models for lower memory usage
- Implement connection pooling for database connections

#### Storage Optimization
- Regular index optimization for vector databases
- Implement data compression where supported
- Use appropriate vector dimensions (384 vs 768 vs 1536)

### 3. Auto-Configuration Script

```python
# scripts/auto_optimize.py
def auto_optimize():
    """Automatically optimize configuration based on usage patterns"""
    
    # Analyze query patterns
    patterns = analyze_query_logs()
    
    # Adjust chunk size
    if patterns["avg_query_length"] > 100:
        update_config("CHUNK_SIZE", 1500)
    
    # Adjust model based on complexity
    if patterns["complexity_score"] > 0.8:
        update_config("LLM_MODEL", "llama2:13b")
    else:
        update_config("LLM_MODEL", "mistral")
    
    # Optimize vector store
    if patterns["total_documents"] > 100000:
        update_config("VECTOR_STORE_TYPE", "qdrant")
    elif patterns["total_documents"] > 10000:
        update_config("VECTOR_STORE_TYPE", "chromadb")
    else:
        update_config("VECTOR_STORE_TYPE", "sqlite")
```

## Quick Start Commands

```bash
# 1. Clone and setup
git clone https://github.com/yourusername/fichat-rag.git
cd fichat-rag

# 2. Run setup wizard
python scripts/configure_local.py

# 3. Pull Ollama model
ollama pull llama2

# 4. Start services
docker-compose -f docker-compose.local.yml up

# 5. Test the setup
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What is RAG?"}'
```

## Troubleshooting

### Common Issues

1. **Ollama Connection Error**
   ```bash
   # Ensure Ollama is running
   ollama serve
   
   # Check connection
   curl http://localhost:11434/api/tags
   ```

2. **SQLite Vector Extension Not Found**
   ```bash
   # Install sqlite-vss
   pip install sqlite-vss
   
   # Or use pre-built binary
   wget https://github.com/asg017/sqlite-vss/releases/download/v0.1.0/sqlite-vss-linux-amd64.tar.gz
   ```

3. **GPU Not Detected**
   ```bash
   # Check NVIDIA drivers
   nvidia-smi
   
   # Install CUDA toolkit if needed
   ```

## Future Enhancements

1. **Additional Local Models**
   - Support for GGUF format models
   - Integration with llama.cpp
   - Support for specialized models (medical, legal, etc.)

2. **Advanced Features**
   - Multi-modal support (images, audio)
   - Incremental indexing
   - Distributed vector search

3. **Developer Tools**
   - VS Code extension
   - CLI management tool
   - Performance profiler

## Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.