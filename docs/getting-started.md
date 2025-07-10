# Getting Started with FI-Chat RAG

This guide will help you get started with the FI-Chat RAG framework.

## Installation

### Quick Install

```bash
pip install fichat-rag
```

### Install from Source

```bash
git clone https://github.com/yourusername/fichat-rag.git
cd fichat-rag
pip install -e .
```

### Install with Extras

```bash
# For PDF support
pip install fichat-rag[pdf]

# For all features
pip install fichat-rag[all]
```

## Quick Start

### 1. Basic Usage

```python
from fichat_rag import RAG, Config

# Initialize with default configuration
rag = RAG()

# Add documents
rag.add_documents(["document.txt", "report.pdf"])

# Query
response = rag.query("What are the main findings?")
print(response)
```

### 2. Using Ollama (Local LLM)

```python
from fichat_rag import RAG, Config, OllamaLLM

# Configure for Ollama
config = Config(
    llm_provider="ollama",
    llm_model="llama2",
    vector_store_type="memory"
)

rag = RAG(config=config)
```

### 3. Using PostgreSQL Vector Store

```python
config = Config(
    vector_store_type="postgres",
    vector_store_config={
        "host": "localhost",
        "port": 5432,
        "database": "fichat_rag",
        "user": "rag_user",
        "password": "rag_pass"
    }
)

rag = RAG(config=config)
```

## Docker Setup

### 1. Start All Services

```bash
docker-compose up -d
```

This starts:
- Ollama (local LLM)
- PostgreSQL with pgvector
- MinIO (optional object storage)
- RAG API service

### 2. Pull Ollama Model

```bash
docker exec -it fichat-ollama ollama pull llama2
```

### 3. Test the Setup

```bash
cd examples
python ollama_local.py
```

## Environment Configuration

Create a `.env` file based on `.env.example`:

```bash
cp .env.example .env
```

Key configurations:
- `LLM_PROVIDER`: Choose between "ollama" or "openai"
- `VECTOR_STORE_TYPE`: Choose between "postgres", "memory", or "chromadb"
- `EMBEDDING_MODEL`: Sentence transformer model to use

## Next Steps

- Check out the [examples](../examples) directory
- Read the [API Reference](./api-reference.md)
- Learn about [Advanced Usage](./advanced-usage.md)