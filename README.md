# fiChat RAG Framework

A standalone, production-ready Retrieval-Augmented Generation (RAG) framework extracted from FI-Chat. This framework provides a flexible and efficient solution for building RAG applications with support for local LLMs via Ollama.

## Features

- 🚀 **Production-Ready**: Battle-tested components from FI-Chat
- 🔧 **Flexible LLM Support**: Built-in Ollama integration, OpenAI, and custom providers
- 📊 **Multiple Vector Stores**: PostgreSQL with pgvector, ChromaDB, Qdrant
- 🔍 **Hybrid Search**: Combines vector similarity and keyword search
- 📄 **Advanced Document Processing**: Semantic chunking with configurable strategies
- 🌐 **Web Scraping**: Firecrawl integration for web content extraction
- 🎯 **Reranking**: Cross-encoder based reranking for improved relevance
- 🐳 **Docker Support**: Easy local setup with Docker Compose
- ⚡ **Async Operations**: High-performance async/await patterns

## Quick Start

### Installation

```bash
pip install fichat-rag
```

### Basic Usage

```python
from fichat_rag import RAG, OllamaLLM

# Initialize RAG with Ollama
rag = RAG(
    llm=OllamaLLM(model="llama2"),
    vector_store="postgres",
    embedding_model="sentence-transformers/all-MiniLM-L6-v2"
)

# Add documents from files
rag.add_documents(["document.pdf", "data.json"])

# Add documents from web
rag.add_documents(["https://example.com/article"])

# Query
response = rag.query("What are the key findings?")
print(response)
```

### Web Scraping with Firecrawl

```python
from fichat_rag import RAG
from fichat_rag.ingestion.loaders import FirecrawlLoader

# Set your Firecrawl API key
os.environ["FIRECRAWL_API_KEY"] = "your-api-key"

# Initialize RAG
rag = RAG()

# Scrape and add web content
rag.add_documents([
    "https://docs.example.com/guide",
    "https://blog.example.com/article"
])

# Or use the loader directly for more control
loader = FirecrawlLoader()
pages = loader.crawl("https://docs.example.com", limit=10)
rag.add_documents(pages)
```

### Docker Setup

```bash
# Start all services (Ollama, PostgreSQL, pgvector)
docker-compose up -d

# Run example
python examples/basic_rag.py
```

## Architecture

The framework is organized into modular components:

- **Core**: LLM interfaces, embeddings, and configuration
- **Storage**: Vector database implementations
- **Ingestion**: Document loaders and processors
- **Retrieval**: Search, filtering, and reranking
- **Generation**: Prompt management and response formatting

## Supported LLMs

- Ollama (all models)
- OpenAI (GPT-3.5, GPT-4)
- Custom LLM providers (implement the base interface)

## Supported Vector Stores

- PostgreSQL with pgvector
- ChromaDB
- Qdrant
- In-memory (for testing)

## Documentation

See the [docs](./docs) directory for detailed documentation:

- [Getting Started](./docs/getting-started.md)
- [Configuration](./docs/configuration.md)
- [Advanced Usage](./docs/advanced-usage.md)
- [API Reference](./docs/api-reference.md)

## Examples

Check the [examples](./examples) directory for:

- Basic RAG implementation
- Ollama integration
- Multi-source retrieval
- Web scraping RAG
- Custom document processors

## Contributing

Contributions are welcome! Please read our [Contributing Guide](CONTRIBUTING.md) for details.

## License

MIT License - see [LICENSE](LICENSE) for details.

## Acknowledgments

This framework is extracted from [FI-Chat](https://github.com/yourusername/FI-Chat), a production financial chat application.
