# fiChat RAG Framework

A standalone, production-ready Retrieval-Augmented Generation (RAG) framework extracted from FI-Chat. This framework provides a flexible and efficient solution for building RAG applications with support for local LLMs via Ollama.

## Features

- 🚀 **Production-Ready**: Battle-tested components from fiChat
- 🔧 **Flexible LLM Support**: Built-in Ollama integration, and custom providers
- 📊 **Multiple Vector Stores**: PostgreSQL with pgvector, ChromaDB, Qdrant
- 🔍 **Hybrid Search**: Combines vector similarity and keyword search
- 📄 **Advanced Document Processing**: Semantic chunking with configurable strategies
- 🌐 **Web Scraping**: Firecrawl integration for web content extraction
- 🎯 **Reranking**: Cross-encoder based reranking for improved relevance
- 🐳 **Docker Support**: Easy local setup with Docker Compose
- ⚡ **Async Operations**: High-performance async/await patterns

## System Requirements

### Operating System Support

FI-Chat RAG is a cross-platform framework that runs on:
- **Linux** (Ubuntu 20.04+, Debian 10+, RHEL 8+, and other modern distributions)
- **Windows** (Windows 10/11 with native Python or WSL2)
- **macOS** (macOS 10.15 Catalina or later, including Apple Silicon)

### Python Requirements
- **Python 3.8 or higher** (3.8, 3.9, 3.10, 3.11 tested)
- pip package manager

### Core Dependencies
- **Vector Database** (choose one):
  - PostgreSQL 15+ with pgvector extension
  - ChromaDB
  - Qdrant
  - In-memory store (for development/testing)
- **PyTorch** 2.0+ (automatically installed, GPU support optional)
- **NLTK** data files (automatically downloaded on first use)

### Optional Dependencies
- **Ollama** (for local LLM support) - requires Docker or native installation
- **CUDA** 11.7+ (for GPU acceleration with PyTorch)
- **Docker & Docker Compose** (for containerized deployment)
- **MinIO** (for S3-compatible object storage)

### Hardware Requirements
- **Minimum**: 8GB RAM, 20GB disk space
- **Recommended**: 16GB+ RAM, 50GB+ disk space, NVIDIA GPU with 8GB+ VRAM
- **Production**: 32GB+ RAM, 100GB+ SSD, NVIDIA GPU with 16GB+ VRAM

## Platform-Specific Setup Instructions

### Windows 11 Setup

#### Option 1: Native Windows Installation

1. **Install Python 3.8+**:
   ```powershell
   # Download from python.org or use Windows Package Manager
   winget install Python.Python.3.11
   ```

2. **Install PostgreSQL with pgvector**:
   ```powershell
   # Download PostgreSQL installer from postgresql.org
   # After installation, install pgvector extension:
   git clone https://github.com/pgvector/pgvector.git
   cd pgvector
   nmake /f Makefile.win
   nmake /f Makefile.win install
   ```

3. **Install FI-Chat RAG**:
   ```powershell
   # Create virtual environment
   python -m venv fichat-env
   .\fichat-env\Scripts\activate

   # Install the package
   pip install fichat-rag[all]
   ```

4. **Install Ollama (optional)**:
   ```powershell
   # Download Ollama for Windows from ollama.ai
   # Or use WSL2 for better compatibility
   ```

#### Option 2: WSL2 Installation (Recommended)

1. **Enable WSL2**:
   ```powershell
   # Run as Administrator
   wsl --install
   wsl --set-default-version 2
   ```

2. **Install Ubuntu from Microsoft Store**

3. **Follow Linux installation steps** within WSL2:
   ```bash
   # Update packages
   sudo apt update && sudo apt upgrade

   # Install Python and dependencies
   sudo apt install python3.10 python3-pip python3-venv
   
   # Install PostgreSQL
   sudo apt install postgresql postgresql-contrib
   sudo apt install postgresql-15-pgvector

   # Install FI-Chat RAG
   pip install fichat-rag[all]
   ```

### macOS Setup

#### Intel Mac

1. **Install Homebrew** (if not already installed):
   ```bash
   /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
   ```

2. **Install Python and PostgreSQL**:
   ```bash
   brew install python@3.11
   brew install postgresql@15
   brew install pgvector
   ```

3. **Install FI-Chat RAG**:
   ```bash
   # Create virtual environment
   python3 -m venv fichat-env
   source fichat-env/bin/activate

   # Install the package
   pip install fichat-rag[all]
   ```

4. **Install Ollama** (optional):
   ```bash
   brew install ollama
   ollama serve
   ```

#### Apple Silicon (M1/M2/M3)

1. **Install Rosetta 2** (if needed):
   ```bash
   softwareupdate --install-rosetta
   ```

2. **Install dependencies**:
   ```bash
   # Use Homebrew for ARM64
   brew install python@3.11
   brew install postgresql@15
   brew install pgvector

   # For PyTorch with Metal Performance Shaders
   pip install torch torchvision torchaudio
   ```

3. **Install FI-Chat RAG**:
   ```bash
   # Ensure using ARM64 Python
   python3 -m venv fichat-env
   source fichat-env/bin/activate
   
   # Install with Apple Silicon optimizations
   pip install fichat-rag[all]
   ```

### Server Deployment

#### Docker Deployment (Recommended)

1. **Prerequisites**:
   ```bash
   # Install Docker and Docker Compose
   curl -fsSL https://get.docker.com -o get-docker.sh
   sudo sh get-docker.sh
   sudo usermod -aG docker $USER
   
   # Install Docker Compose
   sudo curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
   sudo chmod +x /usr/local/bin/docker-compose
   ```

2. **Deploy with Docker Compose**:
   ```bash
   # Clone the repository
   git clone https://github.com/yourusername/fichat-rag.git
   cd fichat-rag

   # Configure environment
   cp .env.example .env
   # Edit .env with your settings

   # Start all services
   docker-compose up -d

   # Check status
   docker-compose ps
   ```

3. **Configure for production**:
   ```yaml
   # docker-compose.prod.yml
   version: '3.8'
   services:
     rag-api:
       image: fichat-rag:latest
       environment:
         - ENVIRONMENT=production
         - LOG_LEVEL=info
       deploy:
         replicas: 3
         resources:
           limits:
             cpus: '2'
             memory: 4G
   ```

#### Kubernetes Deployment

1. **Create deployment manifests**:
   ```yaml
   # fichat-rag-deployment.yaml
   apiVersion: apps/v1
   kind: Deployment
   metadata:
     name: fichat-rag
   spec:
     replicas: 3
     selector:
       matchLabels:
         app: fichat-rag
     template:
       metadata:
         labels:
           app: fichat-rag
       spec:
         containers:
         - name: rag-api
           image: fichat-rag:latest
           ports:
           - containerPort: 8000
           resources:
             requests:
               memory: "2Gi"
               cpu: "1"
             limits:
               memory: "4Gi"
               cpu: "2"
   ```

2. **Deploy to cluster**:
   ```bash
   kubectl apply -f fichat-rag-deployment.yaml
   kubectl apply -f fichat-rag-service.yaml
   kubectl apply -f fichat-rag-ingress.yaml
   ```

#### Cloud Deployment

**AWS EC2/ECS**:
```bash
# Using AWS CDK or CloudFormation
# Example: Deploy on EC2 with Auto Scaling
aws ec2 run-instances \
  --image-id ami-0abcdef1234567890 \
  --instance-type t3.xlarge \
  --user-data file://install-fichat-rag.sh
```

**Google Cloud Run**:
```bash
# Build and deploy container
gcloud run deploy fichat-rag \
  --image gcr.io/project-id/fichat-rag \
  --platform managed \
  --memory 4Gi \
  --cpu 2
```

**Azure Container Instances**:
```bash
az container create \
  --resource-group myResourceGroup \
  --name fichat-rag \
  --image fichat-rag:latest \
  --cpu 2 \
  --memory 4
```

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

## Dependencies and Package Details

### Core Python Packages

```txt
# Core dependencies (automatically installed)
numpy>=1.24.0              # Numerical operations
requests>=2.31.0           # HTTP client
aiohttp>=3.9.0            # Async HTTP client
pydantic>=2.0.0           # Data validation
pyyaml>=6.0               # YAML parsing

# ML/AI dependencies
torch>=2.0.0              # PyTorch for embeddings and ML
sentence-transformers>=2.2.0  # Embedding models
tiktoken>=0.5.0           # Token counting
nltk>=3.8.0               # Text processing
chardet>=5.0.0            # Character encoding detection

# Vector store clients
psycopg2-binary>=2.9.0    # PostgreSQL client
pgvector>=0.2.0           # pgvector extension
chromadb>=0.4.0           # ChromaDB client (optional)
qdrant-client>=1.7.0      # Qdrant client (optional)

# Optional features
PyPDF2>=3.0.0             # PDF processing (install with [pdf])
pdfplumber>=0.10.0        # Advanced PDF extraction (install with [pdf])
firecrawl-py>=0.0.14      # Web scraping (install with [web])

# Development dependencies
pytest>=7.0.0             # Testing framework
black>=23.0.0             # Code formatter
flake8>=6.0.0             # Linter
mypy>=1.0.0               # Type checker
```

### Installation Options

```bash
# Basic installation
pip install fichat-rag

# With PDF support
pip install fichat-rag[pdf]

# With web scraping support
pip install fichat-rag[web]

# With all optional features
pip install fichat-rag[all]

# Development installation
pip install fichat-rag[dev]

# From source
git clone https://github.com/yourusername/fichat-rag.git
cd fichat-rag
pip install -e .[all,dev]
```

### Environment Variables

```bash
# LLM Configuration
export LLM_PROVIDER=ollama          # or openai, custom
export LLM_MODEL=llama2             # model name
export OPENAI_API_KEY=sk-...        # if using OpenAI

# Database Configuration
export POSTGRES_HOST=localhost
export POSTGRES_PORT=5432
export POSTGRES_DB=fichat_rag
export POSTGRES_USER=raguser
export POSTGRES_PASSWORD=secure_password

# Vector Store Selection
export VECTOR_STORE=postgres        # or chromadb, qdrant, memory

# Web Scraping
export FIRECRAWL_API_KEY=fc-...     # for web scraping

# Optional Services
export MINIO_ENDPOINT=localhost:9000
export MINIO_ACCESS_KEY=minioadmin
export MINIO_SECRET_KEY=minioadmin

# Performance Settings
export RAG_BATCH_SIZE=32
export RAG_MAX_WORKERS=4
export RAG_CACHE_SIZE=1000
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
