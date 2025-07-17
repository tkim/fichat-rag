#!/usr/bin/env python3
"""Interactive configuration wizard for local fiChat-RAG deployment."""

import os
import sys
import json
import subprocess
import platform
import psutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple


class LocalSetupWizard:
    """Interactive setup wizard for local deployment."""
    
    def __init__(self):
        self.config = {}
        self.hardware = {}
        self.recommendations = {}
        
    def detect_hardware(self) -> Dict:
        """Detect available hardware resources."""
        hardware = {
            "platform": platform.system(),
            "cpu_cores": os.cpu_count(),
            "ram_gb": round(psutil.virtual_memory().total / (1024**3), 1),
            "gpu_available": self._check_gpu(),
            "disk_space_gb": self._get_disk_space()
        }
        
        self.hardware = hardware
        return hardware
    
    def _check_gpu(self) -> bool:
        """Check if GPU is available."""
        # Check for NVIDIA GPU
        try:
            result = subprocess.run(
                ["nvidia-smi"], 
                capture_output=True, 
                text=True,
                check=False
            )
            return result.returncode == 0
        except FileNotFoundError:
            pass
        
        # Check for Apple Silicon
        if platform.system() == "Darwin" and platform.processor() == "arm":
            return True
        
        return False
    
    def _get_disk_space(self) -> float:
        """Get available disk space in GB."""
        stat = psutil.disk_usage('/')
        return round(stat.free / (1024**3), 1)
    
    def check_dependencies(self) -> Dict[str, bool]:
        """Check if required dependencies are installed."""
        deps = {}
        
        # Check Docker
        deps["docker"] = self._check_command("docker", "--version")
        
        # Check Docker Compose
        deps["docker_compose"] = (
            self._check_command("docker-compose", "--version") or
            self._check_command("docker", "compose", "version")
        )
        
        # Check Ollama
        deps["ollama"] = self._check_command("ollama", "--version")
        
        # Check Python
        deps["python"] = sys.version_info >= (3, 8)
        
        # Check Git
        deps["git"] = self._check_command("git", "--version")
        
        return deps
    
    def _check_command(self, *args) -> bool:
        """Check if a command is available."""
        try:
            result = subprocess.run(
                args,
                capture_output=True,
                check=False
            )
            return result.returncode == 0
        except FileNotFoundError:
            return False
    
    def recommend_config(self, hardware: Dict) -> Dict:
        """Recommend configuration based on hardware."""
        recommendations = {}
        
        # LLM Model recommendation
        if hardware["gpu_available"] and hardware["ram_gb"] >= 16:
            recommendations["llm_model"] = "llama2:13b"
            recommendations["embedding_device"] = "cuda"
        elif hardware["ram_gb"] >= 16:
            recommendations["llm_model"] = "mistral"
            recommendations["embedding_device"] = "cpu"
        elif hardware["ram_gb"] >= 8:
            recommendations["llm_model"] = "llama2"
            recommendations["embedding_device"] = "cpu"
        else:
            recommendations["llm_model"] = "phi"
            recommendations["embedding_device"] = "cpu"
        
        # Vector database recommendation
        if hardware["ram_gb"] >= 16:
            recommendations["vector_db"] = "qdrant"
        elif hardware["ram_gb"] >= 8:
            recommendations["vector_db"] = "chromadb"
        else:
            recommendations["vector_db"] = "sqlite"
        
        # Embedding model recommendation
        if hardware["gpu_available"]:
            recommendations["embedding_model"] = "sentence-transformers/all-mpnet-base-v2"
        elif hardware["ram_gb"] >= 8:
            recommendations["embedding_model"] = "sentence-transformers/all-MiniLM-L6-v2"
        else:
            recommendations["embedding_model"] = "sentence-transformers/all-MiniLM-L6-v2"
        
        # Performance settings
        if hardware["ram_gb"] >= 16:
            recommendations["chunk_size"] = 1500
            recommendations["chunk_overlap"] = 150
            recommendations["batch_size"] = 32
        elif hardware["ram_gb"] >= 8:
            recommendations["chunk_size"] = 1000
            recommendations["chunk_overlap"] = 100
            recommendations["batch_size"] = 16
        else:
            recommendations["chunk_size"] = 500
            recommendations["chunk_overlap"] = 50
            recommendations["batch_size"] = 8
        
        self.recommendations = recommendations
        return recommendations
    
    def interactive_config(self) -> Dict:
        """Interactive configuration process."""
        print("\n🔧 Configuration Options")
        print("="*50)
        
        # LLM Model selection
        print("\n📊 Available Ollama Models:")
        models = ["llama2", "llama2:13b", "mistral", "codellama", "phi", "neural-chat"]
        for i, model in enumerate(models, 1):
            recommended = " (recommended)" if model == self.recommendations.get("llm_model") else ""
            print(f"  {i}. {model}{recommended}")
        
        while True:
            try:
                choice = input(f"\nSelect model [1-{len(models)}]: ").strip()
                if choice == "":
                    self.config["llm_model"] = self.recommendations["llm_model"]
                    break
                idx = int(choice) - 1
                if 0 <= idx < len(models):
                    self.config["llm_model"] = models[idx]
                    break
            except ValueError:
                pass
            print("Invalid choice. Please try again.")
        
        # Vector Database selection
        print("\n💾 Available Vector Databases:")
        databases = [
            ("sqlite", "SQLite with VSS - Lightweight, file-based"),
            ("chromadb", "ChromaDB - Feature-rich, easy to use"),
            ("qdrant", "Qdrant - High performance, advanced features"),
            ("postgres", "PostgreSQL with pgvector - Mature, scalable"),
            ("memory", "In-Memory - Fast but no persistence")
        ]
        
        for i, (db_type, desc) in enumerate(databases, 1):
            recommended = " (recommended)" if db_type == self.recommendations.get("vector_db") else ""
            print(f"  {i}. {db_type} - {desc}{recommended}")
        
        while True:
            try:
                choice = input(f"\nSelect database [1-{len(databases)}]: ").strip()
                if choice == "":
                    self.config["vector_db"] = self.recommendations["vector_db"]
                    break
                idx = int(choice) - 1
                if 0 <= idx < len(databases):
                    self.config["vector_db"] = databases[idx][0]
                    break
            except ValueError:
                pass
            print("Invalid choice. Please try again.")
        
        # Deployment type
        print("\n🚀 Deployment Type:")
        print("  1. Docker Compose (recommended)")
        print("  2. Local Python environment")
        
        deploy_choice = input("\nSelect deployment type [1-2] (default: 1): ").strip()
        self.config["deployment"] = "docker" if deploy_choice != "2" else "local"
        
        # Advanced options
        advanced = input("\n⚙️  Configure advanced options? [y/N]: ").strip().lower()
        if advanced == "y":
            self._configure_advanced()
        else:
            # Use recommendations for advanced settings
            self.config.update({
                "embedding_model": self.recommendations["embedding_model"],
                "embedding_device": self.recommendations["embedding_device"],
                "chunk_size": self.recommendations["chunk_size"],
                "chunk_overlap": self.recommendations["chunk_overlap"],
                "batch_size": self.recommendations["batch_size"]
            })
        
        return self.config
    
    def _configure_advanced(self):
        """Configure advanced options."""
        print("\n🔬 Advanced Configuration")
        print("-"*50)
        
        # Embedding model
        print("\n📐 Embedding Models:")
        embed_models = [
            "sentence-transformers/all-MiniLM-L6-v2",
            "sentence-transformers/all-mpnet-base-v2",
            "sentence-transformers/all-MiniLM-L12-v2",
            "BAAI/bge-small-en-v1.5",
            "BAAI/bge-base-en-v1.5"
        ]
        
        for i, model in enumerate(embed_models, 1):
            print(f"  {i}. {model}")
        
        choice = input(f"\nSelect embedding model [1-{len(embed_models)}]: ").strip()
        try:
            idx = int(choice) - 1
            if 0 <= idx < len(embed_models):
                self.config["embedding_model"] = embed_models[idx]
        except:
            self.config["embedding_model"] = self.recommendations["embedding_model"]
        
        # Performance settings
        self.config["chunk_size"] = self._get_int_input(
            "Chunk size", 
            self.recommendations["chunk_size"],
            100, 5000
        )
        
        self.config["chunk_overlap"] = self._get_int_input(
            "Chunk overlap",
            self.recommendations["chunk_overlap"],
            0, 500
        )
        
        self.config["batch_size"] = self._get_int_input(
            "Batch size",
            self.recommendations["batch_size"],
            1, 128
        )
    
    def _get_int_input(self, name: str, default: int, min_val: int, max_val: int) -> int:
        """Get integer input with validation."""
        while True:
            value = input(f"\n{name} [{min_val}-{max_val}] (default: {default}): ").strip()
            if value == "":
                return default
            try:
                int_val = int(value)
                if min_val <= int_val <= max_val:
                    return int_val
            except ValueError:
                pass
            print(f"Please enter a number between {min_val} and {max_val}")
    
    def generate_env_file(self, config: Dict):
        """Generate .env file."""
        env_content = f"""# fiChat-RAG Local Configuration
# Generated by configure_local.py

# LLM Configuration
LLM_PROVIDER=ollama
LLM_MODEL={config['llm_model']}
LLM_TEMPERATURE=0.7
LLM_MAX_TOKENS=2000

# Ollama Settings
OLLAMA_BASE_URL=http://localhost:11434

# Vector Store Configuration
VECTOR_STORE_TYPE={config['vector_db']}
"""
        
        # Add database-specific settings
        if config['vector_db'] == 'sqlite':
            env_content += """
# SQLite Settings
SQLITE_DB_PATH=./data/fichat.db
"""
        elif config['vector_db'] == 'chromadb':
            env_content += """
# ChromaDB Settings
CHROMADB_PERSIST_DIR=./data/chroma_db
CHROMADB_COLLECTION_NAME=fichat
"""
        elif config['vector_db'] == 'qdrant':
            env_content += """
# Qdrant Settings
QDRANT_PATH=./data/qdrant_data
QDRANT_COLLECTION_NAME=fichat
"""
        elif config['vector_db'] == 'postgres':
            env_content += """
# PostgreSQL Settings
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
POSTGRES_DB=fichat_rag
POSTGRES_USER=rag_user
POSTGRES_PASSWORD=rag_pass
"""
        
        # Add embedding and performance settings
        env_content += f"""
# Embeddings Configuration
EMBEDDING_MODEL={config['embedding_model']}
EMBEDDING_DEVICE={config.get('embedding_device', 'cpu')}
EMBEDDING_BATCH_SIZE={config.get('batch_size', 16)}

# Chunking Configuration
CHUNK_SIZE={config.get('chunk_size', 1000)}
CHUNK_OVERLAP={config.get('chunk_overlap', 100)}

# Retrieval Configuration
RETRIEVAL_TOP_K=5
RETRIEVAL_SEARCH_TYPE=hybrid
RETRIEVAL_RERANK=true

# Logging
LOG_LEVEL=INFO
"""
        
        # Write to file
        with open(".env.local", "w") as f:
            f.write(env_content)
        
        print(f"\n✅ Created .env.local file")
    
    def generate_docker_compose(self, config: Dict):
        """Generate appropriate docker-compose file."""
        if config.get('deployment') != 'docker':
            return
        
        # Determine which compose file to use
        if config['vector_db'] in ['sqlite', 'memory'] and config['llm_model'] in ['phi', 'llama2']:
            compose_file = "docker-compose.local-minimal.yml"
        else:
            compose_file = "docker-compose.local-full.yml"
        
        print(f"\n✅ Use {compose_file} for deployment")
        
        # Create convenience script
        script_content = f"""#!/bin/bash
# fiChat-RAG Local Deployment Script

echo "🚀 Starting fiChat-RAG with local configuration..."

# Export environment variables
export VECTOR_STORE_TYPE={config['vector_db']}
export OLLAMA_MODEL={config['llm_model']}
export EMBEDDING_MODEL={config['embedding_model']}

# Start services
docker-compose -f {compose_file} up -d

echo "✅ Services started!"
echo ""
echo "📍 Access points:"
echo "  - API: http://localhost:8000"
echo "  - API Docs: http://localhost:8000/docs"

if [ "{config['vector_db']}" = "chromadb" ]; then
    echo "  - ChromaDB UI: http://localhost:8001"
elif [ "{config['vector_db']}" = "qdrant" ]; then
    echo "  - Qdrant Dashboard: http://localhost:6333/dashboard"
fi

echo ""
echo "📝 To view logs: docker-compose -f {compose_file} logs -f"
echo "🛑 To stop: docker-compose -f {compose_file} down"
"""
        
        with open("start_local.sh", "w") as f:
            f.write(script_content)
        
        os.chmod("start_local.sh", 0o755)
        print("✅ Created start_local.sh script")
    
    def generate_python_script(self, config: Dict):
        """Generate Python startup script for local deployment."""
        if config.get('deployment') != 'local':
            return
        
        script_content = f'''#!/usr/bin/env python3
"""Local fiChat-RAG startup script."""

import os
import sys
from pathlib import Path

# Set environment variables
os.environ.update({{
    "LLM_PROVIDER": "ollama",
    "LLM_MODEL": "{config['llm_model']}",
    "VECTOR_STORE_TYPE": "{config['vector_db']}",
    "EMBEDDING_MODEL": "{config['embedding_model']}",
    "EMBEDDING_DEVICE": "{config.get('embedding_device', 'cpu')}",
    "CHUNK_SIZE": "{config.get('chunk_size', 1000)}",
    "CHUNK_OVERLAP": "{config.get('chunk_overlap', 100)}"
}})

# Add project to path
sys.path.insert(0, str(Path(__file__).parent))

from src import RAG, Config

def main():
    """Run local fiChat-RAG instance."""
    print("🚀 Starting fiChat-RAG...")
    
    # Initialize
    config = Config.from_env()
    rag = RAG(config=config)
    
    print("✅ fiChat-RAG initialized!")
    print(f"   - LLM: {config.llm_model}")
    print(f"   - Vector Store: {config.vector_store_type}")
    print(f"   - Embeddings: {config.embedding_model}")
    
    # Interactive mode
    print("\\n" + "="*50)
    print("Interactive RAG Console")
    print("Type 'quit' to exit")
    print("="*50 + "\\n")
    
    while True:
        try:
            query = input("Query> ").strip()
            if query.lower() in ['quit', 'exit', 'q']:
                break
            
            if query:
                response = rag.query(query)
                print(f"\\nResponse: {response}\\n")
                
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"Error: {e}")
    
    print("\\nGoodbye!")

if __name__ == "__main__":
    main()
'''
        
        with open("run_local.py", "w") as f:
            f.write(script_content)
        
        os.chmod("run_local.py", 0o755)
        print("✅ Created run_local.py script")
    
    def run(self):
        """Run the configuration wizard."""
        print("🚀 fiChat-RAG Local Setup Wizard")
        print("="*50)
        
        # Detect hardware
        print("\n📊 Detecting Hardware...")
        hardware = self.detect_hardware()
        print(f"  - Platform: {hardware['platform']}")
        print(f"  - CPU Cores: {hardware['cpu_cores']}")
        print(f"  - RAM: {hardware['ram_gb']}GB")
        print(f"  - GPU: {'Yes' if hardware['gpu_available'] else 'No'}")
        print(f"  - Free Disk: {hardware['disk_space_gb']}GB")
        
        # Check dependencies
        print("\n🔍 Checking Dependencies...")
        deps = self.check_dependencies()
        all_good = True
        for dep, available in deps.items():
            status = "✅" if available else "❌"
            print(f"  {status} {dep}")
            if not available and dep in ["docker", "ollama"]:
                all_good = False
        
        if not all_good:
            print("\n⚠️  Missing required dependencies!")
            print("Please install:")
            if not deps["docker"]:
                print("  - Docker: https://docs.docker.com/get-docker/")
            if not deps["ollama"]:
                print("  - Ollama: https://ollama.ai/download")
            
            if input("\nContinue anyway? [y/N]: ").strip().lower() != "y":
                return
        
        # Get recommendations
        print("\n💡 Generating Recommendations...")
        recommendations = self.recommend_config(hardware)
        
        # Interactive configuration
        config = self.interactive_config()
        
        # Generate files
        print("\n📝 Generating Configuration Files...")
        self.generate_env_file(config)
        self.generate_docker_compose(config)
        self.generate_python_script(config)
        
        # Final instructions
        print("\n✅ Configuration Complete!")
        print("="*50)
        
        if config.get('deployment') == 'docker':
            print("\n🐳 Docker Deployment:")
            print("  1. Start Ollama: ollama serve")
            print(f"  2. Pull model: ollama pull {config['llm_model']}")
            print("  3. Run: ./start_local.sh")
        else:
            print("\n🐍 Python Deployment:")
            print("  1. Install dependencies: pip install -r requirements.txt")
            print("  2. Start Ollama: ollama serve")
            print(f"  3. Pull model: ollama pull {config['llm_model']}")
            print("  4. Run: python run_local.py")
        
        print("\n📚 Documentation:")
        print("  - Local Setup Guide: docs/LOCAL_SETUP_GUIDE.md")
        print("  - Integration Plan: docs/OLLAMA_LOCAL_INTEGRATION_PLAN.md")
        
        # Save configuration summary
        summary = {
            "hardware": hardware,
            "recommendations": recommendations,
            "selected_config": config,
            "timestamp": str(Path.ctime(Path.cwd()))
        }
        
        with open("local_config_summary.json", "w") as f:
            json.dump(summary, f, indent=2)
        
        print("\n💾 Configuration saved to local_config_summary.json")


if __name__ == "__main__":
    wizard = LocalSetupWizard()
    wizard.run()