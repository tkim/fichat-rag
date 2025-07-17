"""Example demonstrating different local vector store options with Ollama."""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src import RAG, Config
from src.storage.base import Document


def demo_sqlite_store():
    """Demonstrate SQLite vector store."""
    print("\n" + "="*50)
    print("SQLite Vector Store Demo")
    print("="*50 + "\n")
    
    # Configure for SQLite
    config = Config(
        llm_provider="ollama",
        llm_model="phi",  # Using smaller model for demo
        vector_store_type="sqlite",
        vector_store_config={
            "db_path": "./data/fichat_sqlite.db"
        },
        embedding_model="sentence-transformers/all-MiniLM-L6-v2"
    )
    
    # Initialize RAG
    rag = RAG(config=config)
    
    # Add sample documents
    docs = [
        Document(
            content="SQLite is a lightweight, file-based database perfect for embedded applications.",
            metadata={"source": "sqlite_info", "type": "database"}
        ),
        Document(
            content="Vector databases enable similarity search using mathematical vectors.",
            metadata={"source": "vector_db_info", "type": "concept"}
        )
    ]
    
    rag.add_documents(docs)
    print(f"Added {len(docs)} documents to SQLite store")
    
    # Query
    response = rag.query("What is SQLite?", k=2)
    print(f"\nQuery: What is SQLite?")
    print(f"Response: {response}")
    
    return rag


def demo_chromadb_store():
    """Demonstrate ChromaDB vector store."""
    print("\n" + "="*50)
    print("ChromaDB Vector Store Demo")
    print("="*50 + "\n")
    
    # Configure for ChromaDB
    config = Config(
        llm_provider="ollama",
        llm_model="phi",
        vector_store_type="chromadb",
        vector_store_config={
            "persist_directory": "./data/chroma_db",
            "collection_name": "fichat_demo"
        },
        embedding_model="sentence-transformers/all-MiniLM-L6-v2"
    )
    
    # Initialize RAG
    rag = RAG(config=config)
    
    # Add sample documents
    docs = [
        Document(
            content="ChromaDB is an AI-native open-source embedding database designed for LLM applications.",
            metadata={"source": "chroma_info", "type": "database"}
        ),
        Document(
            content="Embedding databases store and search high-dimensional vectors efficiently.",
            metadata={"source": "embedding_db_info", "type": "concept"}
        )
    ]
    
    rag.add_documents(docs)
    print(f"Added {len(docs)} documents to ChromaDB store")
    
    # Query
    response = rag.query("What is ChromaDB?", k=2)
    print(f"\nQuery: What is ChromaDB?")
    print(f"Response: {response}")
    
    return rag


def compare_performance():
    """Compare performance of different vector stores."""
    print("\n" + "="*50)
    print("Performance Comparison")
    print("="*50 + "\n")
    
    import time
    
    # Test documents
    test_docs = [
        Document(
            content=f"Document {i}: This is a test document for performance comparison. " * 10,
            metadata={"id": i, "type": "test"}
        )
        for i in range(100)
    ]
    
    stores = ["memory", "sqlite", "chromadb"]
    results = {}
    
    for store_type in stores:
        print(f"\nTesting {store_type}...")
        
        # Configure
        config = Config(
            llm_provider="ollama",
            llm_model="phi",
            vector_store_type=store_type,
            vector_store_config={
                "db_path": f"./data/test_{store_type}.db" if store_type == "sqlite" else {},
                "persist_directory": f"./data/test_{store_type}" if store_type == "chromadb" else {}
            },
            embedding_model="sentence-transformers/all-MiniLM-L6-v2"
        )
        
        try:
            # Initialize
            rag = RAG(config=config)
            
            # Time document addition
            start = time.time()
            rag.add_documents(test_docs)
            add_time = time.time() - start
            
            # Time search
            start = time.time()
            for i in range(10):
                rag.query(f"Find document about test {i}", k=5, include_sources=True)
            search_time = (time.time() - start) / 10
            
            results[store_type] = {
                "add_time": add_time,
                "search_time": search_time,
                "docs_per_second": len(test_docs) / add_time
            }
            
        except Exception as e:
            print(f"Error with {store_type}: {e}")
            results[store_type] = {"error": str(e)}
    
    # Print results
    print("\n" + "-"*50)
    print("Results:")
    print("-"*50)
    
    for store, metrics in results.items():
        if "error" in metrics:
            print(f"\n{store}: ERROR - {metrics['error']}")
        else:
            print(f"\n{store}:")
            print(f"  - Add time: {metrics['add_time']:.2f}s")
            print(f"  - Search time: {metrics['search_time']:.3f}s per query")
            print(f"  - Throughput: {metrics['docs_per_second']:.1f} docs/s")


def main():
    """Run all demos."""
    print("Local Vector Store Demonstrations with Ollama")
    
    # Check if Ollama is running
    import requests
    try:
        response = requests.get("http://localhost:11434/api/tags")
        if response.status_code != 200:
            raise Exception("Ollama not responding")
    except:
        print("\nError: Ollama is not running!")
        print("Please start Ollama with: ollama serve")
        print("And pull a model with: ollama pull phi")
        return
    
    # Create data directory
    Path("./data").mkdir(exist_ok=True)
    
    # Run demos
    demo_sqlite_store()
    demo_chromadb_store()
    compare_performance()
    
    print("\n✅ All demos completed!")


if __name__ == "__main__":
    main()