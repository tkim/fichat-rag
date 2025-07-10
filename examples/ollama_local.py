"""Example of using fichat-rag with Ollama for fully local RAG."""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src import RAG, Config, OllamaLLM
from src.storage.base import Document


def check_ollama_running():
    """Check if Ollama is running."""
    import requests
    try:
        response = requests.get("http://localhost:11434/api/tags")
        return response.status_code == 200
    except:
        return False


def main():
    """Run local RAG with Ollama."""
    
    # Check if Ollama is running
    if not check_ollama_running():
        print("Error: Ollama is not running!")
        print("Please start Ollama with: ollama serve")
        print("And pull a model with: ollama pull llama2")
        return
    
    # Configure for local operation
    config = Config(
        # Use Ollama for LLM
        llm_provider="ollama",
        llm_model="llama2",  # Change to your preferred model
        llm_temperature=0.7,
        llm_max_tokens=1000,
        
        # Use local embeddings
        embedding_model="sentence-transformers/all-MiniLM-L6-v2",
        embedding_device="cpu",  # Use "cuda" if you have GPU
        
        # Use in-memory vector store for demo
        vector_store_type="memory",
        
        # Retrieval settings
        retrieval_top_k=5,
        retrieval_search_type="hybrid",
        retrieval_rerank=False,  # Disable reranking for faster response
        
        # Chunking settings
        chunk_size=500,
        chunk_overlap=50
    )
    
    # Initialize RAG
    print("Initializing local RAG system...")
    rag = RAG(config=config)
    
    # Option 1: Add documents from files
    print("\nOption 1: Add documents from files")
    print("Place your PDF, TXT, or MD files in the 'documents' folder")
    
    doc_folder = Path("documents")
    if doc_folder.exists():
        files = list(doc_folder.glob("**/*"))
        supported_files = [f for f in files if f.suffix.lower() in ['.pdf', '.txt', '.md']]
        
        if supported_files:
            print(f"Found {len(supported_files)} documents to process...")
            for file in supported_files:
                print(f"Adding {file.name}...")
                try:
                    rag.add_documents(str(file))
                except Exception as e:
                    print(f"Error adding {file.name}: {e}")
        else:
            print("No supported documents found in 'documents' folder")
    
    # Option 2: Add custom documents programmatically
    print("\nOption 2: Adding some example documents...")
    
    example_docs = [
        Document(
            content="""
            Ollama is a tool for running large language models locally. It supports various 
            open-source models including Llama 2, Mistral, and Code Llama. Ollama provides 
            a simple API that makes it easy to integrate with applications.
            """,
            metadata={"source": "ollama_info", "type": "documentation"}
        ),
        Document(
            content="""
            Retrieval-Augmented Generation (RAG) is a technique that enhances language models 
            by retrieving relevant information from a knowledge base. This allows the model to 
            access up-to-date and specific information beyond its training data.
            """,
            metadata={"source": "rag_info", "type": "concept"}
        ),
        Document(
            content="""
            Vector databases store data as high-dimensional vectors, enabling efficient 
            similarity search. Popular vector databases include Pinecone, Weaviate, ChromaDB, 
            and PostgreSQL with pgvector extension. They are essential for RAG systems.
            """,
            metadata={"source": "vector_db_info", "type": "technology"}
        )
    ]
    
    rag.add_documents(example_docs)
    print(f"Added {len(example_docs)} example documents")
    
    # Interactive Q&A
    print("\n" + "="*50)
    print("Local RAG system is ready!")
    print("Ask questions about the loaded documents.")
    print("Type 'quit' to exit")
    print("="*50 + "\n")
    
    while True:
        question = input("Your question: ").strip()
        
        if question.lower() in ['quit', 'exit', 'q']:
            break
        
        if not question:
            continue
        
        try:
            print("\nThinking...\n")
            
            # Query with streaming
            print("Answer: ", end="", flush=True)
            
            for chunk in rag.query(question, stream=True, include_sources=False):
                print(chunk, end="", flush=True)
            
            print("\n")
            
            # Get sources separately
            response = rag.query(question, k=3, include_sources=True)
            
            print("\nRelevant sources:")
            for i, source in enumerate(response['sources']):
                source_name = source['metadata'].get('source', 'Unknown')
                print(f"{i+1}. {source_name}: {source['content'][:100]}...")
            
        except Exception as e:
            print(f"\nError: {e}")
        
        print("\n" + "-"*50 + "\n")
    
    print("\nGoodbye!")


if __name__ == "__main__":
    # Create documents folder if it doesn't exist
    Path("documents").mkdir(exist_ok=True)
    
    main()