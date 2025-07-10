"""Basic RAG example using the fichat-rag framework."""

import os
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src import RAG, Config, OllamaLLM

def main():
    """Run a basic RAG example."""
    
    # Configure RAG
    config = Config(
        llm_provider="ollama",
        llm_model="llama2",
        vector_store_type="memory",  # Use in-memory store for demo
        embedding_model="sentence-transformers/all-MiniLM-L6-v2",
        chunk_size=500,
        chunk_overlap=50,
        retrieval_top_k=5,
        retrieval_rerank=True
    )
    
    # Initialize RAG
    print("Initializing RAG system...")
    rag = RAG(config=config)
    
    # Add some sample documents
    print("\nAdding sample documents...")
    
    sample_docs = [
        """
        Artificial Intelligence (AI) is the simulation of human intelligence in machines 
        that are programmed to think and learn. Machine learning is a subset of AI that 
        enables systems to learn and improve from experience without being explicitly programmed.
        Deep learning is a subset of machine learning that uses neural networks with multiple layers.
        """,
        """
        Natural Language Processing (NLP) is a branch of AI that helps computers understand, 
        interpret and manipulate human language. NLP combines computational linguistics with 
        machine learning and deep learning models. Common NLP tasks include text classification,
        named entity recognition, machine translation, and question answering.
        """,
        """
        Large Language Models (LLMs) are AI models trained on vast amounts of text data.
        Examples include GPT-3, GPT-4, and LLaMA. These models can generate human-like text,
        answer questions, summarize documents, and perform various language tasks.
        Retrieval-Augmented Generation (RAG) enhances LLMs by incorporating external knowledge.
        """
    ]
    
    # Add documents with metadata
    for i, doc_text in enumerate(sample_docs):
        rag.add_documents(
            doc_text,
            metadata={"source": f"sample_doc_{i+1}", "topic": "AI"}
        )
    
    print(f"Added {len(sample_docs)} documents to the RAG system.")
    
    # Interactive query loop
    print("\nRAG system ready! Enter your questions (type 'quit' to exit):\n")
    
    while True:
        query = input("Question: ").strip()
        
        if query.lower() in ['quit', 'exit', 'q']:
            break
        
        if not query:
            continue
        
        try:
            # Query the RAG system
            print("\nSearching for relevant information...")
            response = rag.query(
                query,
                include_sources=True,
                k=3
            )
            
            # Display answer
            print(f"\nAnswer: {response['answer']}")
            
            # Display sources
            print("\nSources:")
            for i, source in enumerate(response['sources']):
                print(f"\n{i+1}. {source['metadata'].get('source', 'Unknown source')}")
                print(f"   {source['content'][:200]}...")
                
        except Exception as e:
            print(f"\nError: {e}")
        
        print("\n" + "-"*50 + "\n")
    
    print("\nGoodbye!")


if __name__ == "__main__":
    main()