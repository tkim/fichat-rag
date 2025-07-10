"""Web scraping example using Firecrawl with fichat-rag."""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src import RAG, Config
from src.ingestion.loaders import FirecrawlLoader, WebLoader
from src.storage.base import Document


def main():
    """Run web scraping RAG example."""
    
    # Check for Firecrawl API key
    if not os.getenv("FIRECRAWL_API_KEY"):
        print("Error: FIRECRAWL_API_KEY environment variable not set!")
        print("Get your API key from: https://firecrawl.dev")
        print("Then set it with: export FIRECRAWL_API_KEY='your-api-key'")
        return
    
    # Configure RAG for web content
    config = Config(
        llm_provider="ollama",
        llm_model="llama2",
        vector_store_type="memory",
        chunk_size=1000,  # Larger chunks for web content
        chunk_overlap=100,
        retrieval_top_k=5
    )
    
    print("Initializing RAG system with Firecrawl...")
    rag = RAG(config=config)
    
    # Initialize Firecrawl loader
    web_loader = FirecrawlLoader(
        formats=["markdown"],  # Get clean markdown
        only_main_content=True,  # Remove navigation, footers, etc.
        timeout=30000
    )
    
    # Example 1: Scrape a single URL
    print("\n" + "="*50)
    print("Example 1: Scraping a single URL")
    print("="*50)
    
    url = "https://docs.firecrawl.dev/introduction"
    print(f"\nScraping: {url}")
    
    try:
        # Load the webpage
        doc_data = web_loader.load(url)
        
        # Create a Document object
        doc = Document(
            content=doc_data["content"],
            metadata=doc_data["metadata"]
        )
        
        # Add to RAG
        rag.add_documents([doc])
        print(f"✓ Added document: {doc.metadata.get('title', 'Untitled')}")
        
        # Query about the content
        question = "What is Firecrawl and what are its main features?"
        print(f"\nQuestion: {question}")
        
        response = rag.query(question, include_sources=False)
        print(f"Answer: {response}")
        
    except Exception as e:
        print(f"Error scraping URL: {e}")
    
    # Example 2: Crawl multiple pages from a website
    print("\n" + "="*50)
    print("Example 2: Crawling multiple pages")
    print("="*50)
    
    base_url = "https://docs.firecrawl.dev"
    print(f"\nCrawling website: {base_url}")
    print("This may take a moment...")
    
    try:
        # Crawl the website (limited to 5 pages for demo)
        crawled_docs = web_loader.crawl(
            base_url,
            limit=5,
            max_depth=2
        )
        
        print(f"\n✓ Crawled {len(crawled_docs)} pages")
        
        # Convert to Document objects and add to RAG
        documents = []
        for doc_data in crawled_docs:
            doc = Document(
                content=doc_data["content"],
                metadata=doc_data["metadata"]
            )
            documents.append(doc)
            print(f"  - {doc.metadata.get('title', doc.metadata.get('source'))}")
        
        rag.add_documents(documents)
        
        # Query about the crawled content
        question = "How do I use Firecrawl with Python?"
        print(f"\nQuestion: {question}")
        
        response_data = rag.query(question, include_sources=True)
        print(f"\nAnswer: {response_data['answer']}")
        
        print("\nSources:")
        for i, source in enumerate(response_data['sources'][:3]):
            print(f"{i+1}. {source['metadata'].get('source')}")
        
    except Exception as e:
        print(f"Error crawling website: {e}")
    
    # Example 3: Map website URLs
    print("\n" + "="*50)
    print("Example 3: Mapping website URLs")
    print("="*50)
    
    try:
        print(f"\nMapping all URLs from: {base_url}")
        
        urls = web_loader.map_website(base_url, limit=20)
        print(f"\n✓ Found {len(urls)} URLs:")
        
        for i, url in enumerate(urls[:10]):  # Show first 10
            print(f"  {i+1}. {url}")
        
        if len(urls) > 10:
            print(f"  ... and {len(urls) - 10} more")
    
    except Exception as e:
        print(f"Error mapping website: {e}")
    
    # Example 4: Scrape with custom options
    print("\n" + "="*50)
    print("Example 4: Advanced scraping options")
    print("="*50)
    
    # Create a loader with screenshot capability
    advanced_loader = FirecrawlLoader(
        formats=["markdown", "html"],
        screenshot=True,
        include_links=True,
        only_main_content=False  # Get full page content
    )
    
    url = "https://firecrawl.dev"
    print(f"\nScraping with advanced options: {url}")
    
    try:
        doc_data = advanced_loader.load(url)
        
        print(f"\n✓ Scraped successfully!")
        print(f"  Title: {doc_data['metadata'].get('title')}")
        print(f"  Description: {doc_data['metadata'].get('description')}")
        print(f"  Content length: {len(doc_data['content'])} characters")
        
        if doc_data['metadata'].get('screenshot_url'):
            print(f"  Screenshot: {doc_data['metadata']['screenshot_url']}")
        
        if doc_data['metadata'].get('links'):
            print(f"  Found {len(doc_data['metadata']['links'])} links")
    
    except Exception as e:
        print(f"Error with advanced scraping: {e}")
    
    # Interactive Q&A with web content
    print("\n" + "="*50)
    print("Interactive Q&A with scraped content")
    print("Type 'quit' to exit")
    print("="*50 + "\n")
    
    while True:
        question = input("Ask about the scraped content: ").strip()
        
        if question.lower() in ['quit', 'exit', 'q']:
            break
        
        if not question:
            continue
        
        try:
            response = rag.query(question, k=3)
            print(f"\nAnswer: {response}\n")
        except Exception as e:
            print(f"Error: {e}\n")
    
    print("\nGoodbye!")


if __name__ == "__main__":
    main()