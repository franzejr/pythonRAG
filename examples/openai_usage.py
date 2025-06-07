"""
OpenAI integration example for PythonRAG.

This example demonstrates how to use the PythonRAG package with OpenAI's
embedding and language models for advanced RAG workflows.

Prerequisites:
- Install OpenAI package: pip install openai
- Set your OpenAI API key: export OPENAI_API_KEY="your-api-key-here"
"""

import os
from typing import List, Dict, Any
from pathlib import Path

from pythonrag import RAGPipeline


def check_openai_setup() -> bool:
    """Check if OpenAI is properly configured."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("❌ OpenAI API key not found!")
        print("   Please set your API key: export OPENAI_API_KEY='your-api-key-here'")
        return False

    try:
        import openai

        print("✅ OpenAI package is installed")
        print(f"✅ API key is set (ends with: ...{api_key[-4:]})")
        return True
    except ImportError:
        print("❌ OpenAI package not installed!")
        print("   Please install it: pip install openai")
        return False


def demonstrate_openai_config():
    """Demonstrate different OpenAI model configurations."""
    print("\n" + "=" * 50)
    print("OpenAI Model Configuration Examples")
    print("=" * 50)

    # Example 1: OpenAI embeddings with OpenAI chat
    print("\n1. OpenAI Embeddings + OpenAI Chat Model:")
    config1 = {
        "embedding_model": "text-embedding-3-small",  # Latest OpenAI embedding model
        "llm_model": "gpt-4o-mini",  # Cost-effective GPT-4 model
        "vector_db": {"type": "in_memory"},
        "chunk_size": 1000,
        "top_k": 5,
    }
    print(f"   Configuration: {config1}")

    # Example 2: OpenAI embeddings with higher-end chat model
    print("\n2. High-Performance Configuration:")
    config2 = {
        "embedding_model": "text-embedding-3-large",  # Higher quality embeddings
        "llm_model": "gpt-4o",  # Latest GPT-4 model
        "vector_db": {"type": "in_memory"},
        "chunk_size": 800,
        "top_k": 8,
    }
    print(f"   Configuration: {config2}")

    # Example 3: Mixed configuration (OpenAI embeddings, different LLM)
    print("\n3. Mixed Model Configuration:")
    config3 = {
        "embedding_model": "text-embedding-3-small",
        "llm_model": "claude-3-haiku-20240307",  # Anthropic model
        "vector_db": {"type": "chroma"},
        "chunk_size": 1200,
        "top_k": 6,
    }
    print(f"   Configuration: {config3}")


def create_sample_documents() -> List[str]:
    """Create sample documents for RAG demonstration."""
    return [
        """
        Artificial Intelligence (AI) is a broad field of computer science focused on building 
        smart machines capable of performing tasks that typically require human intelligence. 
        AI systems can learn, reason, perceive, and make decisions. Machine learning, a subset 
        of AI, enables computers to learn and improve from experience without being explicitly 
        programmed for every task.
        """,
        """
        Natural Language Processing (NLP) is a branch of artificial intelligence that helps 
        computers understand, interpret and manipulate human language. NLP draws from many 
        disciplines, including computer science and computational linguistics, in its pursuit 
        to bridge the gap between human communication and computer understanding.
        """,
        """
        Large Language Models (LLMs) are advanced AI systems trained on vast amounts of text 
        data to understand and generate human-like text. Examples include GPT-4, Claude, and 
        PaLM. These models can perform various tasks such as writing, summarization, translation, 
        and question-answering by predicting the most likely next words in a sequence.
        """,
        """
        Retrieval-Augmented Generation (RAG) is a technique that combines information retrieval 
        with text generation. It first retrieves relevant documents from a knowledge base, then 
        uses this information to generate more accurate and contextually relevant responses. 
        RAG helps overcome the knowledge cutoff limitations of language models.
        """,
        """
        Vector databases are specialized databases designed to store and query high-dimensional 
        vectors efficiently. They are crucial for RAG systems as they enable semantic search 
        by storing document embeddings and finding similar content based on vector similarity 
        rather than exact keyword matches.
        """,
    ]


def demonstrate_openai_rag_workflow():
    """Demonstrate a complete RAG workflow with OpenAI models."""
    print("\n" + "=" * 50)
    print("OpenAI RAG Workflow Demonstration")
    print("=" * 50)

    if not check_openai_setup():
        print("\n⚠️  Cannot proceed without proper OpenAI setup")
        return

    print("\n1. Initializing RAG pipeline with OpenAI models...")

    # Create RAG pipeline with OpenAI configuration
    rag = RAGPipeline(
        embedding_model="text-embedding-3-small",
        llm_model="gpt-4o-mini",
        chunk_size=800,
        chunk_overlap=100,
        top_k=3,
    )

    print("✅ RAG pipeline initialized")

    # Display configuration
    print("\n2. Pipeline Configuration:")
    stats = rag.get_stats()
    for key, value in stats.items():
        print(f"   {key}: {value}")

    # Sample documents
    documents = create_sample_documents()
    print(f"\n3. Sample documents prepared ({len(documents)} documents)")

    # Try to add documents (will show NotImplementedError until functionality is built)
    print("\n4. Adding documents to RAG system...")
    try:
        rag.add_documents(documents)
        print("✅ Documents added successfully")
    except NotImplementedError as e:
        print(f"⚠️  Expected: {e}")
        print("   (This will work once the core functionality is implemented)")

    # Sample queries
    sample_queries = [
        "What is the difference between AI and machine learning?",
        "How does RAG improve language model responses?",
        "What are vector databases used for?",
        "Explain how NLP relates to artificial intelligence.",
    ]

    print(f"\n5. Sample queries prepared ({len(sample_queries)} queries)")
    for i, query in enumerate(sample_queries, 1):
        print(f"   {i}. {query}")

    # Try querying (will show NotImplementedError until functionality is built)
    print("\n6. Testing query functionality...")
    try:
        response = rag.query(sample_queries[0])
        print(f"✅ Query response: {response}")
    except NotImplementedError as e:
        print(f"⚠️  Expected: {e}")
        print("   (This will work once the core functionality is implemented)")


def demonstrate_cost_optimization():
    """Show how to optimize costs when using OpenAI models."""
    print("\n" + "=" * 50)
    print("OpenAI Cost Optimization Tips")
    print("=" * 50)

    tips = [
        {
            "title": "Choose the right embedding model",
            "description": "text-embedding-3-small is more cost-effective for most use cases",
            "cost_comparison": "~5x cheaper than text-embedding-3-large",
        },
        {
            "title": "Use GPT-4o-mini for most tasks",
            "description": "GPT-4o-mini provides excellent performance at lower cost",
            "cost_comparison": "~15x cheaper than GPT-4o",
        },
        {
            "title": "Optimize chunk size",
            "description": "Larger chunks mean fewer API calls for embeddings",
            "recommendation": "Use 800-1200 characters per chunk",
        },
        {
            "title": "Cache embeddings",
            "description": "Store embeddings to avoid recomputing for the same documents",
            "implementation": "Use persistent vector databases like Chroma or Pinecone",
        },
        {
            "title": "Limit context window",
            "description": "Only include the most relevant chunks in the prompt",
            "recommendation": "Use top_k=3-5 for most queries",
        },
    ]

    for i, tip in enumerate(tips, 1):
        print(f"\n{i}. {tip['title']}")
        print(f"   📝 {tip['description']}")
        if "cost_comparison" in tip:
            print(f"   💰 {tip['cost_comparison']}")
        if "recommendation" in tip:
            print(f"   ✅ {tip['recommendation']}")
        if "implementation" in tip:
            print(f"   🔧 {tip['implementation']}")


def main():
    """Main function to demonstrate OpenAI usage with PythonRAG."""
    print("🤖 PythonRAG + OpenAI Integration Example")
    print("=" * 60)

    # Check prerequisites
    print("\n📋 Prerequisites Check:")
    openai_ready = check_openai_setup()

    # Show different configuration options
    demonstrate_openai_config()

    # Demonstrate workflow
    demonstrate_openai_rag_workflow()

    # Cost optimization tips
    demonstrate_cost_optimization()

    print("\n" + "=" * 60)
    print("🎉 Example completed!")
    print("\nNext steps:")
    print("1. Set up your OpenAI API key if not already done")
    print("2. Install optional dependencies: pip install 'pythonrag[all]'")
    print("3. Wait for the core RAG functionality to be implemented")
    print("4. Start building your own RAG applications!")

    if not openai_ready:
        print("\n⚠️  Remember to set up OpenAI properly before using this configuration")


if __name__ == "__main__":
    main()
