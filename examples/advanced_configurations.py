"""
Advanced Configuration Examples for PythonRAG.

This example demonstrates various configuration patterns and use cases
for the PythonRAG package, including different model combinations,
vector database options, and optimization strategies.
"""

from pythonrag import RAGPipeline
from pythonrag.exceptions import ConfigurationError
import json


def demonstrate_embedding_model_options():
    """Show different embedding model configuration options."""
    print("🔗 Embedding Model Configuration Options")
    print("=" * 50)

    embedding_configs = [
        {
            "name": "Sentence Transformers (Local)",
            "model": "sentence-transformers/all-MiniLM-L6-v2",
            "pros": ["Fast", "Free", "Good quality", "Privacy-friendly"],
            "cons": ["Requires local compute", "Limited to model size"],
        },
        {
            "name": "OpenAI Embeddings (Small)",
            "model": "text-embedding-3-small",
            "pros": ["High quality", "Low cost", "No local compute"],
            "cons": ["Requires API key", "Data sent to OpenAI"],
        },
        {
            "name": "OpenAI Embeddings (Large)",
            "model": "text-embedding-3-large",
            "pros": ["Highest quality", "Best performance"],
            "cons": ["Higher cost", "Requires API key"],
        },
        {
            "name": "Hugging Face Models",
            "model": "BAAI/bge-large-en-v1.5",
            "pros": ["State-of-the-art", "Open source", "Customizable"],
            "cons": ["Larger model size", "More compute intensive"],
        },
    ]

    for i, config in enumerate(embedding_configs, 1):
        print(f"\n{i}. {config['name']}")
        print(f"   Model: {config['model']}")
        print(f"   ✅ Pros: {', '.join(config['pros'])}")
        print(f"   ⚠️  Cons: {', '.join(config['cons'])}")


def demonstrate_llm_model_options():
    """Show different LLM configuration options."""
    print("\n🤖 Language Model Configuration Options")
    print("=" * 50)

    llm_configs = [
        {
            "name": "OpenAI GPT-4o Mini",
            "model": "gpt-4o-mini",
            "cost": "Low",
            "quality": "High",
            "use_case": "General purpose, cost-effective",
        },
        {
            "name": "OpenAI GPT-4o",
            "model": "gpt-4o",
            "cost": "High",
            "quality": "Highest",
            "use_case": "Complex reasoning, highest quality",
        },
        {
            "name": "Anthropic Claude Haiku",
            "model": "claude-3-haiku-20240307",
            "cost": "Low",
            "quality": "Good",
            "use_case": "Fast responses, simple tasks",
        },
        {
            "name": "Anthropic Claude Sonnet",
            "model": "claude-3-5-sonnet-20241022",
            "cost": "Medium",
            "quality": "Very High",
            "use_case": "Balanced performance and cost",
        },
        {
            "name": "Local Ollama Models",
            "model": "llama3.1:8b",
            "cost": "Free",
            "quality": "Good",
            "use_case": "Privacy, no API dependency",
        },
    ]

    for i, config in enumerate(llm_configs, 1):
        print(f"\n{i}. {config['name']}")
        print(f"   Model: {config['model']}")
        print(f"   💰 Cost: {config['cost']}")
        print(f"   🎯 Quality: {config['quality']}")
        print(f"   📝 Use Case: {config['use_case']}")


def demonstrate_vector_database_options():
    """Show different vector database configuration options."""
    print("\n🗄️  Vector Database Configuration Options")
    print("=" * 50)

    vector_db_configs = [
        {
            "name": "In-Memory (Default)",
            "config": {"type": "in_memory"},
            "pros": ["Simple setup", "Fast for small datasets"],
            "cons": ["Not persistent", "Memory limited"],
        },
        {
            "name": "ChromaDB (Local)",
            "config": {"type": "chroma", "persist_directory": "./chroma_db"},
            "pros": ["Persistent", "Good for development", "Free"],
            "cons": ["Single machine", "Limited scalability"],
        },
        {
            "name": "Pinecone (Cloud)",
            "config": {
                "type": "pinecone",
                "api_key": "your-pinecone-key",
                "environment": "us-west1-gcp-free",
                "index_name": "pythonrag-index",
            },
            "pros": ["Highly scalable", "Managed service", "Fast queries"],
            "cons": ["Requires API key", "Paid service"],
        },
        {
            "name": "Weaviate (Cloud/Self-hosted)",
            "config": {
                "type": "weaviate",
                "url": "https://your-cluster.weaviate.network",
                "api_key": "your-weaviate-key",
            },
            "pros": ["Flexible", "GraphQL API", "Hybrid search"],
            "cons": ["More complex setup", "Learning curve"],
        },
        {
            "name": "Qdrant (Local/Cloud)",
            "config": {
                "type": "qdrant",
                "url": "http://localhost:6333",
                "api_key": None,
            },
            "pros": ["High performance", "Rich filtering", "Open source"],
            "cons": ["Setup complexity", "Resource intensive"],
        },
    ]

    for i, config in enumerate(vector_db_configs, 1):
        print(f"\n{i}. {config['name']}")
        print(f"   Config: {json.dumps(config['config'], indent=6)}")
        print(f"   ✅ Pros: {', '.join(config['pros'])}")
        print(f"   ⚠️  Cons: {', '.join(config['cons'])}")


def create_configuration_examples():
    """Create and test different RAG pipeline configurations."""
    print("\n⚙️  RAG Pipeline Configuration Examples")
    print("=" * 50)

    configurations = [
        {
            "name": "Development Setup",
            "description": "Quick setup for local development and testing",
            "config": {
                "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
                "llm_model": "gpt-4o-mini",
                "vector_db": {"type": "in_memory"},
                "chunk_size": 500,
                "chunk_overlap": 50,
                "top_k": 3,
            },
        },
        {
            "name": "Production Setup (Cost-Optimized)",
            "description": "Balanced cost and performance for production",
            "config": {
                "embedding_model": "text-embedding-3-small",
                "llm_model": "gpt-4o-mini",
                "vector_db": {"type": "chroma", "persist_directory": "./prod_db"},
                "chunk_size": 1000,
                "chunk_overlap": 100,
                "top_k": 5,
            },
        },
        {
            "name": "High-Performance Setup",
            "description": "Maximum quality for critical applications",
            "config": {
                "embedding_model": "text-embedding-3-large",
                "llm_model": "gpt-4o",
                "vector_db": {
                    "type": "pinecone",
                    "api_key": "your-key",
                    "environment": "us-west1-gcp",
                    "index_name": "high-perf-index",
                },
                "chunk_size": 800,
                "chunk_overlap": 150,
                "top_k": 8,
            },
        },
        {
            "name": "Privacy-First Setup",
            "description": "All processing local, no external API calls",
            "config": {
                "embedding_model": "sentence-transformers/all-mpnet-base-v2",
                "llm_model": "ollama:llama3.1:8b",
                "vector_db": {"type": "chroma", "persist_directory": "./private_db"},
                "chunk_size": 1200,
                "chunk_overlap": 200,
                "top_k": 4,
            },
        },
    ]

    for i, setup in enumerate(configurations, 1):
        print(f"\n{i}. {setup['name']}")
        print(f"   📝 {setup['description']}")
        print(f"   ⚙️  Configuration:")

        try:
            # Create the pipeline (this will work once implemented)
            rag = RAGPipeline(**setup["config"])
            stats = rag.get_stats()

            for key, value in stats.items():
                print(f"      {key}: {value}")

            print("   ✅ Configuration valid")

        except Exception as e:
            print(f"   ⚠️  Configuration issue: {e}")


def demonstrate_chunking_strategies():
    """Show different document chunking strategies."""
    print("\n📄 Document Chunking Strategies")
    print("=" * 50)

    chunking_strategies = [
        {
            "name": "Small Chunks (High Precision)",
            "chunk_size": 300,
            "chunk_overlap": 50,
            "use_case": "Precise answers, specific facts",
            "pros": ["High precision", "Less noise"],
            "cons": ["May miss context", "More chunks to process"],
        },
        {
            "name": "Medium Chunks (Balanced)",
            "chunk_size": 800,
            "chunk_overlap": 100,
            "use_case": "General purpose applications",
            "pros": ["Good balance", "Reasonable context"],
            "cons": ["Moderate precision", "Standard performance"],
        },
        {
            "name": "Large Chunks (High Context)",
            "chunk_size": 1500,
            "chunk_overlap": 200,
            "use_case": "Complex reasoning, long-form content",
            "pros": ["Rich context", "Better for reasoning"],
            "cons": ["More noise", "Higher costs"],
        },
        {
            "name": "Adaptive Chunks",
            "chunk_size": "sentence_boundary",
            "chunk_overlap": "semantic",
            "use_case": "Semantic coherence preservation",
            "pros": ["Natural boundaries", "Better coherence"],
            "cons": ["Variable sizes", "More complex processing"],
        },
    ]

    for i, strategy in enumerate(chunking_strategies, 1):
        print(f"\n{i}. {strategy['name']}")
        print(f"   Chunk Size: {strategy['chunk_size']}")
        print(f"   Overlap: {strategy['chunk_overlap']}")
        print(f"   🎯 Use Case: {strategy['use_case']}")
        print(f"   ✅ Pros: {', '.join(strategy['pros'])}")
        print(f"   ⚠️  Cons: {', '.join(strategy['cons'])}")


def demonstrate_retrieval_strategies():
    """Show different retrieval strategies and their configurations."""
    print("\n🔍 Retrieval Strategy Options")
    print("=" * 50)

    retrieval_strategies = [
        {
            "name": "Simple Similarity Search",
            "top_k": 5,
            "method": "cosine_similarity",
            "description": "Basic vector similarity search",
            "best_for": "General purpose, fast responses",
        },
        {
            "name": "Diverse Retrieval",
            "top_k": 8,
            "method": "mmr",  # Maximal Marginal Relevance
            "description": "Maximizes relevance while minimizing redundancy",
            "best_for": "Comprehensive answers, avoiding repetition",
        },
        {
            "name": "Threshold-based Retrieval",
            "top_k": 10,
            "method": "similarity_threshold",
            "threshold": 0.7,
            "description": "Only return results above similarity threshold",
            "best_for": "High precision, filtering irrelevant content",
        },
        {
            "name": "Hybrid Search",
            "top_k": 6,
            "method": "hybrid",
            "description": "Combines semantic and keyword search",
            "best_for": "Balancing semantic understanding and exact matches",
        },
    ]

    for i, strategy in enumerate(retrieval_strategies, 1):
        print(f"\n{i}. {strategy['name']}")
        print(f"   Top-K: {strategy['top_k']}")
        print(f"   Method: {strategy['method']}")
        if "threshold" in strategy:
            print(f"   Threshold: {strategy['threshold']}")
        print(f"   📝 {strategy['description']}")
        print(f"   🎯 Best for: {strategy['best_for']}")


def main():
    """Main function to demonstrate advanced configurations."""
    print("🚀 PythonRAG Advanced Configuration Guide")
    print("=" * 60)

    print("\nThis guide shows various configuration options for the PythonRAG package.")
    print("Choose the configuration that best fits your use case and requirements.")

    # Show all configuration options
    demonstrate_embedding_model_options()
    demonstrate_llm_model_options()
    demonstrate_vector_database_options()
    create_configuration_examples()
    demonstrate_chunking_strategies()
    demonstrate_retrieval_strategies()

    print("\n" + "=" * 60)
    print("🎉 Configuration guide completed!")
    print("\n💡 Key Takeaways:")
    print("   • Start with simple configurations for development")
    print("   • Consider costs when choosing OpenAI models")
    print("   • Use persistent vector databases for production")
    print("   • Optimize chunk sizes for your specific use case")
    print("   • Test different retrieval strategies for best results")

    print("\n📚 Next Steps:")
    print("   1. Choose a configuration that fits your needs")
    print("   2. Set up required API keys and dependencies")
    print("   3. Test with small datasets first")
    print("   4. Scale up based on performance requirements")


if __name__ == "__main__":
    main()
