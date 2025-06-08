#!/usr/bin/env python3
"""
Simple Qdrant Pipeline Usage Example

This example shows how to use the QdrantPipeline implementation.

Usage:
    python examples/qdrant_example.py

Requirements:
    pip install qdrant-client openai sentence-transformers

Environment Variables:
    OPENAI_API_KEY: Your OpenAI API key
    QDRANT_URL: Qdrant server URL (optional, defaults to local)
    QDRANT_API_KEY: Qdrant API key (for cloud instances)
"""

import os
import sys
from pathlib import Path

# Add the parent directory to Python path to import our package
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.pythonrag.pipelines import QdrantPipeline


def main():
    """Simple example demonstrating the Qdrant Pipeline."""
    print("🚀 Qdrant Pipeline Example")
    print("=" * 40)

    # Check environment variables
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ OPENAI_API_KEY environment variable is required")
        print("Please set it with: export OPENAI_API_KEY='your-api-key'")
        return

    try:
        # Initialize the pipeline
        print("\n📊 Initializing Qdrant Pipeline...")

        rag = QdrantPipeline(
            embedding_model="all-MiniLM-L6-v2",
            llm_model="gpt-4o-mini",
            qdrant_url=os.getenv("QDRANT_URL"),
            qdrant_api_key=os.getenv("QDRANT_API_KEY"),
            collection_name="example_docs",
            top_k=3,
        )

        print("✅ Pipeline initialized successfully!")

        # Add sample documents
        print("\n📚 Adding sample documents...")

        sample_docs = [
            {
                "content": "Python is a high-level programming language known for its simplicity and readability.",
                "metadata": {"source": "Python Guide", "topic": "programming"},
            },
            {
                "content": "Machine learning is a subset of AI that enables computers to learn from data.",
                "metadata": {"source": "ML Handbook", "topic": "AI"},
            },
        ]

        rag.add_documents(sample_docs)
        print(f"✅ Added {len(sample_docs)} documents!")

        # Query the system
        print("\n🤖 Querying the system...")
        question = "What is Python?"
        response = rag.query(question)

        print(f"\n❓ Question: {question}")
        print(f"🤖 Response: {response}")

        # Show stats
        print("\n📈 Pipeline Stats:")
        stats = rag.get_stats()
        for key, value in stats.items():
            print(f"  {key}: {value}")

    except Exception as e:
        print(f"\n❌ Error: {e}")


if __name__ == "__main__":
    main()
