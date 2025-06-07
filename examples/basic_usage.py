"""
Basic usage example for PythonRAG.

This example demonstrates how to use the PythonRAG package for
Retrieval-Augmented Generation workflows.
"""

from pythonrag import RAGPipeline


def main():
    """Demonstrate basic PythonRAG usage."""
    print("PythonRAG Basic Usage Example")
    print("=" * 40)

    # Initialize a RAG pipeline with default settings
    print("1. Initializing RAG pipeline...")
    rag = RAGPipeline(
        embedding_model="sentence-transformers/all-MiniLM-L6-v2",
        llm_model="gpt-3.5-turbo",
        chunk_size=1000,
        top_k=5,
    )

    # Display pipeline configuration
    print("2. Pipeline configuration:")
    stats = rag.get_stats()
    for key, value in stats.items():
        print(f"   {key}: {value}")

    # Example documents (this will fail until implementation is complete)
    print("\n3. Adding documents (not yet implemented)...")
    try:
        documents = [
            "The quick brown fox jumps over the lazy dog.",
            "Python is a versatile programming language used for web development, data science, and AI.",
            "Retrieval-Augmented Generation combines information retrieval with text generation.",
        ]
        rag.add_documents(documents)
    except NotImplementedError as e:
        print(f"   Expected error: {e}")

    # Example query (this will also fail until implementation is complete)
    print("\n4. Querying the system (not yet implemented)...")
    try:
        response = rag.query("What is Python used for?")
        print(f"   Response: {response}")
    except NotImplementedError as e:
        print(f"   Expected error: {e}")

    print("\n5. Example completed!")
    print("   The actual functionality will be implemented in subsequent iterations.")


if __name__ == "__main__":
    main()
