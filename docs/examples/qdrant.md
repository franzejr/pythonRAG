# Qdrant Integration

This page demonstrates how to use the `QdrantPipeline` implementation for building RAG systems with Qdrant as the vector database backend.

## Quick Start

```python
from pythonrag.pipelines import QdrantPipeline

# Initialize with Qdrant
rag = QdrantPipeline(
    embedding_model="all-MiniLM-L6-v2",
    llm_model="gpt-4o-mini",
    collection_name="my_documents"
)

# Add documents
rag.add_documents([
    "Your document content here...",
    "More documents..."
])

# Query
response = rag.query("What is this about?")
print(response)
```

## What is Qdrant?

Qdrant is an open-source vector database written in Rust, designed for high-performance similarity search and vector storage. It's particularly well-suited for RAG applications due to:

- **Performance**: Highly optimized vector search algorithms
- **Scalability**: Horizontal scaling with clustering
- **Flexibility**: Rich filtering and payload capabilities
- **Ease of use**: Simple REST API and Python client
- **Deployment options**: Self-hosted or cloud-managed

## Prerequisites

### 1. Install Dependencies

```bash
pip install qdrant-client openai sentence-transformers
```

### 2. Set Up Qdrant

=== "Local Qdrant (Docker)"

    ```bash
    # Pull and run Qdrant locally
    docker pull qdrant/qdrant
    docker run -p 6333:6333 qdrant/qdrant
    ```

=== "Qdrant Cloud"

    1. Sign up at [cloud.qdrant.io](https://cloud.qdrant.io)
    2. Create a cluster
    3. Get your API key and endpoint URL
    4. Set environment variables:
       ```bash
       export QDRANT_API_KEY="your-api-key"
       export QDRANT_URL="your-cluster-url"
       ```

### 3. API Keys

```bash
export OPENAI_API_KEY="your-openai-key"  # For LLM responses
export QDRANT_API_KEY="your-qdrant-key"  # Optional for cloud
```

## Basic Usage

### Simple Example

```python
from pythonrag.pipelines import QdrantPipeline

# Initialize pipeline
rag = QdrantPipeline(
    embedding_model="all-MiniLM-L6-v2",
    llm_model="gpt-4o-mini",
    collection_name="demo_docs"
)

# Add some documents
documents = [
    "Python is a programming language known for its simplicity.",
    "Machine learning is a subset of artificial intelligence.",
    "RAG combines retrieval and generation for better AI responses."
]

rag.add_documents(documents)

# Query the system
question = "What is Python?"
answer = rag.query(question)
print(f"Q: {question}")
print(f"A: {answer}")
```

### Advanced Configuration

```python
from pythonrag.pipelines import QdrantPipeline
import os

rag = QdrantPipeline(
    # Model configuration
    embedding_model="all-MiniLM-L6-v2",
    llm_model="gpt-4o-mini",
    
    # Qdrant configuration
    qdrant_url=os.getenv("QDRANT_URL", "http://localhost:6333"),
    qdrant_api_key=os.getenv("QDRANT_API_KEY"),
    collection_name="advanced_rag",
    vector_size=384,  # Must match embedding model
    
    # Chunking configuration
    chunk_size=800,
    chunk_overlap=100,
    
    # Retrieval configuration
    top_k=5
)
```

## Working with Documents

### Adding Text Documents

```python
# Simple text documents
documents = [
    "First document content...",
    "Second document content..."
]
rag.add_documents(documents)

# Documents with metadata
docs_with_metadata = [
    {
        "content": "Document about Python programming",
        "metadata": {"category": "programming", "language": "python"}
    },
    {
        "content": "Document about machine learning",
        "metadata": {"category": "AI", "difficulty": "intermediate"}
    }
]
rag.add_documents(docs_with_metadata)
```

### Adding Files

```python
# Add document from file
rag.add_document_file(
    "path/to/document.txt",
    metadata={"source": "document.txt", "type": "technical"}
)
```

## Configuration Options

### Qdrant Connection

```python
# Local Qdrant
rag = QdrantPipeline(
    qdrant_url="http://localhost:6333"
)

# Cloud Qdrant
rag = QdrantPipeline(
    qdrant_url="https://your-cluster.qdrant.io",
    qdrant_api_key="your-api-key"
)

# Custom collection settings
rag = QdrantPipeline(
    collection_name="my_collection",
    vector_size=384  # Must match your embedding model
)
```

### Embedding Models

```python
# Local Sentence Transformers (default)
rag = QdrantPipeline(embedding_model="all-MiniLM-L6-v2")

# Other popular models
rag = QdrantPipeline(embedding_model="all-mpnet-base-v2")  # Better quality
rag = QdrantPipeline(embedding_model="paraphrase-multilingual-MiniLM-L12-v2")  # Multilingual
```

### Language Models

```python
# OpenAI models
rag = QdrantPipeline(llm_model="gpt-4o-mini")     # Fast and cost-effective
rag = QdrantPipeline(llm_model="gpt-4o")          # Best quality
rag = QdrantPipeline(llm_model="gpt-3.5-turbo")   # Legacy support
```

### Chunking Strategy

```python
rag = QdrantPipeline(
    chunk_size=1000,    # Characters per chunk
    chunk_overlap=200,  # Overlap between chunks
    top_k=5            # Number of chunks to retrieve
)
```

## Querying and Retrieval

### Basic Queries

```python
# Simple query
response = rag.query("What is machine learning?")

# Query with custom parameters
response = rag.query(
    "Explain neural networks",
    top_k=3,                    # Retrieve top 3 chunks
    context_length=1500         # Limit context size
)
```

## Pipeline Management

### Getting Statistics

```python
stats = rag.get_stats()
print(f"Embedding model: {stats['embedding_model']}")
print(f"LLM model: {stats['llm_model']}")
print(f"Vector DB type: {stats['vector_db_type']}")
print(f"Chunk size: {stats['chunk_size']}")
print(f"Documents: {stats.get('document_count', 'N/A')}")
```

### Resetting the Pipeline

```python
# Clear all documents and reset
rag.reset()
```

## Complete Example

Here's a complete working example you can run:

```python
#!/usr/bin/env python3
"""
Complete Qdrant Pipeline Example
"""
import os
from pythonrag.pipelines import QdrantPipeline

def main():
    # Check for required API key
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ Please set OPENAI_API_KEY environment variable")
        return
    
    print("🚀 Qdrant Pipeline Demo")
    print("=" * 40)
    
    try:
        # Initialize pipeline
        print("📊 Initializing pipeline...")
        rag = QdrantPipeline(
            embedding_model="all-MiniLM-L6-v2",
            llm_model="gpt-4o-mini",
            collection_name="demo_collection",
            top_k=3
        )
        print("✅ Pipeline initialized!")
        
        # Add sample documents
        print("\n📚 Adding documents...")
        sample_docs = [
            {
                "content": "Python is a high-level programming language known for its simplicity and readability. It's widely used in web development, data science, and AI.",
                "metadata": {"topic": "programming", "language": "python"}
            },
            {
                "content": "Machine learning is a subset of artificial intelligence that enables computers to learn and improve from experience without being explicitly programmed.",
                "metadata": {"topic": "AI", "difficulty": "beginner"}
            },
            {
                "content": "RAG (Retrieval-Augmented Generation) combines information retrieval with text generation to create more accurate and informative AI responses.",
                "metadata": {"topic": "AI", "technique": "RAG"}
            }
        ]
        
        rag.add_documents(sample_docs)
        print(f"✅ Added {len(sample_docs)} documents!")
        
        # Show stats
        print("\n📈 Pipeline Statistics:")
        stats = rag.get_stats()
        for key, value in stats.items():
            print(f"  {key}: {value}")
        
        # Query examples
        questions = [
            "What is Python?",
            "How does machine learning work?",
            "What is RAG?"
        ]
        
        print("\n🤖 Sample Queries:")
        print("-" * 40)
        
        for question in questions:
            print(f"\n❓ {question}")
            try:
                answer = rag.query(question)
                print(f"💡 {answer}")
            except Exception as e:
                print(f"❌ Error: {e}")
        
        print("\n✅ Demo completed successfully!")
        
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()
```

Save this as `qdrant_demo.py` and run:

```bash
export OPENAI_API_KEY="your-key-here"
python qdrant_demo.py
```

## Error Handling

The QdrantPipeline handles various error conditions gracefully:

```python
from pythonrag.exceptions import (
    ConfigurationError,
    VectorDatabaseError,
    EmbeddingError
)

try:
    rag = QdrantPipeline(
        qdrant_url="invalid-url",
        qdrant_api_key="invalid-key"
    )
except ConfigurationError as e:
    print(f"Configuration error: {e}")

try:
    rag.add_documents(["document"])
except VectorDatabaseError as e:
    print(f"Vector database error: {e}")
```

## Performance Tips

### Batch Processing
```python
# Process documents in batches for better performance
large_document_list = [...]  # Many documents
rag.add_documents(large_document_list)  # Batch processed internally
```

### Embedding Caching
```python
# The pipeline automatically caches embeddings to avoid recomputation
rag.add_documents(["doc1", "doc2"])  # Embeddings generated
rag.add_documents(["doc1", "doc3"])  # doc1 embedding reused
```

### Cost Optimization
```python
# Use cost-effective models
rag = QdrantPipeline(
    embedding_model="all-MiniLM-L6-v2",  # Free local model
    llm_model="gpt-4o-mini",             # Most cost-effective OpenAI model
    chunk_size=800,                      # Smaller chunks = lower costs
    top_k=3                              # Fewer chunks = lower costs
)
```

## Next Steps

- **Try different embedding models**: Experiment with various models for your use case
- **Customize chunking**: Adjust chunk size and overlap for your document types
- **Add filters**: Use Qdrant's payload filtering for advanced queries
- **Scale up**: Deploy Qdrant cluster for production workloads
- **Monitor performance**: Track query latency and accuracy metrics

## Troubleshooting

### Common Issues

**Connection Failed**
```
❌ Failed to connect to Qdrant: Connection refused
```
- Ensure Qdrant is running: `docker run -p 6333:6333 qdrant/qdrant`
- Check the URL and port

**API Key Issues**
```
❌ OPENAI_API_KEY environment variable is required
```
- Set your OpenAI API key: `export OPENAI_API_KEY="your-key"`

**Vector Size Mismatch**
```
❌ Vector dimension mismatch
```
- Ensure `vector_size` matches your embedding model's output size
- `all-MiniLM-L6-v2` = 384 dimensions
- `text-embedding-3-small` = 1536 dimensions

For more help, check our [FAQ](../faq.md) or open an issue on GitHub. 
 