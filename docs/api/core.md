# Core API Reference

This page documents the core classes and functions in PythonRAG.

## RAGPipeline

::: pythonrag.core.RAGPipeline
    options:
      show_root_heading: true
      show_source: true
      heading_level: 3

## Core Components

The RAGPipeline is the main interface for building RAG applications. It orchestrates:

- **Document Processing**: Text chunking and preprocessing
- **Embedding Generation**: Converting text to vector representations
- **Vector Storage**: Storing and indexing embeddings
- **Retrieval**: Finding relevant documents for queries
- **Generation**: Using LLMs to generate responses

## Usage Examples

### Basic Initialization

```python
from pythonrag import RAGPipeline

# Default configuration
rag = RAGPipeline()

# Custom configuration
rag = RAGPipeline(
    embedding_model="text-embedding-3-small",
    llm_model="gpt-4o-mini",
    chunk_size=800,
    top_k=5
)
```

### Adding Documents

```python
# Add text documents
documents = [
    "First document content...",
    "Second document content...",
]
rag.add_documents(documents)

# Add documents with metadata
documents = ["Document content..."]
metadata = [{"source": "file1.txt", "category": "technical"}]
rag.add_documents(documents, metadata=metadata)

# Add from file
rag.add_document_file("path/to/document.txt")
```

### Querying

```python
# Simple query
response = rag.query("What is machine learning?")

# Query with custom parameters
response = rag.query(
    "What is machine learning?",
    top_k=3,
    context_length=2000
)
```

### Configuration Management

```python
# Get current configuration
stats = rag.get_stats()
print(stats)

# Reset the pipeline
rag.reset()
```

## Configuration Options

### Embedding Models

| Model | Type | Description |
|-------|------|-------------|
| `sentence-transformers/all-MiniLM-L6-v2` | Local | Fast, lightweight, good quality |
| `text-embedding-3-small` | OpenAI | High quality, cost-effective |
| `text-embedding-3-large` | OpenAI | Highest quality, more expensive |
| `BAAI/bge-large-en-v1.5` | Hugging Face | State-of-the-art performance |

### Language Models

| Model | Provider | Cost | Quality | Use Case |
|-------|----------|------|---------|----------|
| `gpt-4o-mini` | OpenAI | Low | High | General purpose |
| `gpt-4o` | OpenAI | High | Highest | Complex reasoning |
| `claude-3-haiku-20240307` | Anthropic | Low | Good | Fast responses |
| `claude-3-5-sonnet-20241022` | Anthropic | Medium | Very High | Balanced |
| `ollama:llama3.1:8b` | Local | Free | Good | Privacy-first |

### Vector Database Configuration

```python
# In-memory (default)
vector_db = {"type": "in_memory"}

# ChromaDB (persistent)
vector_db = {
    "type": "chroma",
    "persist_directory": "./vector_db"
}

# Pinecone (cloud)
vector_db = {
    "type": "pinecone",
    "api_key": "your-key",
    "index_name": "my-index",
    "environment": "us-west1-gcp"
}

# Weaviate
vector_db = {
    "type": "weaviate",
    "url": "http://localhost:8080",
    "api_key": "your-key"  # optional
}

# Qdrant
vector_db = {
    "type": "qdrant",
    "url": "http://localhost:6333",
    "api_key": "your-key"  # optional
}
```

### Chunking Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `chunk_size` | 1000 | Maximum characters per chunk |
| `chunk_overlap` | 200 | Characters to overlap between chunks |
| `top_k` | 5 | Number of chunks to retrieve |

## Error Handling

PythonRAG provides specific exceptions for different error conditions:

```python
from pythonrag.exceptions import (
    PythonRAGError,
    ConfigurationError,
    EmbeddingError,
    VectorDatabaseError,
    DocumentProcessingError,
    LLMError
)

try:
    rag = RAGPipeline(embedding_model="invalid-model")
except ConfigurationError as e:
    print(f"Configuration error: {e}")

try:
    rag.add_documents(["document"])
except DocumentProcessingError as e:
    print(f"Document processing error: {e}")
```

## Performance Considerations

### Memory Usage

- **In-memory**: RAM usage scales with document count
- **Persistent**: Minimal RAM, slower initial load
- **Cloud**: Minimal local resources

### Cost Optimization

```python
# Cost-effective configuration
rag = RAGPipeline(
    embedding_model="text-embedding-3-small",  # 5x cheaper than large
    llm_model="gpt-4o-mini",                   # 15x cheaper than gpt-4o
    chunk_size=1000,                           # Fewer API calls
    top_k=3                                    # Smaller context
)
```

### Batch Processing

```python
# Process multiple documents efficiently
documents = ["doc1", "doc2", "doc3", ...]
rag.add_documents(documents)  # Batch processing
```

## Advanced Usage

### Custom Vector Database

```python
# Example: Custom vector database configuration
custom_db = {
    "type": "custom",
    "class_name": "MyVectorDB",
    "init_params": {
        "host": "localhost",
        "port": 9200
    }
}

rag = RAGPipeline(vector_db=custom_db)
```

### Pipeline Monitoring

```python
# Monitor pipeline statistics
stats = rag.get_stats()
print(f"Documents: {stats.get('document_count', 0)}")
print(f"Index size: {stats.get('index_size_mb', 0)} MB")
```

## Thread Safety

!!! warning "Thread Safety"
    RAGPipeline instances are **not thread-safe**. For concurrent access:
    
    ```python
    import threading
    
    # Create separate instances per thread
    def worker_thread():
        rag = RAGPipeline()  # New instance per thread
        # ... use rag
    
    # Or use locks for shared instance
    rag_lock = threading.Lock()
    
    def worker_with_lock():
        with rag_lock:
            result = rag.query("question")
    ```

## Migration Guide

### From v0.1.0 to v0.2.0

```python
# Old way (v0.1.0)
rag = RAGPipeline(model="gpt-3.5-turbo")

# New way (v0.2.0+)
rag = RAGPipeline(
    embedding_model="text-embedding-3-small",
    llm_model="gpt-3.5-turbo"
)
``` 
