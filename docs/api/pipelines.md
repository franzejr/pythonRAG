# Pipelines API Reference

This page documents the concrete pipeline implementations in PythonRAG.

## QdrantPipeline

::: pythonrag.pipelines.QdrantPipeline
    options:
      show_root_heading: true
      show_source: true
      heading_level: 3

## Overview

The `QdrantPipeline` class provides a complete implementation of the `RAGPipeline` abstract base class using Qdrant as the vector database backend.

## Key Features

- **Qdrant Integration**: Full support for Qdrant vector database
- **Sentence Transformers**: Local embedding models for privacy and cost efficiency
- **OpenAI Integration**: GPT models for high-quality text generation
- **Automatic Collection Management**: Creates and manages Qdrant collections
- **Document Chunking**: Intelligent text splitting with configurable overlap
- **Error Handling**: Comprehensive error handling and logging

## Quick Start

```python
from pythonrag.pipelines import QdrantPipeline

# Initialize with default settings
rag = QdrantPipeline()

# Add documents
rag.add_documents([
    "Your document content here...",
    "More documents..."
])

# Query the system
response = rag.query("What is this about?")
```

## Constructor Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `embedding_model` | str | "all-MiniLM-L6-v2" | Sentence Transformers model name |
| `llm_model` | str | "gpt-4o-mini" | OpenAI model name |
| `qdrant_url` | str | None | Qdrant server URL (defaults to localhost) |
| `qdrant_api_key` | str | None | Qdrant API key (optional) |
| `collection_name` | str | "documents" | Qdrant collection name |
| `vector_size` | int | 384 | Vector dimensions (must match embedding model) |
| `chunk_size` | int | 1000 | Maximum characters per chunk |
| `chunk_overlap` | int | 200 | Characters to overlap between chunks |
| `top_k` | int | 5 | Number of chunks to retrieve |

## Configuration Examples

### Local Qdrant

```python
from pythonrag.pipelines import QdrantPipeline

rag = QdrantPipeline(
    qdrant_url="http://localhost:6333",
    collection_name="my_documents"
)
```

### Cloud Qdrant

```python
import os
from pythonrag.pipelines import QdrantPipeline

rag = QdrantPipeline(
    qdrant_url=os.getenv("QDRANT_URL"),
    qdrant_api_key=os.getenv("QDRANT_API_KEY"),
    collection_name="production_docs"
)
```

### Advanced Configuration

```python
rag = QdrantPipeline(
    embedding_model="all-mpnet-base-v2",  # Better quality embeddings
    llm_model="gpt-4o",                   # Best OpenAI model
    qdrant_url="https://your-cluster.qdrant.io",
    qdrant_api_key="your-api-key",
    collection_name="advanced_rag",
    vector_size=768,                      # Match all-mpnet-base-v2
    chunk_size=800,
    chunk_overlap=100,
    top_k=3
)
```

## Methods

### add_documents()

Add documents to the pipeline for indexing.

```python
# Simple text documents
documents = ["Document 1 content", "Document 2 content"]
rag.add_documents(documents)

# Documents with metadata
docs = [
    {
        "content": "Document content...",
        "metadata": {"source": "file1.txt", "category": "tech"}
    }
]
rag.add_documents(docs)
```

### query()

Query the RAG system with a question.

```python
# Basic query
response = rag.query("What is machine learning?")

# Query with parameters
response = rag.query(
    "Explain neural networks",
    top_k=3,
    context_length=1500
)
```

### add_document_file()

Add a document from a file.

```python
rag.add_document_file(
    "path/to/document.txt",
    metadata={"source": "document.txt", "type": "manual"}
)
```

### get_stats()

Get pipeline statistics.

```python
stats = rag.get_stats()
print(stats)
# Output:
# {
#     'embedding_model': 'all-MiniLM-L6-v2',
#     'llm_model': 'gpt-4o-mini',
#     'vector_db_type': 'qdrant',
#     'chunk_size': 1000,
#     'chunk_overlap': 200,
#     'top_k': 5,
#     'documents_added': 42
# }
```

### reset()

Clear all documents from the pipeline.

```python
rag.reset()  # Removes all documents from Qdrant collection
```

## Embedding Models

The QdrantPipeline supports any Sentence Transformers model:

| Model | Vector Size | Quality | Speed | Use Case |
|-------|-------------|---------|--------|----------|
| `all-MiniLM-L6-v2` | 384 | Good | Fast | General purpose, default |
| `all-mpnet-base-v2` | 768 | Better | Medium | Higher quality needs |
| `paraphrase-multilingual-MiniLM-L12-v2` | 384 | Good | Medium | Multilingual support |
| `sentence-transformers/all-MiniLM-L12-v2` | 384 | Good | Medium | Balanced performance |

### Custom Embedding Models

```python
# Use any Sentence Transformers model
rag = QdrantPipeline(
    embedding_model="BAAI/bge-large-en-v1.5",
    vector_size=1024  # Must match model output size
)
```

## Language Models

Supports any OpenAI-compatible model:

```python
# OpenAI models
rag = QdrantPipeline(llm_model="gpt-4o-mini")      # Cost-effective
rag = QdrantPipeline(llm_model="gpt-4o")           # Best quality
rag = QdrantPipeline(llm_model="gpt-3.5-turbo")    # Legacy

# Custom OpenAI-compatible endpoints
rag = QdrantPipeline(llm_model="local-llm")  # Requires OPENAI_BASE_URL
```

## Error Handling

The QdrantPipeline provides specific error handling:

```python
from pythonrag.exceptions import (
    ConfigurationError,
    VectorDatabaseError,
    EmbeddingError
)

try:
    rag = QdrantPipeline(qdrant_url="invalid-url")
except ConfigurationError as e:
    print(f"Config error: {e}")

try:
    rag.add_documents(["test"])
except VectorDatabaseError as e:
    print(f"Qdrant error: {e}")
```

## Performance Considerations

### Memory Usage
- **Embedding Model**: Loaded once, cached in memory
- **Document Storage**: Vectors stored in Qdrant, minimal local memory
- **Batch Processing**: Documents processed in batches for efficiency

### Network Usage
- **Local Qdrant**: No network overhead
- **Cloud Qdrant**: Network calls for vector operations
- **OpenAI API**: Network calls for LLM responses only

### Cost Optimization
```python
# Cost-effective configuration
rag = QdrantPipeline(
    embedding_model="all-MiniLM-L6-v2",  # Free local embeddings
    llm_model="gpt-4o-mini",             # Cheapest OpenAI model
    chunk_size=800,                      # Smaller chunks = fewer tokens
    top_k=3                              # Fewer chunks = lower LLM costs
)
```

## Thread Safety

The QdrantPipeline is thread-safe for read operations (queries) but not for write operations (adding documents). For concurrent writes, implement external synchronization:

```python
import threading

lock = threading.Lock()

def add_documents_safe(rag, documents):
    with lock:
        rag.add_documents(documents)
```

## Integration Examples

### Flask Web API

```python
from flask import Flask, request, jsonify
from pythonrag.pipelines import QdrantPipeline

app = Flask(__name__)
rag = QdrantPipeline()

@app.route("/query", methods=["POST"])
def query():
    data = request.json
    response = rag.query(data["question"])
    return jsonify({"answer": response})

@app.route("/documents", methods=["POST"])
def add_docs():
    data = request.json
    rag.add_documents(data["documents"])
    return jsonify({"status": "success"})
```

### Async Usage

```python
import asyncio
from concurrent.futures import ThreadPoolExecutor
from pythonrag.pipelines import QdrantPipeline

async def async_query(rag, question):
    loop = asyncio.get_event_loop()
    with ThreadPoolExecutor() as executor:
        response = await loop.run_in_executor(executor, rag.query, question)
    return response

# Usage
async def main():
    rag = QdrantPipeline()
    response = await async_query(rag, "What is AI?")
    print(response)
```

## Extending QdrantPipeline

You can extend the QdrantPipeline for custom functionality:

```python
from pythonrag.pipelines import QdrantPipeline

class CustomQdrantPipeline(QdrantPipeline):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.query_count = 0
    
    def query(self, question, **kwargs):
        self.query_count += 1
        print(f"Query #{self.query_count}: {question}")
        return super().query(question, **kwargs)
    
    def get_query_stats(self):
        return {"total_queries": self.query_count}
```

## Troubleshooting

### Common Issues

**Qdrant Connection Failed**
```python
# Test connection manually
from qdrant_client import QdrantClient
try:
    client = QdrantClient("http://localhost:6333")
    print("✅ Qdrant connection successful")
except Exception as e:
    print(f"❌ Connection failed: {e}")
```

**Vector Size Mismatch**
```python
# Check embedding model dimensions
from sentence_transformers import SentenceTransformer
model = SentenceTransformer("all-MiniLM-L6-v2")
test_embedding = model.encode(["test"])
print(f"Vector size: {len(test_embedding[0])}")  # Should be 384
```

**OpenAI API Issues**
```python
import os
if not os.getenv("OPENAI_API_KEY"):
    print("❌ Set OPENAI_API_KEY environment variable")
```

For more troubleshooting tips, see our [FAQ](../faq.md). 
