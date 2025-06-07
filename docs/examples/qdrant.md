# Qdrant Integration Examples

This page demonstrates how to extend the PythonRAG pipeline with Qdrant vector database integration. Our examples show how to build upon the base `RAGPipeline` class to create fully functional RAG systems with Qdrant as the vector store backend.

## Architecture: Extending RAGPipeline

Our Qdrant integration demonstrates how to extend the base `RAGPipeline` class with custom implementations:

```python
from pythonrag import RAGPipeline

class QdrantRAGPipeline(RAGPipeline):
    """Extended RAG Pipeline with Qdrant implementation."""
    
    def add_documents(self, documents, metadata=None):
        # Custom implementation with Qdrant storage
        pass
    
    def query(self, question, top_k=None):
        # Custom implementation with Qdrant search
        pass
```

This approach provides:

- **Framework compliance**: Uses the standard RAG interface
- **Custom backends**: Full control over vector database operations  
- **Extensibility**: Easy to add features like filtering, hybrid search
- **Testing**: Consistent interface for unit and integration tests

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
pip install "pythonrag[vectordb]" qdrant-client
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
export OPENAI_API_KEY="your-openai-key"  # For embeddings and LLM
export QDRANT_API_KEY="your-qdrant-key"  # Optional for cloud
```

## Pipeline Extension Example

### Complete Implementation

Our Qdrant pipeline example (`examples/qdrant_pipeline_example.py`) shows how to extend the base RAGPipeline:

```python
python examples/qdrant_pipeline_example.py
```

**Features:**
- ✅ Document chunking and indexing
- ✅ OpenAI embeddings (text-embedding-3-small)
- ✅ Qdrant vector storage
- ✅ Interactive Q&A interface
- ✅ Source attribution
- ✅ Cost optimization
- ✅ Error handling

### Sample Session

```
🚀 PythonRAG Interactive Demo with Qdrant
==================================================
✅ OpenAI client initialized
✅ Connected to local Qdrant
✅ Created new collection: pythonrag_interactive_demo
✅ Interactive Qdrant RAG system initialized!

📚 Processing 4 documents...
   Created 12 chunks
   Generating embeddings...
   Progress: 10/12 embeddings generated
   Uploading to Qdrant...
✅ Successfully indexed 12 chunks

📊 Collection Stats:
   collection_name: pythonrag_interactive_demo
   points_count: 12
   vector_size: 1536
   distance_metric: Cosine

==================================================
🎯 Ready to answer questions about AI, LLMs, RAG, and Vector Databases!
   Type 'help' for available commands
   Type 'exit' to quit
==================================================

❓ Your question: What is the difference between AI and Machine Learning?

🔍 Searching for: 'What is the difference between AI and Machine Learning?'
   Found 3 relevant chunks
   🤖 Generating answer...
   ✅ Answer generated

💡 Answer:
----------------------------------------
Based on the provided context, here's the key difference between AI and Machine Learning:

**Artificial Intelligence (AI)** is the broader field of computer science focused on building smart machines capable of performing tasks that typically require human intelligence. These tasks include learning, reasoning, problem-solving, perception, and language understanding.

**Machine Learning (ML)** is a subset of AI that specifically enables computers to learn and improve from experience without being explicitly programmed. It uses algorithms and statistical models to analyze and draw inferences from patterns in data.

In simpler terms:
- **AI** is the overall goal of creating intelligent machines
- **Machine Learning** is one of the methods used to achieve AI

The relationship is hierarchical - all machine learning is part of AI, but not all AI uses machine learning. AI can also include other approaches like rule-based systems, expert systems, and symbolic reasoning.

📚 Sources (from 1 documents):
----------------------------------------
1. [AI_Overview.md] (similarity: 0.857)
   Artificial Intelligence (AI) is a broad field of computer science focused on building smart machines capable of performing tasks that...

2. [AI_Overview.md] (similarity: 0.834)
   Machine Learning is a subset of AI that enables computers to learn and improve from experience without being explicitly programmed...

3. [AI_Overview.md] (similarity: 0.798)
   General AI, also called Strong AI or Artificial General Intelligence (AGI), would have the ability to understand, learn, and apply...
```

## Configuration Examples

### Basic Qdrant Setup

```python
from pythonrag import RAGPipeline

# Local Qdrant
rag = RAGPipeline(
    embedding_model="text-embedding-3-small",
    llm_model="gpt-4o-mini",
    vector_db={
        "type": "qdrant",
        "url": "http://localhost:6333",
        "collection_name": "my_documents"
    }
)
```

### Cloud Qdrant Setup

```python
import os

rag = RAGPipeline(
    embedding_model="text-embedding-3-small",
    llm_model="gpt-4o-mini",
    vector_db={
        "type": "qdrant",
        "url": os.getenv("QDRANT_URL"),
        "api_key": os.getenv("QDRANT_API_KEY"),
        "collection_name": "production_docs"
    }
)
```

### Advanced Configuration

```python
rag = RAGPipeline(
    embedding_model="text-embedding-3-small",
    llm_model="gpt-4o-mini",
    vector_db={
        "type": "qdrant",
        "url": "http://localhost:6333",
        "collection_name": "advanced_collection",
        "distance_metric": "cosine",  # cosine, euclidean, dot_product
        "vector_size": 1536,  # Must match embedding model
        "create_collection": True,  # Auto-create if missing
        "timeout": 30,  # Connection timeout
        "prefer_grpc": False  # Use REST API
    },
    chunk_size=1000,
    chunk_overlap=200,
    top_k=5
)
```

## Performance Optimization

### 1. Vector Configuration

```python
# Optimize for your embedding model
vector_configs = {
    "text-embedding-3-small": {"size": 1536, "distance": "cosine"},
    "text-embedding-3-large": {"size": 3072, "distance": "cosine"},
    "all-MiniLM-L6-v2": {"size": 384, "distance": "cosine"},
    "all-mpnet-base-v2": {"size": 768, "distance": "cosine"}
}
```

### 2. Batch Operations

```python
# Process documents in batches for better performance
documents = ["doc1", "doc2", ...]  # Large list

# Batch size optimization
batch_size = 100
for i in range(0, len(documents), batch_size):
    batch = documents[i:i + batch_size]
    rag.add_documents(batch)
```

### 3. Collection Management

```python
# Pre-create optimized collection
from qdrant_client import QdrantClient, models

client = QdrantClient("http://localhost:6333")

client.create_collection(
    collection_name="optimized_collection",
    vectors_config=models.VectorParams(
        size=1536,
        distance=models.Distance.COSINE,
        # Performance optimizations
        hnsw_config=models.HnswConfigDiff(
            m=16,  # Number of connections
            ef_construct=100,  # Search quality during construction
            full_scan_threshold=10000  # When to use brute force
        )
    )
)
```

## Real-World Use Cases

### 1. Document Q&A System

```python
class DocumentQASystem:
    def __init__(self):
        self.rag = RAGPipeline(
            embedding_model="text-embedding-3-small",
            llm_model="gpt-4o-mini",
            vector_db={
                "type": "qdrant",
                "url": "http://localhost:6333",
                "collection_name": "company_docs"
            }
        )
    
    def index_documents(self, file_paths):
        """Index company documents."""
        for file_path in file_paths:
            self.rag.add_document_file(file_path)
    
    def ask_question(self, question):
        """Answer questions about indexed documents."""
        return self.rag.query(question)
```

### 2. Customer Support Bot

```python
class SupportBot:
    def __init__(self):
        self.rag = RAGPipeline(
            embedding_model="text-embedding-3-small",
            llm_model="gpt-4o-mini",
            vector_db={
                "type": "qdrant",
                "url": os.getenv("QDRANT_URL"),
                "api_key": os.getenv("QDRANT_API_KEY"),
                "collection_name": "support_kb"
            }
        )
    
    def handle_query(self, user_question, user_context=None):
        """Handle customer support queries with context."""
        # Add user context to the query if available
        if user_context:
            enhanced_question = f"User context: {user_context}\nQuestion: {user_question}"
        else:
            enhanced_question = user_question
        
        response = self.rag.query(enhanced_question)
        return {
            "answer": response,
            "confidence": "high",  # Based on similarity scores
            "suggested_actions": ["contact_human", "check_docs"]
        }
```

### 3. Code Documentation Search

```python
class CodeDocSearch:
    def __init__(self, repo_path):
        self.rag = RAGPipeline(
            embedding_model="text-embedding-3-small",
            llm_model="gpt-4o-mini",
            vector_db={
                "type": "qdrant",
                "collection_name": "code_docs"
            }
        )
        self._index_codebase(repo_path)
    
    def _index_codebase(self, repo_path):
        """Index code files and documentation."""
        import glob
        
        # Index documentation files
        doc_files = glob.glob(f"{repo_path}/**/*.md", recursive=True)
        for doc_file in doc_files:
            self.rag.add_document_file(doc_file)
        
        # Index Python docstrings
        py_files = glob.glob(f"{repo_path}/**/*.py", recursive=True)
        for py_file in py_files:
            # Extract and index docstrings
            # Implementation would parse Python files
            pass
    
    def search_api(self, query):
        """Search API documentation."""
        return self.rag.query(f"API documentation: {query}")
```

## Monitoring and Maintenance

### Collection Statistics

```python
def monitor_collection(rag_pipeline):
    """Monitor Qdrant collection health."""
    stats = rag_pipeline.get_stats()
    
    print(f"Documents: {stats.get('document_count', 0)}")
    print(f"Vector DB: {stats.get('vector_db_type')}")
    
    # Get detailed Qdrant stats
    if hasattr(rag_pipeline, '_vector_db'):
        db = rag_pipeline._vector_db
        if db.get('type') == 'qdrant':
            client = db['client']
            collection_name = db['collection_name']
            
            collection_info = client.get_collection(collection_name)
            print(f"Points: {collection_info.points_count}")
            print(f"Segments: {collection_info.segments_count}")
            print(f"Index status: {collection_info.status}")
```

### Performance Monitoring

```python
import time

def benchmark_search(rag_pipeline, queries, iterations=5):
    """Benchmark search performance."""
    total_time = 0
    
    for query in queries:
        start_time = time.time()
        
        for _ in range(iterations):
            result = rag_pipeline.query(query)
        
        query_time = (time.time() - start_time) / iterations
        total_time += query_time
        
        print(f"Query: '{query[:50]}...'")
        print(f"Average time: {query_time:.3f}s")
    
    avg_time = total_time / len(queries)
    print(f"\nOverall average: {avg_time:.3f}s per query")
```

## Troubleshooting

### Common Issues

#### Connection Errors

```python
# Test Qdrant connection
from qdrant_client import QdrantClient

try:
    client = QdrantClient("http://localhost:6333")
    collections = client.get_collections()
    print("✅ Qdrant connection successful")
except Exception as e:
    print(f"❌ Qdrant connection failed: {e}")
    print("Solutions:")
    print("1. Check if Qdrant is running: docker ps")
    print("2. Verify the URL and port")
    print("3. Check firewall settings")
```

#### Collection Issues

```python
# Debug collection configuration
def debug_collection(client, collection_name):
    try:
        collection = client.get_collection(collection_name)
        config = collection.config
        
        print(f"Collection: {collection_name}")
        print(f"Status: {collection.status}")
        print(f"Points: {collection.points_count}")
        print(f"Vector size: {config.params.vectors.size}")
        print(f"Distance: {config.params.vectors.distance}")
        
    except Exception as e:
        print(f"Collection error: {e}")
        print("Solutions:")
        print("1. Check collection name spelling")
        print("2. Verify vector dimensions match embedding model")
        print("3. Recreate collection if corrupted")
```

#### Performance Issues

```python
# Optimize search performance
def optimize_search_performance():
    """Tips for better search performance."""
    print("Performance optimization checklist:")
    print("✓ Use appropriate vector size for your model")
    print("✓ Choose the right distance metric (usually cosine)")
    print("✓ Tune HNSW parameters for your use case")
    print("✓ Consider payload optimization")
    print("✓ Monitor memory usage")
    print("✓ Use batch operations for bulk updates")
    print("✓ Implement proper error handling")
```

## Cost Optimization

### Embedding Costs

```python
# Calculate embedding costs for Qdrant + OpenAI
def estimate_embedding_costs(num_documents, avg_doc_length):
    """Estimate OpenAI embedding costs."""
    # Average characters per token (rough estimate)
    chars_per_token = 4
    
    # Calculate tokens
    total_chars = num_documents * avg_doc_length
    total_tokens = total_chars / chars_per_token
    
    # text-embedding-3-small pricing (as of 2024)
    cost_per_1k_tokens = 0.00002
    total_cost = (total_tokens / 1000) * cost_per_1k_tokens
    
    print(f"Estimated embedding cost:")
    print(f"  Documents: {num_documents:,}")
    print(f"  Total tokens: {total_tokens:,.0f}")
    print(f"  Estimated cost: ${total_cost:.4f}")
    
    return total_cost

# Example usage
estimate_embedding_costs(10000, 2000)  # 10K docs, 2K chars each
```

### Storage Optimization

```python
# Optimize Qdrant storage
storage_tips = {
    "vector_compression": "Use quantization for large collections",
    "payload_optimization": "Store only necessary metadata",
    "collection_sharding": "Split large collections across nodes",
    "backup_strategy": "Regular backups with compression",
    "index_optimization": "Tune HNSW parameters"
}

for tip, description in storage_tips.items():
    print(f"• {tip}: {description}")
```

## Next Steps

1. **Try the interactive example**: Run `python examples/interactive_qdrant_rag.py`
2. **Set up production Qdrant**: Use Qdrant Cloud or deploy with Docker
3. **Optimize for your use case**: Tune vector size, distance metrics, and HNSW parameters
4. **Implement monitoring**: Track performance and collection health
5. **Scale horizontally**: Use Qdrant clustering for large datasets

For more advanced Qdrant features, check the [official documentation](https://qdrant.tech/documentation/). 
