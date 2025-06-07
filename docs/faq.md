# Frequently Asked Questions

Common questions and answers about using PythonRAG.

## Getting Started

### Q: How do I install PythonRAG?

**A:** The simplest way is:
```bash
pip install pythonrag
```

For all features:
```bash
pip install "pythonrag[all]"
```

See the [Installation Guide](getting-started/installation.md) for more options.

### Q: Do I need API keys to use PythonRAG?

**A:** Not necessarily! You can use PythonRAG with local models only:

```python
from pythonrag import RAGPipeline

# Completely local setup
rag = RAGPipeline(
    embedding_model="sentence-transformers/all-MiniLM-L6-v2",
    llm_model="ollama:llama3.1:8b"
)
```

For OpenAI integration, you'll need an API key.

### Q: What's the minimum system requirements?

**A:**
- **Python**: 3.8+
- **RAM**: 4GB minimum, 8GB+ recommended
- **Storage**: 1GB for models and data
- **Network**: Internet connection for API-based models

## Configuration

### Q: Which embedding model should I choose?

**A:** It depends on your needs:

| Use Case | Recommended Model | Why |
|----------|-------------------|-----|
| **Learning/Testing** | `sentence-transformers/all-MiniLM-L6-v2` | Free, fast, good quality |
| **Production (Budget)** | `text-embedding-3-small` | Cost-effective OpenAI model |
| **Production (Quality)** | `text-embedding-3-large` | Highest quality OpenAI model |
| **Privacy-First** | `sentence-transformers/all-mpnet-base-v2` | Local, high quality |

### Q: Which LLM should I use?

**A:** Based on common scenarios:

| Scenario | Model | Cost | Quality |
|----------|-------|------|---------|
| **Development** | `gpt-4o-mini` | Low | High |
| **Production** | `gpt-4o-mini` | Low | High |
| **High-Quality** | `gpt-4o` | High | Highest |
| **Local/Private** | `ollama:llama3.1:8b` | Free | Good |
| **Balanced** | `claude-3-haiku-20240307` | Low | Good |

### Q: How do I optimize for cost?

**A:** Several strategies:

1. **Choose cost-effective models:**
   ```python
   rag = RAGPipeline(
       embedding_model="text-embedding-3-small",  # 5x cheaper than large
       llm_model="gpt-4o-mini"                   # 15x cheaper than gpt-4o
   )
   ```

2. **Use larger chunks (fewer embedding calls):**
   ```python
   rag = RAGPipeline(chunk_size=1200, top_k=3)
   ```

3. **Cache embeddings with persistent storage:**
   ```python
   rag = RAGPipeline(
       vector_db={"type": "chroma", "persist_directory": "./cache"}
   )
   ```

## Usage

### Q: How do I add documents from files?

**A:** Use the `add_document_file` method:

```python
# Single file
rag.add_document_file("document.txt")

# Multiple files
files = ["doc1.txt", "doc2.pdf", "doc3.md"]
for file in files:
    rag.add_document_file(file)
```

### Q: Can I add metadata to documents?

**A:** Yes! Metadata helps with filtering and context:

```python
documents = ["Document content..."]
metadata = [{"source": "manual.pdf", "section": "introduction"}]

rag.add_documents(documents, metadata=metadata)
```

### Q: How do I handle large documents?

**A:** PythonRAG automatically chunks large documents:

```python
# Configure chunking
rag = RAGPipeline(
    chunk_size=1000,      # Characters per chunk
    chunk_overlap=200,    # Overlap between chunks
    top_k=5              # Chunks to retrieve
)

# Add large document
with open("large_document.txt", "r") as f:
    content = f.read()
    rag.add_documents([content])
```

## Troubleshooting

### Q: I get "ModuleNotFoundError: No module named 'pythonrag'"

**A:** This usually means:

1. **Not installed**: Run `pip install pythonrag`
2. **Wrong environment**: Check you're in the correct virtual environment
3. **Development mode**: If developing, run `pip install -e .`

### Q: OpenAI API errors - what do they mean?

**A:** Common OpenAI errors:

| Error | Meaning | Solution |
|-------|---------|----------|
| `Unauthorized` | Invalid API key | Check `OPENAI_API_KEY` |
| `Rate limit exceeded` | Too many requests | Wait and retry |
| `Insufficient quota` | No credits left | Add credits to account |
| `Model not found` | Invalid model name | Check model availability |

### Q: Why are my queries slow?

**A:** Several possible causes:

1. **Large vector database**: Use more selective `top_k`
2. **Complex embeddings**: Try lighter models
3. **Network latency**: Consider local models
4. **Large chunks**: Optimize `chunk_size`

**Optimization:**
```python
# Faster configuration
rag = RAGPipeline(
    embedding_model="sentence-transformers/all-MiniLM-L6-v2",  # Local
    llm_model="gpt-4o-mini",                                  # Fast API
    chunk_size=800,                                           # Smaller chunks
    top_k=3                                                   # Fewer results
)
```

### Q: Memory usage is too high - how to reduce it?

**A:** Memory optimization strategies:

1. **Use persistent storage:**
   ```python
   rag = RAGPipeline(
       vector_db={"type": "chroma", "persist_directory": "./db"}
   )
   ```

2. **Smaller embedding models:**
   ```python
   rag = RAGPipeline(
       embedding_model="sentence-transformers/all-MiniLM-L6-v2"
   )
   ```

3. **Batch processing:**
   ```python
   # Process documents in batches
   for batch in chunks(documents, batch_size=100):
       rag.add_documents(batch)
   ```

## Advanced Usage

### Q: Can I use custom vector databases?

**A:** Yes! PythonRAG supports custom vector database implementations:

```python
# Built-in options
vector_db_configs = {
    "chroma": {"type": "chroma", "persist_directory": "./db"},
    "pinecone": {"type": "pinecone", "api_key": "key", "index_name": "idx"},
    "weaviate": {"type": "weaviate", "url": "http://localhost:8080"},
    "qdrant": {"type": "qdrant", "url": "http://localhost:6333"}
}

rag = RAGPipeline(vector_db=vector_db_configs["chroma"])
```

### Q: How do I implement custom retrieval strategies?

**A:** You can customize the retrieval process:

```python
# Custom retrieval configuration
rag = RAGPipeline(
    retrieval_strategy="hybrid",    # Combines semantic + keyword search
    rerank_model="cross-encoder",   # Re-ranks results for relevance
    top_k=10,                      # Retrieve more, then re-rank
    final_k=5                      # Final number after re-ranking
)
```

### Q: Can I run PythonRAG in production?

**A:** Absolutely! Consider these production practices:

1. **Error handling:**
   ```python
   from pythonrag.exceptions import PythonRAGError
   
   try:
       response = rag.query(user_question)
   except PythonRAGError as e:
       # Log error and provide fallback
       logger.error(f"RAG error: {e}")
       response = "I'm sorry, I couldn't process your question."
   ```

2. **Monitoring:**
   ```python
   # Track usage
   stats = rag.get_stats()
   logger.info(f"Pipeline stats: {stats}")
   ```

3. **Scaling:**
   ```python
   # Use connection pooling for high throughput
   rag = RAGPipeline(
       openai_config={"max_connections": 20}
   )
   ```

### Q: How do I handle multiple languages?

**A:** Use multilingual models:

```python
# Multilingual embedding
rag = RAGPipeline(
    embedding_model="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    llm_model="gpt-4o"  # Supports many languages
)

# Add documents in different languages
documents = [
    "English document content...",
    "Contenu du document français...",
    "Contenido del documento español..."
]
rag.add_documents(documents)
```

## Integration

### Q: How do I integrate with web frameworks?

**A:** Example with FastAPI:

```python
from fastapi import FastAPI
from pythonrag import RAGPipeline

app = FastAPI()
rag = RAGPipeline()  # Initialize once

@app.post("/query")
async def query_rag(question: str):
    try:
        response = rag.query(question)
        return {"answer": response}
    except Exception as e:
        return {"error": str(e)}
```

### Q: Can I use PythonRAG with Jupyter notebooks?

**A:** Yes! Perfect for experimentation:

```python
# In Jupyter cell
from pythonrag import RAGPipeline

rag = RAGPipeline()
rag.add_documents(["Your documents here..."])

# Interactive querying
while True:
    question = input("Ask a question (or 'quit'): ")
    if question.lower() == 'quit':
        break
    
    answer = rag.query(question)
    print(f"Answer: {answer}\n")
```

### Q: How do I deploy to cloud platforms?

**A:** Common deployment patterns:

**Docker:**
```dockerfile
FROM python:3.10-slim
RUN pip install "pythonrag[all]"
COPY . /app
WORKDIR /app
CMD ["python", "app.py"]
```

**Environment variables:**
```bash
export OPENAI_API_KEY="your-key"
export PYTHONRAG_CONFIG="production"
```

## Performance

### Q: What's the expected query latency?

**A:** Typical response times:

| Setup | Embedding | LLM | Total |
|-------|-----------|-----|-------|
| **Local** | 50-200ms | 1-5s | 1-5s |
| **OpenAI** | 100-500ms | 500-2000ms | 1-3s |
| **Hybrid** | 50-200ms | 500-2000ms | 1-2s |

### Q: How many documents can PythonRAG handle?

**A:** Depends on your setup:

| Storage | Documents | Memory | Notes |
|---------|-----------|--------|-------|
| **In-Memory** | 1K-10K | High | Fast queries |
| **ChromaDB** | 10K-1M | Low | Good balance |
| **Pinecone** | 1M+ | Minimal | Cloud scaling |

### Q: How do I benchmark my setup?

**A:** Use the built-in benchmarking:

```python
import time
from pythonrag import RAGPipeline

rag = RAGPipeline()
# ... add documents ...

# Benchmark queries
questions = ["Question 1", "Question 2", ...]
start_time = time.time()

for question in questions:
    response = rag.query(question)

total_time = time.time() - start_time
avg_time = total_time / len(questions)

print(f"Average query time: {avg_time:.2f} seconds")
```

## Migration

### Q: How do I upgrade from v0.1.0 to v0.2.0?

**A:** Key changes:

```python
# Old (v0.1.0)
rag = RAGPipeline(model="gpt-3.5-turbo")

# New (v0.2.0+)
rag = RAGPipeline(
    embedding_model="text-embedding-3-small",
    llm_model="gpt-3.5-turbo"
)
```

### Q: Can I export my vector database?

**A:** Yes, for supported databases:

```python
# ChromaDB - files are in persist_directory
# Copy the entire directory

# For migration:
old_rag = RAGPipeline(vector_db={"type": "chroma", "persist_directory": "./old_db"})
new_rag = RAGPipeline(vector_db={"type": "pinecone", "api_key": "key"})

# Re-add documents (embeddings will be recalculated)
```

## Still Need Help?

If you can't find your answer here:

1. **Check the documentation**: Comprehensive guides available
2. **Search GitHub Issues**: [github.com/franzejr/PythonRAG/issues](https://github.com/franzejr/PythonRAG/issues)
3. **Join discussions**: [GitHub Discussions](https://github.com/franzejr/PythonRAG/discussions)
4. **Report bugs**: Create a new issue with details

**When reporting issues, please include:**
- Python version (`python --version`)
- PythonRAG version (`pip show pythonrag`)
- Error message and stack trace
- Minimal reproducible example
- Operating system 
