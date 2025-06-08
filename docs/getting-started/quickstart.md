# Quick Start

Get up and running with PythonRAG in under 5 minutes! This guide will walk you through creating your first RAG pipeline.

## 🚀 5-Minute Setup

### Step 1: Installation

```bash
pip install pythonrag
```

### Step 2: Basic Pipeline

Create a file called `my_first_rag.py`:

```python
from pythonrag import RAGPipeline

# Create a RAG pipeline
rag = RAGPipeline(
    embedding_model="sentence-transformers/all-MiniLM-L6-v2",
    llm_model="gpt-4o-mini"
)

print("✅ RAG Pipeline created!")
print(f"📊 Configuration: {rag.get_stats()}")
```

### Step 3: Run It

```bash
python my_first_rag.py
```

You should see:
```
✅ RAG Pipeline created!
📊 Configuration: {'embedding_model': 'sentence-transformers/all-MiniLM-L6-v2', ...}
```

🎉 **Congratulations!** You've created your first RAG pipeline.

## 🎯 Complete Example

Once the core functionality is implemented, here's what you'll be able to do:

```python
from pythonrag import RAGPipeline

# Initialize with your preferred models
rag = RAGPipeline(
    embedding_model="text-embedding-3-small",  # OpenAI embedding
    llm_model="gpt-4o-mini",                   # OpenAI LLM
    chunk_size=800,
    top_k=5
)

# Add some documents
documents = [
    "Python is a high-level programming language known for its simplicity.",
    "Machine learning is a subset of AI that enables computers to learn from data.",
    "RAG combines retrieval and generation for more accurate AI responses."
]

rag.add_documents(documents)

# Query your knowledge base
response = rag.query("What is Python used for?")
print(response)
```

## 🔧 Configuration Options

### Embedding Models

=== "Local (Free)"

    ```python
    rag = RAGPipeline(
        embedding_model="sentence-transformers/all-MiniLM-L6-v2"
    )
    ```

=== "OpenAI (Paid)"

    ```python
    # Set your API key first
    # export OPENAI_API_KEY="your-key"
    
    rag = RAGPipeline(
        embedding_model="text-embedding-3-small"
    )
    ```

=== "Hugging Face"

    ```python
    rag = RAGPipeline(
        embedding_model="BAAI/bge-large-en-v1.5"
    )
    ```

### Language Models

=== "OpenAI"

    ```python
    rag = RAGPipeline(
        llm_model="gpt-4o-mini"  # Cost-effective
        # llm_model="gpt-4o"     # Higher quality
    )
    ```

=== "Anthropic"

    ```python
    # Set your API key first
    # export ANTHROPIC_API_KEY="your-key"
    
    rag = RAGPipeline(
        llm_model="claude-3-haiku-20240307"
    )
    ```

=== "Local (Ollama)"

    ```python
    rag = RAGPipeline(
        llm_model="ollama:llama3.1:8b"
    )
    ```

### Vector Databases

=== "In-Memory (Default)"

    ```python
    rag = RAGPipeline(
        vector_db={"type": "in_memory"}
    )
    ```

=== "ChromaDB (Persistent)"

    ```python
    rag = RAGPipeline(
        vector_db={
            "type": "chroma",
            "persist_directory": "./my_vectordb"
        }
    )
    ```

=== "Pinecone (Cloud)"

    ```python
    rag = RAGPipeline(
        vector_db={
            "type": "pinecone",
            "api_key": "your-pinecone-key",
            "index_name": "my-index"
        }
    )
    ```

## 🎨 Common Patterns

### Pattern 1: Development Setup

Perfect for experimentation and learning:

```python
from pythonrag import RAGPipeline

# Free, local setup
rag = RAGPipeline(
    embedding_model="sentence-transformers/all-MiniLM-L6-v2",
    llm_model="gpt-4o-mini",  # You'll need OpenAI API key
    vector_db={"type": "in_memory"},
    chunk_size=500,
    top_k=3
)
```

### Pattern 2: Production Setup

Optimized for production use:

```python
from pythonrag import RAGPipeline

# Production-ready configuration
rag = RAGPipeline(
    embedding_model="text-embedding-3-small",
    llm_model="gpt-4o-mini",
    vector_db={
        "type": "chroma",
        "persist_directory": "./production_db"
    },
    chunk_size=1000,
    top_k=5
)
```

### Pattern 3: Privacy-First Setup

All processing stays local:

```python
from pythonrag import RAGPipeline

# Everything runs locally
rag = RAGPipeline(
    embedding_model="sentence-transformers/all-mpnet-base-v2",
    llm_model="ollama:llama3.1:8b",
    vector_db={
        "type": "chroma",
        "persist_directory": "./private_db"
    },
    chunk_size=1200,
    top_k=4
)
```

## 🧪 Testing Your Setup

### Method 1: Python Script

Create `test_rag.py`:

```python
from pythonrag import RAGPipeline

try:
    rag = RAGPipeline()
    print("✅ RAGPipeline created successfully")
    
    stats = rag.get_stats()
    print(f"📊 Configuration: {stats}")
    
    print("🎉 Setup is working!")
    
except Exception as e:
    print(f"❌ Error: {e}")
```

### Method 2: Interactive Python

```python
>>> from pythonrag import RAGPipeline
>>> rag = RAGPipeline()
>>> rag.get_stats()
{'embedding_model': 'sentence-transformers/all-MiniLM-L6-v2', ...}
```

### Method 3: Command Line

```bash
# Check if CLI works
pythonrag --version

# Create a pipeline via CLI
pythonrag create --embedding-model text-embedding-3-small
```

## 🔑 API Keys Setup

### OpenAI

1. Get your API key from [OpenAI Platform](https://platform.openai.com/api-keys)
2. Set it as an environment variable:

```bash
export OPENAI_API_KEY="sk-your-key-here"
```

Or in Python:
```python
import os
os.environ["OPENAI_API_KEY"] = "sk-your-key-here"
```

### Anthropic

```bash
export ANTHROPIC_API_KEY="your-anthropic-key"
```

### Pinecone

```bash
export PINECONE_API_KEY="your-pinecone-key"
```

## 🎯 Next Steps

Now that you have PythonRAG running, explore these areas:

### Learn the Basics
- **[OpenAI Integration](../examples/openai.md)** - Working with OpenAI models
- **[Qdrant Integration](../examples/qdrant.md)** - Using vector databases

### See Examples
- **[OpenAI Integration](../examples/openai.md)** - Using OpenAI models
- **[Qdrant Integration](../examples/qdrant.md)** - Vector database examples

### Advanced Topics
- **[API Reference](../api/core.md)** - Complete API documentation
- **[FAQ](../faq.md)** - Common questions and troubleshooting
- **[Changelog](../changelog.md)** - What's new in each version

## 🆘 Getting Help

### Something not working?

1. **Check the error message** - Often contains helpful information
2. **Review [Installation Guide](installation.md)** - Ensure proper setup
3. **Browse [FAQ](../faq.md)** - Common issues and solutions
4. **Check [GitHub Issues](https://github.com/franzejr/PythonRAG/issues)** - See if others had similar problems

### Example Error Solutions

#### `ModuleNotFoundError: No module named 'pythonrag'`
```bash
pip install pythonrag
# or if in development:
pip install -e .
```

#### `OpenAI API key not found`
```bash
export OPENAI_API_KEY="your-key-here"
```

#### `No module named 'sentence_transformers'`
```bash
pip install "pythonrag[embeddings]"
```

## 💡 Tips for Success

1. **Start Simple** - Use in-memory storage and local models first
2. **Test Incrementally** - Add one feature at a time
3. **Monitor Costs** - Be aware of API usage with paid services
4. **Use Virtual Environments** - Avoid dependency conflicts
5. **Check Documentation** - This documentation is your friend!

---

**Ready to dive deeper?** Check out our [OpenAI Integration guide](../examples/openai.md) to learn more! 
 