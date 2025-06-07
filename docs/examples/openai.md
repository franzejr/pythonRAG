# OpenAI Integration Examples

This page shows how to integrate PythonRAG with OpenAI's models for embeddings and language generation.

## Prerequisites

1. **OpenAI API Key**: Get one from [OpenAI Platform](https://platform.openai.com/api-keys)
2. **Install OpenAI package**: `pip install openai`
3. **Set environment variable**: `export OPENAI_API_KEY="your-key"`

## Basic OpenAI Setup

```python
import os
from pythonrag import RAGPipeline

# Ensure API key is set
assert os.getenv("OPENAI_API_KEY"), "Please set OPENAI_API_KEY environment variable"

# Create RAG pipeline with OpenAI models
rag = RAGPipeline(
    embedding_model="text-embedding-3-small",
    llm_model="gpt-4o-mini"
)
```

## Embedding Models

### text-embedding-3-small (Recommended)

Cost-effective and high-quality:

```python
rag = RAGPipeline(
    embedding_model="text-embedding-3-small",
    llm_model="gpt-4o-mini"
)
```

**Specifications:**
- **Dimensions**: 1536
- **Cost**: ~$0.00002 per 1K tokens
- **Use case**: General purpose, cost-effective

### text-embedding-3-large

Highest quality embeddings:

```python
rag = RAGPipeline(
    embedding_model="text-embedding-3-large",
    llm_model="gpt-4o"
)
```

**Specifications:**
- **Dimensions**: 3072
- **Cost**: ~$0.00013 per 1K tokens
- **Use case**: Maximum accuracy, critical applications

## Language Models

### GPT-4o-mini (Recommended)

Best balance of cost and performance:

```python
rag = RAGPipeline(
    embedding_model="text-embedding-3-small",
    llm_model="gpt-4o-mini",
    chunk_size=800,
    top_k=5
)
```

**Specifications:**
- **Input**: $0.15 per 1M tokens
- **Output**: $0.60 per 1M tokens
- **Context window**: 128K tokens
- **Use case**: General purpose, cost-effective

### GPT-4o

Highest capability model:

```python
rag = RAGPipeline(
    embedding_model="text-embedding-3-large",
    llm_model="gpt-4o",
    chunk_size=1000,
    top_k=8
)
```

**Specifications:**
- **Input**: $2.50 per 1M tokens
- **Output**: $10.00 per 1M tokens
- **Context window**: 128K tokens
- **Use case**: Complex reasoning, highest quality

## Complete Example

```python
import os
from pythonrag import RAGPipeline

# Check API key
if not os.getenv("OPENAI_API_KEY"):
    print("❌ Please set OPENAI_API_KEY environment variable")
    exit(1)

# Create pipeline
rag = RAGPipeline(
    embedding_model="text-embedding-3-small",
    llm_model="gpt-4o-mini",
    chunk_size=800,
    chunk_overlap=100,
    top_k=5
)

# Sample documents about AI
documents = [
    """
    Artificial Intelligence (AI) is a broad field of computer science focused on 
    building smart machines capable of performing tasks that typically require 
    human intelligence. These tasks include learning, reasoning, problem-solving, 
    perception, and language understanding.
    """,
    """
    Machine Learning is a subset of AI that enables computers to learn and improve 
    from experience without being explicitly programmed. It uses algorithms and 
    statistical models to analyze and draw inferences from patterns in data.
    """,
    """
    Large Language Models (LLMs) like GPT-4 are AI systems trained on vast amounts 
    of text data. They can understand and generate human-like text, making them 
    useful for tasks like writing, translation, summarization, and question-answering.
    """,
    """
    Retrieval-Augmented Generation (RAG) combines the strengths of retrieval-based 
    and generation-based approaches. It first retrieves relevant information from 
    a knowledge base, then uses this information to generate more accurate and 
    contextually relevant responses.
    """
]

# Add documents to the RAG system
print("📚 Adding documents...")
rag.add_documents(documents)

# Query the system
questions = [
    "What is the difference between AI and Machine Learning?",
    "How do Large Language Models work?",
    "What are the benefits of RAG systems?",
    "What tasks can AI systems perform?"
]

print("\n🔍 Querying the system...")
for question in questions:
    print(f"\n❓ Question: {question}")
    try:
        response = rag.query(question)
        print(f"✅ Answer: {response}")
    except Exception as e:
        print(f"❌ Error: {e}")

# Get pipeline statistics
print(f"\n📊 Pipeline stats: {rag.get_stats()}")
```

## Cost Optimization Strategies

### 1. Choose the Right Models

```python
# Cost-effective setup
rag_cheap = RAGPipeline(
    embedding_model="text-embedding-3-small",  # 5x cheaper than large
    llm_model="gpt-4o-mini",                   # 15x cheaper than gpt-4o
)

# High-performance setup
rag_premium = RAGPipeline(
    embedding_model="text-embedding-3-large",
    llm_model="gpt-4o",
)
```

### 2. Optimize Chunk Sizes

```python
# Larger chunks = fewer embedding API calls
rag = RAGPipeline(
    embedding_model="text-embedding-3-small",
    llm_model="gpt-4o-mini",
    chunk_size=1200,  # Larger chunks
    chunk_overlap=150,
    top_k=3           # Fewer chunks in context
)
```

### 3. Cache Embeddings

```python
# Use persistent vector database to avoid re-computing embeddings
rag = RAGPipeline(
    embedding_model="text-embedding-3-small",
    llm_model="gpt-4o-mini",
    vector_db={
        "type": "chroma",
        "persist_directory": "./embedding_cache"
    }
)
```

### 4. Monitor Usage

```python
import openai

# Set up usage tracking
openai.api_key = os.getenv("OPENAI_API_KEY")

# Monitor your usage on OpenAI dashboard
# https://platform.openai.com/usage
```

## Error Handling

```python
from pythonrag.exceptions import EmbeddingError, LLMError

try:
    rag = RAGPipeline(
        embedding_model="text-embedding-3-small",
        llm_model="gpt-4o-mini"
    )
    
    rag.add_documents(["Sample document"])
    response = rag.query("Sample question")
    
except EmbeddingError as e:
    print(f"Embedding error: {e}")
    print("Check your OpenAI API key and quota")
    
except LLMError as e:
    print(f"LLM error: {e}")
    print("Check your OpenAI API key and model availability")
    
except Exception as e:
    print(f"Unexpected error: {e}")
```

## Advanced Configuration

### Custom System Prompts

```python
# Custom configuration for specific use cases
rag = RAGPipeline(
    embedding_model="text-embedding-3-small",
    llm_model="gpt-4o-mini",
    system_prompt="""You are a helpful AI assistant specializing in 
    technical documentation. Provide accurate, concise answers based 
    on the retrieved context. If information is not in the context, 
    say so clearly."""
)
```

### Fine-tuned Models

```python
# Use your fine-tuned OpenAI model
rag = RAGPipeline(
    embedding_model="text-embedding-3-small",
    llm_model="ft:gpt-4o-mini:your-org:model-name:suffix",
)
```

### Streaming Responses

```python
# Enable streaming for long responses
rag = RAGPipeline(
    embedding_model="text-embedding-3-small",
    llm_model="gpt-4o-mini",
    stream_response=True
)

# Get streaming response
for chunk in rag.query_stream("Tell me about machine learning"):
    print(chunk, end="", flush=True)
```

## Performance Tips

### 1. Batch Processing

```python
# Process multiple documents at once
documents = [f"Document {i} content..." for i in range(100)]
rag.add_documents(documents)  # Batched API calls
```

### 2. Parallel Processing

```python
import asyncio
from pythonrag import AsyncRAGPipeline

async def main():
    rag = AsyncRAGPipeline(
        embedding_model="text-embedding-3-small",
        llm_model="gpt-4o-mini"
    )
    
    # Parallel queries
    tasks = [
        rag.query_async("Question 1"),
        rag.query_async("Question 2"),
        rag.query_async("Question 3"),
    ]
    
    responses = await asyncio.gather(*tasks)
    return responses

# Run async
responses = asyncio.run(main())
```

### 3. Connection Pooling

```python
# Configure connection pooling for high-throughput applications
rag = RAGPipeline(
    embedding_model="text-embedding-3-small",
    llm_model="gpt-4o-mini",
    openai_config={
        "max_retries": 3,
        "timeout": 30,
        "max_connections": 20,
        "max_keepalive_connections": 5
    }
)
```

## Troubleshooting

### Common Issues

#### API Key Not Found
```bash
export OPENAI_API_KEY="your-actual-api-key"
# or
echo 'export OPENAI_API_KEY="your-key"' >> ~/.bashrc
source ~/.bashrc
```

#### Rate Limits
```python
from openai import RateLimitError

try:
    response = rag.query("question")
except RateLimitError:
    print("Rate limit exceeded. Please wait and try again.")
    time.sleep(60)  # Wait 1 minute
```

#### Invalid Model
```python
# Check available models
import openai
models = openai.models.list()
print([model.id for model in models.data])
```

## Cost Calculator

```python
def estimate_cost(num_documents, avg_doc_length, num_queries):
    """Estimate OpenAI costs for RAG pipeline"""
    
    # Embedding costs (text-embedding-3-small)
    embedding_tokens = num_documents * (avg_doc_length / 4)  # ~4 chars per token
    embedding_cost = (embedding_tokens / 1000) * 0.00002
    
    # LLM costs (gpt-4o-mini)
    # Assume avg 500 tokens input, 100 tokens output per query
    input_tokens = num_queries * 500
    output_tokens = num_queries * 100
    llm_cost = (input_tokens / 1000000) * 0.15 + (output_tokens / 1000000) * 0.60
    
    total_cost = embedding_cost + llm_cost
    
    print(f"📊 Cost Estimate:")
    print(f"   Embeddings: ${embedding_cost:.4f}")
    print(f"   LLM: ${llm_cost:.4f}")
    print(f"   Total: ${total_cost:.4f}")
    
    return total_cost

# Example usage
estimate_cost(
    num_documents=1000,
    avg_doc_length=2000,
    num_queries=100
)
```

## Next Steps

- **[Vector Databases](../user-guide/vector-databases.md)** - Choose the right storage
- **[Configuration](../user-guide/configuration.md)** - Detailed configuration options
- **[Advanced Examples](advanced.md)** - Complex use cases
- **[Performance Tuning](../user-guide/performance.md)** - Optimize for production 
