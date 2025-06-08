# PythonRAG Examples

This directory contains practical examples demonstrating how to use PythonRAG for various use cases.

## 🚀 Available Examples

### Basic Examples

- **[basic_usage.py](basic_usage.py)** - Simple RAG pipeline setup and usage
- **[openai_usage.py](openai_usage.py)** - OpenAI integration with cost optimization
- **[advanced_configurations.py](advanced_configurations.py)** - Various configuration patterns

### Vector Database Examples

- **[interactive_qdrant_rag.py](interactive_qdrant_rag.py)** ⭐ - **Interactive RAG system with Qdrant**
  - Complete RAG implementation
  - Document chunking and indexing
  - Interactive Q&A interface
  - Source attribution
  - Cost optimization
  - Error handling

- **[qdrant_pipeline_example.py](qdrant_pipeline_example.py)** ⭐ - **Framework Extension Example**
  - Shows how to extend base `RAGPipeline` class
  - Proper Qdrant integration architecture
  - Full implementation of all pipeline methods
  - Production-ready error handling

### Jupyter Notebooks

- **[getting_started.ipynb](getting_started.ipynb)** - Interactive tutorial notebook

## 🎯 Quick Start

### 1. Interactive Qdrant RAG (Recommended)

This is our most complete example showing a full RAG system:

```bash
# Install dependencies
pip install "pythonrag[all]" qdrant-client openai python-dotenv

# Set your OpenAI API key
export OPENAI_API_KEY="your-key-here"

# Run local Qdrant (optional - will connect to local by default)
docker run -p 6333:6333 qdrant/qdrant

# Run the interactive example
python examples/interactive_qdrant_rag.py
```

**Features:**
- 🤖 Powered by OpenAI GPT-4o-mini and text-embedding-3-small
- 🗄️ Qdrant vector database for high-performance search
- 📚 Automatic document chunking and indexing
- 💬 Interactive command-line interface
- 📊 Real-time statistics and monitoring
- 💰 Cost-optimized configuration

### 2. Basic OpenAI Example

Simple example to get started quickly:

```bash
export OPENAI_API_KEY="your-key-here"
python examples/openai_usage.py
```

### 3. Development Examples

For development and testing without external dependencies:

```bash
python examples/basic_usage.py
```

## 📋 Prerequisites

### Required for all examples:
- Python 3.8+
- PythonRAG installed (`pip install -e .` for development)

### For OpenAI examples:
- OpenAI API key
- `pip install openai`

### For Qdrant examples:
- Qdrant instance (local Docker or cloud)
- `pip install qdrant-client`
- Optional: Qdrant API key for cloud

### For advanced examples:
- Additional dependencies as noted in each example

## 🛠️ Example Details

### Interactive Qdrant RAG

**File:** `interactive_qdrant_rag.py`

This is our flagship example showing a complete RAG system. It includes:

- **Document Processing**: Automatic chunking with overlap
- **Embedding Generation**: OpenAI text-embedding-3-small
- **Vector Storage**: Qdrant for high-performance search
- **Answer Generation**: OpenAI GPT-4o-mini with context
- **Interactive Interface**: Command-line Q&A system
- **Source Attribution**: Shows which documents were used
- **Error Handling**: Graceful degradation and helpful messages

**Sample interaction:**
```
❓ Your question: What is the difference between AI and Machine Learning?

🔍 Searching for relevant context...
   Found 3 relevant chunks
🤖 Generating answer...

💡 Answer:
AI is the broader field while Machine Learning is a subset of AI...

📚 Sources:
1. [AI_Overview.md] (similarity: 0.857)
2. [ML_Guide.md] (similarity: 0.834)
```

### Cost Optimization

All examples follow cost optimization best practices:

- **text-embedding-3-small**: 5x cheaper than text-embedding-3-large
- **gpt-4o-mini**: 15x cheaper than gpt-4o
- **Efficient chunking**: Larger chunks reduce API calls
- **Persistent storage**: Avoid re-indexing documents

## 🔧 Configuration

### Environment Variables

```bash
# Required for OpenAI examples
export OPENAI_API_KEY="your-openai-api-key"

# Optional for Qdrant Cloud
export QDRANT_API_KEY="your-qdrant-api-key"
export QDRANT_URL="your-qdrant-cloud-url"
```

### Local Development

For development without external services:

```bash
# Use local models only
python examples/basic_usage.py

# Test core functionality
python examples/advanced_configurations.py
```

## 🐳 Docker Setup

### Qdrant (for vector database examples)

```bash
# Run Qdrant locally
docker pull qdrant/qdrant
docker run -p 6333:6333 qdrant/qdrant

# Verify it's running
curl http://localhost:6333/
```

## 📚 Documentation

Each example is documented in our main documentation:

- **[Qdrant Integration Guide](../docs/examples/qdrant.md)** - Comprehensive Qdrant guide
- **[OpenAI Integration Guide](../docs/examples/openai.md)** - OpenAI usage and optimization
- **[Getting Started](../docs/getting-started/quickstart.md)** - Quick start guide
- **[API Reference](../docs/api/core.md)** - Complete API documentation

## 🆘 Troubleshooting

### Common Issues

#### `ModuleNotFoundError: No module named 'pythonrag'`
```bash
# Install in development mode
pip install -e .
```

#### `OPENAI_API_KEY not found`
```bash
# Set your API key
export OPENAI_API_KEY="your-key-here"
```

#### `Failed to connect to Qdrant`
```bash
# Start local Qdrant
docker run -p 6333:6333 qdrant/qdrant

# Or check if running
docker ps
curl http://localhost:6333/
```

#### `ImportError: Missing required package`
```bash
# Install all dependencies
pip install "pythonrag[all]" qdrant-client openai python-dotenv
```

### Getting Help

1. Check the [FAQ](../docs/faq.md)
2. Review [documentation](../docs/)
3. Open an [issue](https://github.com/franzejr/PythonRAG/issues)

## 🎯 Next Steps

1. **Start with the interactive example**: It's the most complete demonstration
2. **Read the documentation**: Comprehensive guides for each feature
3. **Explore configurations**: Try different models and settings
4. **Build your own**: Use these examples as templates for your projects

Happy building with PythonRAG! 🚀 
 