# PythonRAG

<div align="center">

**A modern Python package for Retrieval-Augmented Generation (RAG) workflows**

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

[🚀 Quick Start](getting-started/quickstart.md){ .md-button .md-button--primary }
[📖 Examples](examples/openai.md){ .md-button }
[🔧 API Reference](api/core.md){ .md-button }

</div>

## Overview

PythonRAG is a comprehensive toolkit that simplifies the implementation of RAG systems in Python. It provides a clean, intuitive API for building retrieval-augmented generation pipelines with support for various vector databases, embedding models, and language models.

## ✨ Features

- **🚀 Easy to Use**: Simple API for quick RAG implementation
- **🔧 Flexible**: Support for multiple vector databases and embedding models  
- **📚 Comprehensive**: Built-in document processing and chunking strategies
- **⚡ Performance**: Optimized for speed and efficiency
- **🛠️ Extensible**: Plugin architecture for custom components
- **💰 Cost-Aware**: Built-in optimization for OpenAI and other paid APIs
- **🔒 Privacy-First**: Support for local models and on-premise deployment

## 🎯 Quick Example

```python
from pythonrag import RAGPipeline

# Initialize a RAG pipeline
rag = RAGPipeline(
    embedding_model="sentence-transformers/all-MiniLM-L6-v2",
    llm_model="gpt-4o-mini"
)

# Add documents
rag.add_documents([
    "Document content here...",
    "More document content..."
])

# Query the system
response = rag.query("What is the main topic of the documents?")
print(response)
```

## 🎨 Supported Integrations

### 🤖 Language Models
- **OpenAI**: GPT-4o, GPT-4o-mini, GPT-3.5-turbo
- **Anthropic**: Claude 3.5 Sonnet, Claude 3 Haiku
- **Local Models**: Ollama, Hugging Face Transformers
- **Other APIs**: Compatible with any OpenAI-compatible API

### 🔗 Embedding Models
- **OpenAI**: text-embedding-3-small, text-embedding-3-large
- **Sentence Transformers**: All models from the library
- **Hugging Face**: BGE, E5, and custom models
- **Local Options**: Privacy-focused local embedding generation

### 🗄️ Vector Databases
- **ChromaDB**: Local persistent storage
- **Pinecone**: Cloud-native vector database
- **Weaviate**: Open-source vector database
- **Qdrant**: High-performance vector search
- **In-Memory**: For rapid prototyping

## 📋 Use Cases

### 🎓 **Educational**
- Build Q&A systems over course materials
- Create interactive study assistants
- Generate summaries of academic papers

### 💼 **Business**
- Customer support knowledge bases
- Document analysis and insights
- Automated report generation

### 🔬 **Research**
- Literature review assistants
- Data analysis with natural language
- Experiment documentation and querying

### 🛠️ **Development**
- Code documentation search
- API reference assistants
- Technical knowledge management

## 🚀 Getting Started

1. **[Installation](getting-started/installation.md)** - Install PythonRAG and dependencies
2. **[Quick Start](getting-started/quickstart.md)** - Your first RAG pipeline in 5 minutes
3. **[OpenAI Integration](examples/openai.md)** - Working with OpenAI models
4. **[Vector Databases](examples/qdrant.md)** - Using Qdrant for vector storage

## 💡 Why PythonRAG?

### **Developer-Friendly**
- Clean, intuitive API design
- Comprehensive documentation and examples
- Type hints and IDE support
- Extensive testing and quality assurance

### **Production-Ready**
- Error handling and logging
- Performance optimizations
- Monitoring and observability hooks
- Scalable architecture patterns

### **Cost-Optimized**
- Smart chunking strategies
- Efficient embedding caching
- Model selection guidance
- Usage monitoring and alerts

### **Flexible & Extensible**
- Plugin architecture for custom components
- Support for multiple providers
- Easy migration between services
- Custom retrieval strategies

## 🎯 Roadmap

- [x] Core RAG pipeline implementation
- [x] OpenAI integration
- [x] Multiple vector database support
- [ ] Advanced chunking strategies
- [ ] Multi-modal RAG support
- [ ] Performance optimizations
- [ ] Integration with popular LLM frameworks
- [ ] Advanced retrieval techniques (hybrid search, re-ranking)

## 🤝 Community

- **💬 Discussions**: [GitHub Discussions](https://github.com/franzejr/PythonRAG/discussions)
- **🐛 Issues**: [GitHub Issues](https://github.com/franzejr/PythonRAG/issues)
- **📖 Documentation**: You're reading it!

## 📄 License

PythonRAG is released under the [MIT License](https://github.com/franzejr/PythonRAG/blob/main/LICENSE).

---

<div align="center">

**Ready to build amazing RAG applications?**

[Get Started Now](getting-started/installation.md){ .md-button .md-button--primary }

</div> 
