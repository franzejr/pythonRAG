# PythonRAG

[![CI](https://github.com/franzejr/pythonRAG/actions/workflows/ci.yml/badge.svg)](https://github.com/franzejr/pythonRAG/actions/workflows/ci.yml)
[![Documentation](https://github.com/franzejr/pythonRAG/actions/workflows/docs.yml/badge.svg)](https://github.com/franzejr/pythonRAG/actions/workflows/docs.yml)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> **🚀 Modern Python package for Retrieval-Augmented Generation (RAG) workflows**

PythonRAG provides a simple, extensible framework for building RAG applications with support for multiple embedding models, language models, and vector databases. Designed with modern Python packaging standards and comprehensive testing.

## Overview

PythonRAG is a comprehensive toolkit that simplifies the implementation of RAG systems in Python. It provides a clean, intuitive API for building retrieval-augmented generation pipelines with support for various vector databases, embedding models, and language models.

## Features

- 🚀 **Easy to Use**: Simple API for quick RAG implementation
- 🔧 **Flexible**: Support for multiple vector databases and embedding models  
- 📚 **Comprehensive**: Built-in document processing and chunking strategies
- ⚡ **Performance**: Optimized for speed and efficiency
- 🛠️ **Extensible**: Plugin architecture for custom components

## Architecture

PythonRAG follows an **interface-based architecture** that promotes modularity and extensibility. The core `RAGPipeline` class acts as a contract/interface that defines the standard methods all RAG implementations must provide:

```python
class RAGPipeline:
    def add_documents(self, documents, metadata=None) -> None
    def query(self, question, top_k=None) -> str  
    def add_document_file(self, file_path, metadata=None) -> None
    def get_stats(self) -> Dict[str, Any]
    def reset(self) -> None
```

This design enables:
- **🔄 Easy swapping** between different vector databases (Qdrant, Chroma, Pinecone)
- **🧪 Consistent testing** with the same interface across implementations  
- **🏗️ Clean extensions** by inheriting from the base pipeline
- **📈 Future growth** without breaking existing code

### Example: Extending with Qdrant

```python
class QdrantRAGPipeline(RAGPipeline):
    """RAG implementation with Qdrant vector database"""
    
    def add_documents(self, documents, metadata=None):
        # Custom Qdrant implementation
        pass
    
    def query(self, question, top_k=None):
        # Custom Qdrant search + LLM generation
        pass
```

**📋 See [architecture_idea.md](architecture_idea.md) for detailed architecture documentation and examples.**

## Installation

```bash
pip install pythonrag
```

## Quick Start

```python
from pythonrag import RAGPipeline

# Initialize a RAG pipeline
rag = RAGPipeline(
    embedding_model="sentence-transformers/all-MiniLM-L6-v2",
    llm_model="gpt-3.5-turbo"
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

## Documentation

- 📖 **Full documentation**: Available at [docs link] (auto-deployed via GitHub Actions)
- 🏗️ **Architecture guide**: [architecture_idea.md](architecture_idea.md) - Detailed explanation of the interface-based design pattern
- 📚 **Examples**: [examples/](examples/) - Working examples with different vector databases
- 🔧 **API Reference**: [docs/api/](docs/api/) - Complete API documentation

## Requirements

- Python 3.8+
- See `requirements.txt` for detailed dependencies

## Development

### Setup Development Environment

```bash
# Clone the repository
git clone https://github.com/franzejr/PythonRAG.git
cd PythonRAG

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install in development mode
pip install -e ".[dev]"
```

### Running Tests

```bash
pytest
```

### Code Style

This project uses `black` for code formatting and `ruff` for linting:

```bash
black .
ruff check .
```

## Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Roadmap

- [ ] Support for more vector databases
- [ ] Advanced chunking strategies
- [ ] Multi-modal RAG support
- [ ] Integration with popular LLM frameworks
- [ ] Performance optimizations

## Support

- 💬 Discussions: [GitHub Discussions](https://github.com/franzejr/PythonRAG/discussions)
- 🐛 Issues: [GitHub Issues](https://github.com/franzejr/PythonRAG/issues)

## Acknowledgments

- Thanks to the open source community
- Built with modern Python packaging standards
- Inspired by the growing RAG ecosystem 
