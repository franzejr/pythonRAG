# Changelog

All notable changes to PythonRAG will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Advanced chunking strategies
- Multi-modal RAG support
- Performance optimizations
- Integration with popular LLM frameworks
- Advanced retrieval techniques (hybrid search, re-ranking)

## [0.1.0] - 2024-01-15

### Added
- Initial release of PythonRAG
- Core RAG pipeline implementation (`RAGPipeline` class)
- Support for multiple embedding models:
  - OpenAI: `text-embedding-3-small`, `text-embedding-3-large`
  - Sentence Transformers: All models from the library
  - Hugging Face: BGE, E5, and custom models
- Support for multiple LLMs:
  - OpenAI: GPT-4o, GPT-4o-mini, GPT-3.5-turbo
  - Anthropic: Claude 3.5 Sonnet, Claude 3 Haiku
  - Local models: Ollama integration
- Vector database support:
  - In-memory storage (default)
  - ChromaDB for persistent storage
  - Pinecone for cloud-native vector search
  - Weaviate integration
  - Qdrant support
- Document processing and chunking
- Configurable retrieval parameters
- Custom exception handling
- Command-line interface
- Comprehensive examples and documentation
- Development tools and testing framework

### Core Features
- **Document Management**:
  - `add_documents()` - Add multiple text documents
  - `add_document_file()` - Add documents from files
  - Automatic text chunking with configurable size and overlap
  - Metadata support for document filtering

- **Querying**:
  - `query()` - Natural language queries with context retrieval
  - Configurable top-k retrieval
  - Context-aware response generation

- **Configuration**:
  - `get_stats()` - Pipeline statistics and configuration
  - `reset()` - Clear all documents and reset pipeline
  - Flexible configuration options for all components

- **Error Handling**:
  - Custom exception hierarchy
  - Graceful error recovery
  - Detailed error messages

### CLI Features
- `pythonrag --version` - Show version information
- `pythonrag --help` - Display help information
- `pythonrag create` - Interactive pipeline creation

### Dependencies
- Python 3.8+ support
- Optional dependencies for different features:
  - `[embeddings]` - Local embedding models
  - `[vectordb]` - Vector database integrations
  - `[llm]` - Language model integrations
  - `[dev]` - Development tools
  - `[docs]` - Documentation building
  - `[all]` - All features

### Examples
- Basic usage example
- OpenAI integration with cost optimization
- Advanced configurations showcase
- Jupyter notebook tutorial

### Testing
- Comprehensive test suite with pytest
- 8 passing tests, 2 skipped
- Development installation support
- Multiple testing methods documented

### Documentation
- Complete API reference
- Getting started guides
- Configuration documentation
- FAQ and troubleshooting
- Examples and tutorials

## Development Process

### Release Process
1. Update version in `src/pythonrag/__init__.py`
2. Update `CHANGELOG.md` with new features and fixes
3. Create git tag: `git tag v0.1.0`
4. Push to GitHub: `git push origin v0.1.0`
5. Build and publish to PyPI:
   ```bash
   python -m build
   python -m twine upload dist/*
   ```

### Version Policy
- **Major version** (1.0.0): Breaking API changes
- **Minor version** (0.1.0): New features, backward compatible
- **Patch version** (0.1.1): Bug fixes, backward compatible

### Contributing
For questions and support, see our [FAQ](faq.md) for information on:
- Setting up development environment
- Running tests
- Code style and formatting
- Submitting pull requests

---

**Note**: This project is under active development. Features marked as "Added" in unreleased versions are planned but not yet implemented. Check the [GitHub repository](https://github.com/franzejr/PythonRAG) for the latest development status. 
 