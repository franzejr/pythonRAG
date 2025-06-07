# Installation

This guide covers how to install PythonRAG and its dependencies for different use cases.

## Quick Installation

=== "Basic Installation"

    For most users, the basic installation includes core dependencies:

    ```bash
    pip install pythonrag
    ```

=== "Full Installation"

    For all features including OpenAI, vector databases, and development tools:

    ```bash
    pip install "pythonrag[all]"
    ```

=== "Development Installation"

    For contributing to PythonRAG:

    ```bash
    git clone https://github.com/franzejr/PythonRAG.git
    cd PythonRAG
    pip install -e ".[dev]"
    ```

## Requirements

### Python Version

PythonRAG requires **Python 3.8 or later**. We recommend using Python 3.10+ for the best experience.

```bash
python --version  # Should be 3.8+
```

### System Requirements

- **Memory**: Minimum 4GB RAM (8GB+ recommended for large document sets)
- **Storage**: At least 1GB free space for models and data
- **Network**: Internet connection for API-based models (OpenAI, Anthropic)

## Installation Options

### 🚀 Option 1: Basic Installation

The basic installation includes core functionality:

```bash
pip install pythonrag
```

**Included:**
- Core RAG pipeline
- In-memory vector storage
- Basic document processing
- CLI interface

### 🧠 Option 2: With Embedding Models

For local embedding generation:

```bash
pip install "pythonrag[embeddings]"
```

**Adds:**
- Sentence Transformers
- Hugging Face Transformers
- PyTorch
- Local embedding generation

### 🗄️ Option 3: With Vector Databases

For persistent vector storage:

```bash
pip install "pythonrag[vectordb]"
```

**Adds:**
- ChromaDB
- Pinecone client
- Weaviate client
- Qdrant client

### 🤖 Option 4: With LLM Integrations

For language model integrations:

```bash
pip install "pythonrag[llm]"
```

**Adds:**
- OpenAI client
- Anthropic client
- LangChain integration

### 🎯 Option 5: Complete Installation

For all features:

```bash
pip install "pythonrag[all]"
```

**Includes everything above plus:**
- Development tools
- Documentation dependencies
- Testing framework

## Development Installation

### Prerequisites

- Git
- Python 3.8+
- pip

### Setup

1. **Clone the repository:**

    ```bash
    git clone https://github.com/franzejr/PythonRAG.git
    cd PythonRAG
    ```

2. **Create virtual environment (recommended):**

    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3. **Install in development mode:**

    ```bash
    pip install -e ".[dev]"
    ```

4. **Set up pre-commit hooks:**

    ```bash
    pre-commit install
    ```

5. **Verify installation:**

    ```bash
    pytest tests/
    python examples/basic_usage.py
    ```

## Virtual Environment Setup

We **strongly recommend** using a virtual environment:

=== "venv (built-in)"

    ```bash
    python -m venv pythonrag-env
    source pythonrag-env/bin/activate  # Linux/Mac
    # or
    pythonrag-env\Scripts\activate     # Windows
    
    pip install pythonrag
    ```

=== "conda"

    ```bash
    conda create -n pythonrag python=3.10
    conda activate pythonrag
    
    pip install pythonrag
    ```

=== "pyenv + virtualenv"

    ```bash
    pyenv install 3.10.12
    pyenv virtualenv 3.10.12 pythonrag
    pyenv activate pythonrag
    
    pip install pythonrag
    ```

## Platform-Specific Instructions

### 🐧 Linux

```bash
# Update package list
sudo apt update

# Install Python and pip (if not already installed)
sudo apt install python3 python3-pip

# Install PythonRAG
pip3 install pythonrag
```

### 🍎 macOS

```bash
# Install via Homebrew (if Python not installed)
brew install python

# Install PythonRAG
pip3 install pythonrag
```

### 🪟 Windows

1. **Install Python** from [python.org](https://www.python.org/downloads/) (3.8+)
2. **Open Command Prompt or PowerShell**
3. **Install PythonRAG:**

    ```cmd
    pip install pythonrag
    ```

## Verification

After installation, verify that PythonRAG is working:

### Basic Test

```python
from pythonrag import RAGPipeline

# Create a pipeline
rag = RAGPipeline()
print("✅ PythonRAG installed successfully!")

# Check configuration
stats = rag.get_stats()
print(f"📊 Configuration: {stats}")
```

### CLI Test

```bash
pythonrag --version
pythonrag --help
```

### Run Examples

```bash
python -c "from pythonrag import RAGPipeline; print('✅ Import successful')"
```

## Troubleshooting

### Common Issues

#### `ModuleNotFoundError: No module named 'pythonrag'`

**Solutions:**
1. Ensure you're in the correct virtual environment
2. Reinstall: `pip install --upgrade pythonrag`
3. Check Python path: `python -c "import sys; print(sys.path)"`

#### Permission Errors

**On Linux/macOS:**
```bash
pip install --user pythonrag
```

**On Windows:**
Run Command Prompt as Administrator

#### Version Conflicts

```bash
pip install --upgrade pip
pip install --upgrade pythonrag
```

#### SSL Certificate Errors

```bash
pip install --trusted-host pypi.org --trusted-host pypi.python.org pythonrag
```

### Getting Help

If you encounter issues:

1. **Check our [FAQ](../faq.md)**
2. **Search [GitHub Issues](https://github.com/franzejr/PythonRAG/issues)**
3. **Create a new issue** with:
   - Python version (`python --version`)
   - PythonRAG version (`pip show pythonrag`)
   - Error message and stack trace
   - Operating system

## Next Steps

Now that PythonRAG is installed:

1. **[Quick Start Guide](quickstart.md)** - Build your first RAG pipeline
2. **[Basic Usage](basic-usage.md)** - Learn core concepts
3. **[Configuration](../user-guide/configuration.md)** - Customize for your needs
4. **[Examples](../examples/basic.md)** - See PythonRAG in action

## Optional Dependencies

### For OpenAI Integration

```bash
pip install openai
export OPENAI_API_KEY="your-api-key"
```

### For Local Models

```bash
# For sentence transformers
pip install sentence-transformers

# For Ollama
pip install ollama
```

### For Advanced Vector Databases

```bash
# Pinecone
pip install pinecone-client

# Weaviate
pip install weaviate-client

# Qdrant
pip install qdrant-client
```

---

**Ready to get started?** Continue to the [Quick Start Guide](quickstart.md)! 
