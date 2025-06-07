# Testing Guide for PythonRAG

This guide explains how to test your PythonRAG package during development and after installation.

## 🚀 Quick Start - Testing Your Package

### Method 1: Development Installation (Recommended)

Install your package in development mode so you can import and test it:

```bash
# Install in development mode (editable install)
pip install -e .

# Now you can run examples and tests
python examples/basic_usage.py
python examples/openai_usage.py
pytest tests/
```

**Why this works:** The `-e` flag creates an "editable" installation that links to your source code, so changes are immediately available.

### Method 2: Using PYTHONPATH (Alternative)

If you don't want to install the package, you can add the source directory to Python's path:

```bash
# Add src directory to Python path
export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"

# Now run examples
python examples/basic_usage.py
```

### Method 3: Direct Import from Source

Modify your examples to import directly from the source:

```python
import sys
from pathlib import Path

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from pythonrag import RAGPipeline
```

## 🧪 Running Tests

### Basic Test Execution

```bash
# Run all tests
pytest

# Run with verbose output
pytest -v

# Run specific test file
pytest tests/test_core.py

# Run specific test method
pytest tests/test_core.py::TestRAGPipeline::test_init_default_parameters
```

### Test Coverage

```bash
# Run tests with coverage report
pytest --cov=pythonrag

# Generate HTML coverage report
pytest --cov=pythonrag --cov-report=html

# View coverage report
open htmlcov/index.html
```

### Using Make Commands

```bash
# Run tests (defined in Makefile)
make test

# Run tests with coverage
make test-cov

# Run all quality checks
make check
```

## 📋 Testing Examples

### 1. Basic Usage Test

```bash
python examples/basic_usage.py
```

**Expected Output:**
```
PythonRAG Basic Usage Example
========================================
1. Initializing RAG pipeline...
2. Pipeline configuration:
   embedding_model: sentence-transformers/all-MiniLM-L6-v2
   llm_model: gpt-3.5-turbo
   vector_db_type: in_memory
   chunk_size: 1000
   chunk_overlap: 200
   top_k: 5

3. Adding documents (not yet implemented)...
   Expected error: Document addition will be implemented in the next iteration

4. Querying the system (not yet implemented)...
   Expected error: Query processing will be implemented in the next iteration

5. Example completed!
```

### 2. OpenAI Integration Test

```bash
python examples/openai_usage.py
```

**Without API Key:**
```
❌ OpenAI API key not found!
   Please set your API key: export OPENAI_API_KEY='your-api-key-here'
```

**With API Key:**
```bash
export OPENAI_API_KEY="your-actual-api-key"
python examples/openai_usage.py
```

### 3. CLI Testing

```bash
# Test CLI help
pythonrag --help

# Test CLI commands
pythonrag create --embedding-model text-embedding-3-small
pythonrag --version
```

### 4. Advanced Configuration Test

```bash
python examples/advanced_configurations.py
```

This shows all available configuration options and validates different setups.

## 🔧 Development Testing Workflow

### 1. Install Development Dependencies

```bash
# Install all development dependencies
pip install -e ".[dev]"

# Or use requirements file
pip install -r requirements-dev.txt
```

### 2. Set Up Pre-commit Hooks

```bash
# Install pre-commit hooks
pre-commit install

# Run pre-commit on all files
pre-commit run --all-files
```

### 3. Code Quality Checks

```bash
# Format code
black src/ tests/ examples/
ruff check --fix src/ tests/ examples/

# Type checking
mypy src/

# Or use make commands
make format
make lint
```

### 4. Full Development Setup

```bash
# Complete development setup
make setup-dev

# This runs:
# - pip install -e ".[dev]"
# - pip install -r requirements-dev.txt
# - pre-commit install
```

## 🐛 Troubleshooting Common Issues

### Issue: `ModuleNotFoundError: No module named 'pythonrag'`

**Solutions:**
1. Install in development mode: `pip install -e .`
2. Add to PYTHONPATH: `export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"`
3. Check you're in the right directory (should contain `pyproject.toml`)

### Issue: Import errors in tests

**Solutions:**
1. Make sure you've installed the package: `pip install -e .`
2. Check that `src/pythonrag/__init__.py` exists
3. Verify the package structure is correct

### Issue: Tests fail with missing dependencies

**Solutions:**
1. Install test dependencies: `pip install -e ".[dev]"`
2. Install from requirements: `pip install -r requirements-dev.txt`

### Issue: OpenAI examples don't work

**Solutions:**
1. Set API key: `export OPENAI_API_KEY="your-key"`
2. Install OpenAI package: `pip install openai`
3. The examples are designed to show what will work when implemented

## 📊 Test Coverage Goals

Current test coverage:
- `src/pythonrag/__init__.py`: 100%
- `src/pythonrag/core.py`: 100%
- `src/pythonrag/exceptions.py`: 100%
- `src/pythonrag/cli.py`: 0% (will be tested when functionality is implemented)

**Target:** Maintain >90% test coverage as functionality is implemented.

## 🚀 Continuous Integration

When you push to GitHub, consider setting up:

1. **GitHub Actions** for automated testing
2. **Codecov** for coverage reporting
3. **Pre-commit.ci** for code quality

Example GitHub Actions workflow:

```yaml
name: Tests
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: [3.8, 3.9, "3.10", "3.11", "3.12"]
    steps:
    - uses: actions/checkout@v3
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: ${{ matrix.python-version }}
    - name: Install dependencies
      run: |
        pip install -e ".[dev]"
    - name: Run tests
      run: |
        pytest --cov=pythonrag
```

## 📝 Adding New Tests

When implementing new functionality:

1. **Write tests first** (TDD approach)
2. **Test both success and failure cases**
3. **Mock external dependencies** (APIs, file systems)
4. **Update test coverage goals**

Example test structure:
```python
def test_new_functionality(self):
    """Test description."""
    # Arrange
    rag = RAGPipeline()
    
    # Act
    result = rag.new_method()
    
    # Assert
    assert result == expected_value
```

## 🎯 Next Steps

1. **Implement core functionality** and add corresponding tests
2. **Set up CI/CD pipeline** for automated testing
3. **Add integration tests** with real APIs (using test keys)
4. **Performance testing** for large document sets
5. **End-to-end testing** with complete RAG workflows

---

**Happy Testing! 🧪✨** 
