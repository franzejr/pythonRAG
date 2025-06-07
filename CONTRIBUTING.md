# Contributing to PythonRAG

Thank you for your interest in contributing to PythonRAG! 🎉 

We welcome contributions of all kinds: bug reports, feature requests, documentation improvements, code contributions, and more.

## 📋 Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [Making Changes](#making-changes)
- [Testing](#testing)
- [Code Style](#code-style)
- [Submitting Changes](#submitting-changes)
- [Documentation](#documentation)
- [Release Process](#release-process)
- [Getting Help](#getting-help)

## 📜 Code of Conduct

By participating in this project, you agree to abide by our Code of Conduct:

### Our Pledge

We pledge to make participation in our project a harassment-free experience for everyone, regardless of age, body size, disability, ethnicity, gender identity and expression, level of experience, nationality, personal appearance, race, religion, or sexual identity and orientation.

### Our Standards

Examples of behavior that contributes to creating a positive environment include:

- Using welcoming and inclusive language
- Being respectful of differing viewpoints and experiences
- Gracefully accepting constructive criticism
- Focusing on what is best for the community
- Showing empathy towards other community members

## 🚀 Getting Started

### Types of Contributions

We welcome several types of contributions:

- **🐛 Bug Reports**: Found something that doesn't work? Let us know!
- **✨ Feature Requests**: Have an idea for a new feature? We'd love to hear it!
- **📚 Documentation**: Improve existing docs or add new ones
- **🔧 Code Contributions**: Bug fixes, new features, performance improvements
- **🎨 Examples**: Add new examples showing how to use PythonRAG
- **🧪 Tests**: Improve test coverage or add new test cases

### Before You Start

1. **Search existing issues** to avoid duplicates
2. **Check the roadmap** in README.md to see if your idea aligns with our plans
3. **Open an issue** to discuss large changes before implementing them
4. **Read this contributing guide** thoroughly

## 💻 Development Setup

### Prerequisites

- Python 3.8+ (we test on 3.8-3.12)
- Git
- A GitHub account

### 1. Fork and Clone

```bash
# Fork the repository on GitHub, then clone your fork
git clone https://github.com/YOUR_USERNAME/PythonRAG.git
cd PythonRAG

# Add the original repository as upstream
git remote add upstream https://github.com/franzejr/PythonRAG.git
```

### 2. Set Up Development Environment

```bash
# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install development dependencies
pip install -e ".[dev,docs,all]"

# Install pre-commit hooks
pre-commit install
```

### 3. Verify Setup

```bash
# Run tests to ensure everything works
pytest

# Check code formatting
black --check src tests
ruff check src tests

# Type checking
mypy src/pythonrag

# Build documentation
mkdocs build
```

## 🔄 Making Changes

### 1. Create a Branch

```bash
# Make sure you're on main and up to date
git checkout main
git pull upstream main

# Create a new branch for your changes
git checkout -b feature/your-feature-name
# or
git checkout -b fix/issue-description
```

### 2. Make Your Changes

- **Write clear, descriptive commit messages**
- **Keep commits atomic** (one logical change per commit)
- **Add tests** for new functionality
- **Update documentation** as needed
- **Follow our code style** (enforced by pre-commit hooks)

### 3. Commit Message Guidelines

We follow conventional commits:

```
type(scope): description

[optional body]

[optional footer]
```

**Types:**
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Code style changes (formatting, etc.)
- `refactor`: Code refactoring
- `test`: Adding or updating tests
- `chore`: Maintenance tasks

**Examples:**
```bash
feat(core): add support for custom embedding models
fix(pipeline): handle empty document lists gracefully
docs(examples): add Qdrant integration tutorial
test(core): improve test coverage for RAGPipeline
```

## 🧪 Testing

We use pytest for testing with comprehensive coverage requirements.

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage report
pytest --cov=pythonrag --cov-report=html

# Run specific test file
pytest tests/test_core.py

# Run tests matching a pattern
pytest -k "test_pipeline"

# Run tests in parallel (faster)
pytest -n auto
```

### Writing Tests

- **Place tests** in the `tests/` directory
- **Mirror the source structure** (e.g., `tests/test_core.py` for `src/pythonrag/core.py`)
- **Use descriptive test names** that explain what is being tested
- **Include both positive and negative test cases**
- **Test edge cases** and error conditions

**Example test structure:**
```python
import pytest
from pythonrag import RAGPipeline
from pythonrag.exceptions import ConfigurationError


class TestRAGPipeline:
    """Test the RAGPipeline class."""
    
    def test_init_with_defaults(self):
        """Test pipeline initialization with default parameters."""
        pipeline = RAGPipeline()
        assert pipeline.chunk_size == 1000
        assert pipeline.top_k == 5
    
    def test_init_with_custom_params(self):
        """Test pipeline initialization with custom parameters."""
        pipeline = RAGPipeline(chunk_size=500, top_k=3)
        assert pipeline.chunk_size == 500
        assert pipeline.top_k == 3
    
    def test_invalid_chunk_size_raises_error(self):
        """Test that invalid chunk size raises ConfigurationError."""
        with pytest.raises(ConfigurationError):
            RAGPipeline(chunk_size=-1)
```

### Test Categories

- **Unit tests**: Test individual functions/methods in isolation
- **Integration tests**: Test component interactions
- **Example tests**: Ensure examples run without errors
- **Documentation tests**: Test code examples in documentation

## 🎨 Code Style

We enforce code style through automated tools:

### Python Code Style

- **Formatter**: Black (line length: 88)
- **Linter**: Ruff (replaces flake8, isort, etc.)
- **Type checker**: MyPy
- **Pre-commit hooks**: Automatically run on commit

### Style Guidelines

```python
# Good: Clear, descriptive names
def generate_embeddings(documents: List[str]) -> List[List[float]]:
    """Generate embeddings for a list of documents."""
    pass

# Good: Type hints
class RAGPipeline:
    def __init__(
        self,
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        chunk_size: int = 1000,
    ) -> None:
        self.embedding_model = embedding_model
        self.chunk_size = chunk_size

# Good: Docstrings
def add_documents(self, documents: List[str]) -> None:
    """
    Add documents to the RAG pipeline.
    
    Args:
        documents: List of document texts to add
        
    Raises:
        ValueError: If documents list is empty
    """
    pass
```

### Documentation Style

- **Use Google-style docstrings**
- **Include examples** in docstrings when helpful
- **Document all public methods** and classes
- **Keep line length under 88 characters**

## 📝 Submitting Changes

### 1. Push Your Changes

```bash
# Make sure tests pass
pytest

# Push to your fork
git push origin feature/your-feature-name
```

### 2. Create a Pull Request

1. **Go to GitHub** and create a pull request from your fork
2. **Use a clear title** that summarizes your changes
3. **Fill out the PR template** completely
4. **Link related issues** using "Fixes #123" or "Relates to #456"

### 3. Pull Request Template

When creating a PR, include:

```markdown
## Description
Brief description of changes made.

## Type of Change
- [ ] Bug fix (non-breaking change which fixes an issue)
- [ ] New feature (non-breaking change which adds functionality)
- [ ] Breaking change (fix or feature that would cause existing functionality to not work as expected)
- [ ] Documentation update

## Testing
- [ ] Tests pass locally
- [ ] Added tests for new functionality
- [ ] Updated documentation

## Checklist
- [ ] Code follows project style guidelines
- [ ] Self-review completed
- [ ] Documentation updated
- [ ] No new warnings introduced
```

### 4. Review Process

- **Automated checks** must pass (CI/CD pipeline)
- **Code review** by at least one maintainer
- **Address feedback** promptly and respectfully
- **Squash commits** if requested before merging

## 📚 Documentation

### Types of Documentation

1. **API Documentation**: Auto-generated from docstrings
2. **User Guides**: Step-by-step tutorials
3. **Examples**: Working code samples
4. **Architecture**: Design decisions and patterns

### Writing Documentation

```bash
# Build docs locally
mkdocs serve

# View at http://localhost:8000
```

**Documentation guidelines:**
- **Write for your audience** (beginners vs. experts)
- **Include working examples**
- **Test all code snippets**
- **Use clear, concise language**
- **Add screenshots** when helpful

### Adding New Documentation

1. **Create markdown files** in `docs/` directory
2. **Update navigation** in `mkdocs.yml`
3. **Cross-reference** related sections
4. **Test locally** before submitting

## 🚀 Release Process

*For maintainers only*

### Version Numbering

We follow [Semantic Versioning](https://semver.org/):
- **MAJOR.MINOR.PATCH** (e.g., 1.2.3)
- **Major**: Breaking changes
- **Minor**: New features (backward compatible)
- **Patch**: Bug fixes (backward compatible)

### Release Steps

1. **Update version** in `src/pythonrag/__init__.py`
2. **Update CHANGELOG.md** with new version details
3. **Create and push tag**: `git tag v1.2.3 && git push origin v1.2.3`
4. **GitHub Actions** automatically:
   - Runs full test suite
   - Builds distribution packages
   - Creates GitHub release
   - Publishes to PyPI
   - Updates documentation

## 💡 Getting Help

### Communication Channels

- **🐛 Bug Reports**: [GitHub Issues](https://github.com/franzejr/PythonRAG/issues)
- **💬 Discussions**: [GitHub Discussions](https://github.com/franzejr/PythonRAG/discussions)
- **📖 Documentation**: Available at project website

### Development Questions

1. **Check existing issues** and discussions first
2. **Search the documentation** for answers
3. **Ask specific questions** with context and examples
4. **Be patient and respectful** in communications

## 🎉 Recognition

Contributors are recognized in several ways:

- **Contributors list** in README.md
- **GitHub contributions** graph
- **Release notes** mention significant contributions
- **Special thanks** in documentation

### First-Time Contributors

We especially welcome first-time contributors! Look for issues labeled:
- `good-first-issue`: Easy issues perfect for newcomers
- `help-wanted`: Issues where we need assistance
- `documentation`: Documentation improvements

## 📞 Questions?

Don't hesitate to ask questions! We're here to help make your contribution experience positive and rewarding.

- **General questions**: [GitHub Discussions](https://github.com/franzejr/PythonRAG/discussions)
- **Specific issues**: [GitHub Issues](https://github.com/franzejr/PythonRAG/issues)

---

**Thank you for contributing to PythonRAG! 🚀**
