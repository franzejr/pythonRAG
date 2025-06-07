"""
PythonRAG - A modern Python package for Retrieval-Augmented Generation (RAG) workflows.

This package provides a comprehensive toolkit for building RAG systems with support
for various vector databases, embedding models, and language models.
"""

__version__ = "0.1.0"
__author__ = "PythonRAG Contributors"
__email__ = "franzejr@gmail.com"

from .core import RAGPipeline
from .exceptions import ConfigurationError, EmbeddingError, PythonRAGError

__all__ = [
    "RAGPipeline",
    "PythonRAGError",
    "ConfigurationError",
    "EmbeddingError",
]
