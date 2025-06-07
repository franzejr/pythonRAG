"""
Custom exceptions for PythonRAG package.
"""


class PythonRAGError(Exception):
    """Base exception for all PythonRAG-related errors."""

    pass


class ConfigurationError(PythonRAGError):
    """Raised when there's an issue with configuration."""

    pass


class EmbeddingError(PythonRAGError):
    """Raised when there's an issue with embedding generation."""

    pass


class VectorDatabaseError(PythonRAGError):
    """Raised when there's an issue with vector database operations."""

    pass


class DocumentProcessingError(PythonRAGError):
    """Raised when there's an issue with document processing."""

    pass


class LLMError(PythonRAGError):
    """Raised when there's an issue with language model operations."""

    pass
