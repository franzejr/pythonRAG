"""
Core RAG pipeline implementation.
"""

import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

logger = logging.getLogger(__name__)


class RAGPipeline(ABC):
    """
    Main RAG pipeline abstract base class that defines the interface for document processing,
    embedding generation, vector storage, and retrieval-augmented generation.

    Args:
        embedding_model: Name or path of the embedding model to use
        llm_model: Name or path of the language model to use
        vector_db: Vector database configuration
        chunk_size: Size of text chunks for processing
        chunk_overlap: Overlap between chunks
        top_k: Number of top results to retrieve

    Example:
        >>> class MyRAGPipeline(RAGPipeline):
        ...     def add_documents(self, documents, metadata=None):
        ...         # Implementation here
        ...         pass
        ...     # ... implement other abstract methods
        >>> rag = MyRAGPipeline(
        ...     embedding_model="sentence-transformers/all-MiniLM-L6-v2",
        ...     llm_model="gpt-3.5-turbo"
        ... )
        >>> rag.add_documents(["Document content here..."])
        >>> response = rag.query("What is this document about?")
    """

    def __init__(
        self,
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        llm_model: str = "gpt-3.5-turbo",
        vector_db: Optional[Dict[str, Any]] = None,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        top_k: int = 5,
        **kwargs: Any,
    ) -> None:
        self.embedding_model = embedding_model
        self.llm_model = llm_model
        self.vector_db_config = vector_db or {"type": "in_memory"}
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.top_k = top_k

        # Initialize components (placeholders for now)
        self._embedding_client = None
        self._llm_client = None
        self._vector_db = None

        logger.info(
            f"Initialized RAGPipeline with embedding_model={embedding_model}, llm_model={llm_model}"
        )

    @abstractmethod
    def add_documents(
        self,
        documents: Union[List[str], List[Dict[str, Any]]],
        metadata: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        """
        Add documents to the RAG pipeline.

        Args:
            documents: List of document texts or document dictionaries
            metadata: Optional metadata for each document
        """
        pass

    @abstractmethod
    def query(
        self,
        question: str,
        top_k: Optional[int] = None,
        context_length: Optional[int] = None,
    ) -> str:
        """
        Query the RAG system with a question.

        Args:
            question: The question to ask
            top_k: Number of top results to retrieve (overrides default)
            context_length: Maximum context length for the LLM

        Returns:
            Generated response from the RAG system
        """
        pass

    @abstractmethod
    def add_document_file(
        self, file_path: Union[str, Path], metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Add a document from a file.

        Args:
            file_path: Path to the document file
            metadata: Optional metadata for the document
        """
        pass

    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the RAG pipeline.

        Returns:
            Dictionary with pipeline statistics
        """
        return {
            "embedding_model": self.embedding_model,
            "llm_model": self.llm_model,
            "vector_db_type": self.vector_db_config.get("type"),
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap,
            "top_k": self.top_k,
            # TODO: Add more statistics like document count, index size, etc.
        }

    @abstractmethod
    def reset(self) -> None:
        """Reset the pipeline by clearing all stored documents."""
        pass
