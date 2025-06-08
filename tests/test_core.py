"""
Unit tests for the core RAGPipeline functionality.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import pytest

from pythonrag import RAGPipeline


class MockRAGPipeline(RAGPipeline):
    """Mock implementation of RAGPipeline for testing."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.documents = []

    def add_documents(
        self,
        documents: Union[List[str], List[Dict[str, Any]]],
        metadata: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        """Mock implementation that just stores documents."""
        self.documents.extend(documents)

    def query(
        self,
        question: str,
        top_k: Optional[int] = None,
        context_length: Optional[int] = None,
    ) -> str:
        """Mock implementation that returns a simple response."""
        return f"Mock response for: {question}"

    def add_document_file(
        self, file_path: Union[str, Path], metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Mock implementation that checks if file exists."""
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        self.documents.append(f"Content from {file_path}")

    def reset(self) -> None:
        """Mock implementation that clears documents."""
        self.documents.clear()


class TestRAGPipeline:
    """Test cases for RAGPipeline abstract base class."""

    def test_cannot_instantiate_abstract_class(self):
        """Test that RAGPipeline cannot be instantiated directly."""
        with pytest.raises(TypeError, match="Can't instantiate abstract class"):
            RAGPipeline()

    def test_init_default_parameters(self):
        """Test RAGPipeline initialization with default parameters."""
        rag = MockRAGPipeline()

        assert rag.embedding_model == "sentence-transformers/all-MiniLM-L6-v2"
        assert rag.llm_model == "gpt-3.5-turbo"
        assert rag.chunk_size == 1000
        assert rag.chunk_overlap == 200
        assert rag.top_k == 5
        assert rag.vector_db_config == {"type": "in_memory"}

    def test_init_custom_parameters(self):
        """Test RAGPipeline initialization with custom parameters."""
        rag = MockRAGPipeline(
            embedding_model="custom-model",
            llm_model="custom-llm",
            chunk_size=500,
            chunk_overlap=100,
            top_k=3,
        )

        assert rag.embedding_model == "custom-model"
        assert rag.llm_model == "custom-llm"
        assert rag.chunk_size == 500
        assert rag.chunk_overlap == 100
        assert rag.top_k == 3

    def test_get_stats(self):
        """Test getting pipeline statistics."""
        rag = MockRAGPipeline(embedding_model="test-model", llm_model="test-llm")

        stats = rag.get_stats()

        assert stats["embedding_model"] == "test-model"
        assert stats["llm_model"] == "test-llm"
        assert stats["vector_db_type"] == "in_memory"
        assert stats["chunk_size"] == 1000
        assert stats["chunk_overlap"] == 200
        assert stats["top_k"] == 5

    def test_add_documents_implementation(self):
        """Test that add_documents can be implemented."""
        rag = MockRAGPipeline()

        rag.add_documents(["test document"])
        assert len(rag.documents) == 1
        assert rag.documents[0] == "test document"

    def test_query_implementation(self):
        """Test that query can be implemented."""
        rag = MockRAGPipeline()

        response = rag.query("test question")
        assert response == "Mock response for: test question"

    def test_add_document_file_implementation(self):
        """Test that add_document_file can be implemented."""
        rag = MockRAGPipeline()

        # Create a temporary file for testing
        import tempfile

        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".txt") as f:
            f.write("test content")
            temp_file_path = f.name

        try:
            rag.add_document_file(temp_file_path)
            assert len(rag.documents) == 1
            assert "test content" not in rag.documents[0]  # Mock doesn't read content
            assert temp_file_path in rag.documents[0]
        finally:
            # Clean up
            Path(temp_file_path).unlink()

    def test_add_document_file_not_found(self):
        """Test that add_document_file handles file not found."""
        rag = MockRAGPipeline()

        with pytest.raises(FileNotFoundError):
            rag.add_document_file("nonexistent_file.txt")

    def test_reset_implementation(self):
        """Test that reset can be implemented."""
        rag = MockRAGPipeline()

        rag.add_documents(["test1", "test2"])
        assert len(rag.documents) == 2

        rag.reset()
        assert len(rag.documents) == 0


class TestRAGPipelineIntegration:
    """Integration tests for RAGPipeline implementations."""

    @pytest.mark.skip(reason="Integration tests require concrete implementations")
    def test_full_pipeline_workflow(self):
        """Test a complete RAG pipeline workflow."""
        # This test will be implemented when we have concrete implementations
        pass

    @pytest.mark.skip(reason="Integration tests require concrete implementations")
    def test_pipeline_with_real_documents(self):
        """Test pipeline with actual document processing."""
        # This test will be implemented when document processing is ready
        pass
