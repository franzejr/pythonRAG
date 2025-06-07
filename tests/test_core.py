"""
Unit tests for the core RAGPipeline functionality.
"""

import pytest
from unittest.mock import Mock, patch

from pythonrag import RAGPipeline
from pythonrag.exceptions import PythonRAGError, ConfigurationError


class TestRAGPipeline:
    """Test cases for RAGPipeline class."""

    def test_init_default_parameters(self):
        """Test RAGPipeline initialization with default parameters."""
        rag = RAGPipeline()

        assert rag.embedding_model == "sentence-transformers/all-MiniLM-L6-v2"
        assert rag.llm_model == "gpt-3.5-turbo"
        assert rag.chunk_size == 1000
        assert rag.chunk_overlap == 200
        assert rag.top_k == 5
        assert rag.vector_db_config == {"type": "in_memory"}

    def test_init_custom_parameters(self):
        """Test RAGPipeline initialization with custom parameters."""
        rag = RAGPipeline(
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
        rag = RAGPipeline(embedding_model="test-model", llm_model="test-llm")

        stats = rag.get_stats()

        assert stats["embedding_model"] == "test-model"
        assert stats["llm_model"] == "test-llm"
        assert stats["vector_db_type"] == "in_memory"
        assert stats["chunk_size"] == 1000
        assert stats["chunk_overlap"] == 200
        assert stats["top_k"] == 5

    def test_add_documents_not_implemented(self):
        """Test that add_documents raises NotImplementedError."""
        rag = RAGPipeline()

        with pytest.raises(NotImplementedError):
            rag.add_documents(["test document"])

    def test_query_not_implemented(self):
        """Test that query raises NotImplementedError."""
        rag = RAGPipeline()

        with pytest.raises(NotImplementedError):
            rag.query("test question")

    def test_add_document_file_not_implemented(self):
        """Test that add_document_file raises NotImplementedError."""
        rag = RAGPipeline()

        with pytest.raises(NotImplementedError):
            rag.add_document_file("test.txt")

    def test_add_document_file_not_found(self):
        """Test that add_document_file raises NotImplementedError (file check will be added later)."""
        rag = RAGPipeline()

        with pytest.raises(NotImplementedError):
            rag.add_document_file("nonexistent_file.txt")

    def test_reset_not_implemented(self):
        """Test that reset raises NotImplementedError."""
        rag = RAGPipeline()

        with pytest.raises(NotImplementedError):
            rag.reset()


class TestRAGPipelineIntegration:
    """Integration tests for RAGPipeline (when functionality is implemented)."""

    @pytest.mark.skip(reason="Functionality not yet implemented")
    def test_full_pipeline_workflow(self):
        """Test a complete RAG pipeline workflow."""
        # This test will be implemented when the core functionality is ready
        pass

    @pytest.mark.skip(reason="Functionality not yet implemented")
    def test_pipeline_with_real_documents(self):
        """Test pipeline with actual document processing."""
        # This test will be implemented when document processing is ready
        pass
