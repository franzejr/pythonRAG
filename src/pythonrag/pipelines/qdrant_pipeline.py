"""
Qdrant RAG Pipeline Implementation

This module contains a concrete implementation of the RAGPipeline abstract base class
using Qdrant as the vector database backend.

Requirements:
    pip install qdrant-client openai sentence-transformers

Environment Variables:
    OPENAI_API_KEY: Your OpenAI API key
    QDRANT_URL: Qdrant server URL (optional, defaults to local)
    QDRANT_API_KEY: Qdrant API key (for cloud instances)
"""

import logging
import os
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from ..core import RAGPipeline
from ..exceptions import (
    ConfigurationError,
    DocumentProcessingError,
    EmbeddingError,
    VectorDatabaseError,
)

try:
    from qdrant_client import QdrantClient
    from qdrant_client.models import Distance, PointStruct, VectorParams
except ImportError as e:
    raise ImportError(
        "Qdrant client not installed. Please install with: pip install qdrant-client"
    ) from e

try:
    import openai
except ImportError as e:
    raise ImportError(
        "OpenAI not installed. Please install with: pip install openai"
    ) from e

try:
    from sentence_transformers import SentenceTransformer
except ImportError as e:
    raise ImportError(
        "SentenceTransformers not installed. Please install with: pip install sentence-transformers"
    ) from e


# Setup logging
logger = logging.getLogger(__name__)


class QdrantPipeline(RAGPipeline):
    """
    Extended RAG Pipeline with Qdrant vector database implementation.

    This class extends the base RAGPipeline to provide full functionality
    with Qdrant as the vector database backend.
    """

    def __init__(
        self,
        embedding_model: str = "all-MiniLM-L6-v2",
        llm_model: str = "gpt-4o-mini",
        qdrant_url: Optional[str] = None,
        qdrant_api_key: Optional[str] = None,
        collection_name: str = "documents",
        vector_size: int = 384,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        top_k: int = 5,
        **kwargs,
    ):
        """
        Initialize the Qdrant RAG Pipeline.

        Args:
            embedding_model: SentenceTransformers model name
            llm_model: OpenAI model name
            qdrant_url: Qdrant server URL
            qdrant_api_key: Qdrant API key
            collection_name: Name of the Qdrant collection
            vector_size: Size of the embedding vectors
            chunk_size: Size of text chunks
            chunk_overlap: Overlap between chunks
            top_k: Number of results to retrieve
        """
        # Initialize base class
        super().__init__(
            embedding_model=embedding_model,
            llm_model=llm_model,
            vector_db={"type": "qdrant", "url": qdrant_url, "api_key": qdrant_api_key},
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            top_k=top_k,
            **kwargs,
        )

        self.collection_name = collection_name
        self.vector_size = vector_size
        self.documents_added = 0

        # Initialize clients
        self._init_qdrant_client(qdrant_url, qdrant_api_key)
        self._init_embedding_model()
        self._init_openai_client()

        # Create collection if it doesn't exist
        self._ensure_collection_exists()

    def _init_qdrant_client(self, url: Optional[str], api_key: Optional[str]):
        """Initialize Qdrant client."""
        try:
            if url:
                self._vector_db = QdrantClient(url=url, api_key=api_key)
                logger.info(f"Connected to Qdrant at {url}")
            else:
                # Try local connection
                self._vector_db = QdrantClient(host="localhost", port=6333)
                logger.info("Connected to local Qdrant instance")

        except Exception as e:
            raise VectorDatabaseError(f"Failed to connect to Qdrant: {e}") from e

    def _init_embedding_model(self):
        """Initialize embedding model."""
        try:
            self._embedding_client = SentenceTransformer(self.embedding_model)
            logger.info(f"Loaded embedding model: {self.embedding_model}")
        except Exception as e:
            raise EmbeddingError(f"Failed to load embedding model: {e}") from e

    def _init_openai_client(self):
        """Initialize OpenAI client."""
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ConfigurationError("OPENAI_API_KEY environment variable is required")

        try:
            self._llm_client = openai.OpenAI(api_key=api_key)
            logger.info(f"Initialized OpenAI client with model: {self.llm_model}")
        except Exception as e:
            raise ConfigurationError(f"Failed to initialize OpenAI client: {e}") from e

    def _ensure_collection_exists(self):
        """Create Qdrant collection if it doesn't exist."""
        try:
            collections = self._vector_db.get_collections().collections
            collection_names = [c.name for c in collections]

            if self.collection_name not in collection_names:
                self._vector_db.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(
                        size=self.vector_size, distance=Distance.COSINE
                    ),
                )
                logger.info(f"Created Qdrant collection: {self.collection_name}")
            else:
                logger.info(f"Using existing Qdrant collection: {self.collection_name}")

        except Exception as e:
            raise VectorDatabaseError(f"Failed to create/access collection: {e}") from e

    def _chunk_text(self, text: str) -> List[str]:
        """Split text into chunks with overlap."""
        if len(text) <= self.chunk_size:
            return [text]

        chunks = []
        start = 0

        while start < len(text):
            end = start + self.chunk_size
            chunk = text[start:end]

            # Try to break at sentence boundaries
            if end < len(text):
                last_period = chunk.rfind(".")
                last_newline = chunk.rfind("\n")
                break_point = max(last_period, last_newline)

                if break_point > start + self.chunk_size // 2:
                    chunk = text[start : start + break_point + 1]
                    end = start + break_point + 1

            chunks.append(chunk.strip())
            start = end - self.chunk_overlap

        return [chunk for chunk in chunks if chunk.strip()]

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
        logger.info(f"Adding {len(documents)} documents to the pipeline")

        if metadata and len(metadata) != len(documents):
            raise ValueError("Metadata list must have same length as documents list")

        points = []

        for i, doc in enumerate(documents):
            # Handle both string and dict documents
            if isinstance(doc, dict):
                text = doc.get("content", doc.get("text", ""))
                doc_metadata = doc.get("metadata", {})
            else:
                text = doc
                doc_metadata = {}

            # Add provided metadata
            if metadata and i < len(metadata):
                doc_metadata.update(metadata[i])

            # Chunk the document
            chunks = self._chunk_text(text)

            for j, chunk in enumerate(chunks):
                try:
                    # Generate embedding
                    embedding = self._embedding_client.encode(chunk).tolist()

                    # Create point
                    point_id = str(uuid.uuid4())
                    point = PointStruct(
                        id=point_id,
                        vector=embedding,
                        payload={
                            "text": chunk,
                            "document_id": i,
                            "chunk_id": j,
                            "timestamp": datetime.now().isoformat(),
                            **doc_metadata,
                        },
                    )
                    points.append(point)

                except Exception as e:
                    logger.error(f"Failed to process chunk {j} of document {i}: {e}")
                    continue

        # Batch insert points
        if points:
            try:
                self._vector_db.upsert(
                    collection_name=self.collection_name, points=points
                )
                self.documents_added += len(documents)
                logger.info(
                    f"Successfully added {len(points)} chunks from {len(documents)} documents"
                )
            except Exception as e:
                raise VectorDatabaseError(f"Failed to insert documents: {e}") from e

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
            top_k: Number of top results to retrieve
            context_length: Maximum context length (not used in this implementation)

        Returns:
            Generated response from the RAG system
        """
        logger.info(f"Processing query: {question[:100]}...")

        k = top_k or self.top_k

        try:
            # Generate embedding for the question
            question_embedding = self._embedding_client.encode(question).tolist()

            # Search for similar documents
            search_results = self._vector_db.search(
                collection_name=self.collection_name,
                query_vector=question_embedding,
                limit=k,
            )

            if not search_results:
                return (
                    "I couldn't find any relevant information to answer your question."
                )

            # Extract context from search results
            contexts = []
            for result in search_results:
                context = f"[Score: {result.score:.3f}] {result.payload['text']}"
                contexts.append(context)

            context_text = "\n\n".join(contexts)

            # Generate response using OpenAI
            system_prompt = """You are a helpful assistant that answers questions based on the provided context.
Use only the information from the context to answer the question. If the context doesn't contain
enough information to answer the question, say so clearly."""

            user_prompt = f"""Context:
{context_text}

Question: {question}

Please provide a comprehensive answer based on the context above."""

            response = self._llm_client.chat.completions.create(
                model=self.llm_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.1,
                max_tokens=500,
            )

            answer = response.choices[0].message.content
            logger.info("Generated response successfully")
            return answer

        except Exception as e:
            logger.error(f"Query processing failed: {e}")
            raise EmbeddingError(f"Failed to process query: {e}") from e

    def add_document_file(
        self, file_path: Union[str, Path], metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Add a document from a file.

        Args:
            file_path: Path to the document file
            metadata: Optional metadata for the document
        """
        file_path = Path(file_path)

        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        logger.info(f"Adding document from file: {file_path}")

        try:
            # Read file content (supporting common text formats)
            if file_path.suffix.lower() == ".txt":
                content = file_path.read_text(encoding="utf-8")
            else:
                raise ValueError(f"Unsupported file type: {file_path.suffix}")

            # Add file metadata
            file_metadata = {
                "filename": file_path.name,
                "filepath": str(file_path),
                "file_size": file_path.stat().st_size,
                **(metadata or {}),
            }

            self.add_documents([content], [file_metadata])

        except Exception as e:
            raise DocumentProcessingError(
                f"Failed to process file {file_path}: {e}"
            ) from e

    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the RAG pipeline."""
        try:
            collection_info = self._vector_db.get_collection(self.collection_name)
            point_count = collection_info.points_count
        except Exception:
            point_count = 0

        base_stats = super().get_stats()
        base_stats.update(
            {
                "collection_name": self.collection_name,
                "vector_size": self.vector_size,
                "documents_added": self.documents_added,
                "total_chunks": point_count,
                "qdrant_url": self.vector_db_config.get("url", "localhost:6333"),
            }
        )
        return base_stats

    def reset(self) -> None:
        """Reset the pipeline by clearing all stored documents."""
        logger.info("Resetting RAG pipeline")

        try:
            # Delete and recreate collection
            self._vector_db.delete_collection(self.collection_name)
            self._ensure_collection_exists()
            self.documents_added = 0
            logger.info("Pipeline reset successfully")
        except Exception as e:
            raise VectorDatabaseError(f"Failed to reset pipeline: {e}") from e
