#!/usr/bin/env python3
"""
Interactive RAG System with Qdrant Vector Database

This example demonstrates a complete RAG system using:
- OpenAI for embeddings (text-embedding-3-small)
- OpenAI GPT-4o-mini for generation
- Qdrant for vector storage
- Interactive command-line interface

Prerequisites:
1. Install dependencies: pip install "pythonrag[all]" qdrant-client openai python-dotenv
2. Set environment variables:
   - OPENAI_API_KEY: Your OpenAI API key
   - QDRANT_API_KEY: Your Qdrant Cloud API key (optional for local Qdrant)
   - QDRANT_URL: Your Qdrant URL (defaults to local)

Usage:
    python examples/interactive_qdrant_rag.py

Features:
- Interactive question-answering
- Source attribution for answers
- Document chunking and indexing
- Error handling and graceful degradation
- Cost-optimized configuration
"""

import os
import sys
import uuid
from typing import List, Dict, Any
from pathlib import Path

# Add the src directory to Python path for development
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

try:
    import openai
    from qdrant_client import QdrantClient, models
    from dotenv import load_dotenv
except ImportError as e:
    print(f"❌ Missing required package: {e}")
    print("Install with: pip install openai qdrant-client python-dotenv")
    sys.exit(1)

# Load environment variables
load_dotenv()

# Configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")

# Models (cost-optimized)
EMBEDDING_MODEL = "text-embedding-3-small"  # 5x cheaper than large
GENERATION_MODEL = "gpt-4o-mini"  # 15x cheaper than gpt-4o

# Collection configuration
COLLECTION_NAME = "pythonrag_interactive_demo"


class InteractiveQdrantRAG:
    """
    Interactive RAG system using Qdrant vector database.

    This class provides a complete RAG implementation with:
    - Document indexing and chunking
    - Vector search with Qdrant
    - OpenAI-powered generation
    - Interactive CLI interface
    """

    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        """Initialize the RAG system."""
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        # Initialize clients
        self._init_openai_client()
        self._init_qdrant_client()
        self._setup_collection()

        print("✅ Interactive Qdrant RAG system initialized!")

    def _init_openai_client(self):
        """Initialize OpenAI client."""
        if not OPENAI_API_KEY:
            raise ValueError("❌ OPENAI_API_KEY environment variable is required!")

        try:
            self.openai_client = openai.OpenAI(api_key=OPENAI_API_KEY)
            # Test the connection
            self.openai_client.models.list()
            print("✅ OpenAI client initialized")
        except Exception as e:
            raise ValueError(f"❌ Failed to initialize OpenAI client: {e}")

    def _init_qdrant_client(self):
        """Initialize Qdrant client."""
        try:
            if QDRANT_API_KEY:
                # Cloud Qdrant
                self.qdrant_client = QdrantClient(
                    url=QDRANT_URL, api_key=QDRANT_API_KEY
                )
                print("✅ Connected to Qdrant Cloud")
            else:
                # Local Qdrant
                self.qdrant_client = QdrantClient(url=QDRANT_URL)
                print("✅ Connected to local Qdrant")

        except Exception as e:
            raise ValueError(f"❌ Failed to connect to Qdrant: {e}")

    def _setup_collection(self):
        """Setup Qdrant collection."""
        try:
            # Check if collection exists
            collections = self.qdrant_client.get_collections()
            collection_names = [col.name for col in collections.collections]

            if COLLECTION_NAME not in collection_names:
                # Create collection with proper vector configuration
                self.qdrant_client.create_collection(
                    collection_name=COLLECTION_NAME,
                    vectors_config=models.VectorParams(
                        size=1536,  # text-embedding-3-small dimension
                        distance=models.Distance.COSINE,
                    ),
                )
                print(f"✅ Created new collection: {COLLECTION_NAME}")
            else:
                print(f"✅ Using existing collection: {COLLECTION_NAME}")

        except Exception as e:
            raise ValueError(f"❌ Failed to setup collection: {e}")

    def _chunk_text(self, text: str, source: str = "unknown") -> List[Dict[str, Any]]:
        """Split text into chunks with metadata."""
        if len(text) <= self.chunk_size:
            return [
                {
                    "text": text.strip(),
                    "source": source,
                    "chunk_id": 0,
                    "total_chunks": 1,
                }
            ]

        chunks = []
        start = 0
        chunk_id = 0

        while start < len(text):
            end = start + self.chunk_size

            # Try to break at sentence boundary
            if end < len(text):
                sentence_end = text.rfind(".", start, end)
                if sentence_end > start:
                    end = sentence_end + 1
                else:
                    # Fall back to word boundary
                    word_end = text.rfind(" ", start, end)
                    if word_end > start:
                        end = word_end

            chunk_text = text[start:end].strip()
            if chunk_text:
                chunks.append(
                    {
                        "text": chunk_text,
                        "source": source,
                        "chunk_id": chunk_id,
                        "total_chunks": -1,  # Will be updated later
                    }
                )
                chunk_id += 1

            start = end - self.chunk_overlap
            if start >= len(text):
                break

        # Update total_chunks
        total_chunks = len(chunks)
        for chunk in chunks:
            chunk["total_chunks"] = total_chunks

        return chunks

    def add_documents(self, documents: List[Dict[str, str]]) -> None:
        """
        Add documents to the vector database.

        Args:
            documents: List of dicts with 'text' and 'source' keys
        """
        print(f"\n📚 Processing {len(documents)} documents...")

        all_chunks = []
        for doc in documents:
            chunks = self._chunk_text(doc["text"], doc["source"])
            all_chunks.extend(chunks)

        print(f"   Created {len(all_chunks)} chunks")

        # Generate embeddings
        print("   Generating embeddings...")
        points = []

        for i, chunk in enumerate(all_chunks):
            try:
                # Generate embedding
                response = self.openai_client.embeddings.create(
                    input=[chunk["text"]], model=EMBEDDING_MODEL
                )
                embedding = response.data[0].embedding

                # Create point for Qdrant
                point = models.PointStruct(
                    id=str(uuid.uuid4()), vector=embedding, payload=chunk
                )
                points.append(point)

                # Show progress
                if (i + 1) % 10 == 0:
                    print(
                        f"   Progress: {i + 1}/{len(all_chunks)} embeddings generated"
                    )

            except Exception as e:
                print(f"   ⚠️  Error processing chunk {i}: {e}")
                continue

        # Upload to Qdrant
        print("   Uploading to Qdrant...")
        try:
            self.qdrant_client.upsert(collection_name=COLLECTION_NAME, points=points)
            print(f"✅ Successfully indexed {len(points)} chunks")
        except Exception as e:
            print(f"❌ Failed to upload to Qdrant: {e}")

    def search(self, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        """Search for relevant context."""
        print(f"\n🔍 Searching for: '{query}'")

        try:
            # Generate query embedding
            response = self.openai_client.embeddings.create(
                input=[query], model=EMBEDDING_MODEL
            )
            query_vector = response.data[0].embedding

            # Search Qdrant
            search_results = self.qdrant_client.query_points(
                collection_name=COLLECTION_NAME, query=query_vector, limit=limit
            ).points

            # Format results
            context_items = []
            for hit in search_results:
                context_items.append(
                    {
                        "text": hit.payload["text"],
                        "source": hit.payload["source"],
                        "chunk_id": hit.payload.get("chunk_id", 0),
                        "score": hit.score,
                    }
                )

            print(f"   Found {len(context_items)} relevant chunks")
            return context_items

        except Exception as e:
            print(f"❌ Search error: {e}")
            return []

    def generate_answer(self, query: str, context_items: List[Dict[str, Any]]) -> str:
        """Generate answer using retrieved context."""
        print("   🤖 Generating answer...")

        if not context_items:
            return "I couldn't find any relevant information to answer your question."

        # Format context
        context_parts = []
        for i, item in enumerate(context_items, 1):
            source = item["source"]
            text = item["text"]
            context_parts.append(f"[Source {i}: {source}]\n{text}")

        context_str = "\n\n".join(context_parts)

        # Create prompt
        prompt = f"""Based on the following context, please provide a comprehensive and accurate answer to the user's question. 

If the context doesn't contain enough information to fully answer the question, say so clearly.

Context:
{context_str}

Question: {query}

Please provide a well-structured answer and mention which sources you're referencing."""

        try:
            response = self.openai_client.chat.completions.create(
                model=GENERATION_MODEL,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a helpful assistant that answers questions based on provided context. Always cite your sources and be honest about limitations in the available information.",
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=0.2,  # Lower temperature for more factual responses
                max_tokens=1000,
            )

            answer = response.choices[0].message.content
            print("   ✅ Answer generated")
            return answer

        except Exception as e:
            print(f"❌ Generation error: {e}")
            return f"I apologize, but I encountered an error while generating the answer: {e}"

    def query(self, question: str, top_k: int = 5) -> Dict[str, Any]:
        """Complete RAG query pipeline."""
        # Search for context
        context_items = self.search(question, limit=top_k)

        # Generate answer
        answer = self.generate_answer(question, context_items)

        return {
            "question": question,
            "answer": answer,
            "context": context_items,
            "source_count": len(set(item["source"] for item in context_items)),
        }

    def get_collection_stats(self) -> Dict[str, Any]:
        """Get statistics about the collection."""
        try:
            collection_info = self.qdrant_client.get_collection(COLLECTION_NAME)
            return {
                "collection_name": COLLECTION_NAME,
                "points_count": collection_info.points_count,
                "vector_size": collection_info.config.params.vectors.size,
                "distance_metric": collection_info.config.params.vectors.distance,
            }
        except Exception as e:
            return {"error": str(e)}


def load_sample_documents() -> List[Dict[str, str]]:
    """Load sample documents for demonstration."""
    return [
        {
            "text": """
            Artificial Intelligence (AI) is a broad field of computer science focused on building smart machines capable of performing tasks that typically require human intelligence. These tasks include learning, reasoning, problem-solving, perception, and language understanding.

            AI can be categorized into two main types: Narrow AI and General AI. Narrow AI, also called Weak AI, is designed to perform specific tasks like playing chess, recognizing speech, or driving cars. This is the type of AI we see today in applications like Siri, Alexa, and recommendation systems.

            General AI, also called Strong AI or Artificial General Intelligence (AGI), would have the ability to understand, learn, and apply knowledge across a wide range of tasks at a level equal to or beyond human capability. This type of AI doesn't exist yet but is the subject of ongoing research.

            Machine Learning is a subset of AI that enables computers to learn and improve from experience without being explicitly programmed. It uses algorithms and statistical models to analyze and draw inferences from patterns in data.
            """,
            "source": "AI_Overview.md",
        },
        {
            "text": """
            Large Language Models (LLMs) like GPT-4 are AI systems trained on vast amounts of text data. They can understand and generate human-like text, making them useful for tasks like writing, translation, summarization, and question-answering.

            LLMs work by predicting the next word in a sequence based on the context of previous words. They use a type of neural network architecture called a Transformer, which was introduced in the paper "Attention Is All You Need" by Vaswani et al. in 2017.

            The training process involves two main phases:
            1. Pre-training: The model learns language patterns from large text datasets
            2. Fine-tuning: The model is adapted for specific tasks or to follow instructions

            Popular LLMs include GPT-4 by OpenAI, Claude by Anthropic, and Llama by Meta. These models have shown remarkable capabilities in natural language understanding and generation, but they also have limitations including potential biases, factual errors, and the tendency to generate plausible-sounding but incorrect information.
            """,
            "source": "LLM_Guide.md",
        },
        {
            "text": """
            Retrieval-Augmented Generation (RAG) combines the strengths of retrieval-based and generation-based approaches. It first retrieves relevant information from a knowledge base, then uses this information to generate more accurate and contextually relevant responses.

            The RAG process typically involves these steps:
            1. Document Indexing: Break documents into chunks and create vector embeddings
            2. Query Processing: Convert user queries into vector embeddings
            3. Retrieval: Find the most similar document chunks using vector similarity search
            4. Generation: Use a language model to generate responses based on retrieved context

            Benefits of RAG include:
            - More accurate and factual responses
            - Ability to cite sources
            - Can work with domain-specific knowledge
            - Reduces hallucination in language models
            - Allows for real-time knowledge updates

            RAG is particularly useful for question-answering systems, customer support, and knowledge management applications. Popular vector databases for RAG include Pinecone, Weaviate, Chroma, and Qdrant.
            """,
            "source": "RAG_Explained.md",
        },
        {
            "text": """
            Vector databases are specialized databases designed to store and query high-dimensional vectors efficiently. They are essential for modern AI applications, particularly those involving semantic search, recommendation systems, and retrieval-augmented generation.

            Key features of vector databases include:
            - Efficient similarity search using algorithms like HNSW (Hierarchical Navigable Small World)
            - Support for various distance metrics (cosine, euclidean, dot product)
            - Horizontal scaling capabilities
            - Integration with machine learning workflows
            - Real-time indexing and updates

            Popular vector databases include:

            Qdrant: Open-source vector database written in Rust, known for its performance and easy deployment. Supports filtering, payload storage, and distributed deployment.

            Pinecone: Cloud-native vector database with managed infrastructure, automatic scaling, and high availability.

            Weaviate: Open-source vector database with GraphQL API, automatic vectorization, and hybrid search capabilities.

            Chroma: Lightweight vector database focused on simplicity and developer experience, perfect for prototyping and small-scale applications.

            Each has its own strengths depending on the use case, scale requirements, and deployment preferences.
            """,
            "source": "Vector_Databases.md",
        },
    ]


def main():
    """Main interactive loop."""
    print("🚀 PythonRAG Interactive Demo with Qdrant")
    print("=" * 50)

    # Check prerequisites
    if not OPENAI_API_KEY:
        print("❌ Please set your OPENAI_API_KEY environment variable!")
        print("   Get your API key from: https://platform.openai.com/api-keys")
        return

    try:
        # Initialize RAG system
        rag = InteractiveQdrantRAG(chunk_size=800, chunk_overlap=100)

        # Load sample documents
        sample_docs = load_sample_documents()
        rag.add_documents(sample_docs)

        # Show collection stats
        stats = rag.get_collection_stats()
        print(f"\n📊 Collection Stats:")
        for key, value in stats.items():
            print(f"   {key}: {value}")

        # Interactive loop
        print("\n" + "=" * 50)
        print("🎯 Ready to answer questions about AI, LLMs, RAG, and Vector Databases!")
        print("   Type 'help' for available commands")
        print("   Type 'exit' to quit")
        print("=" * 50)

        while True:
            try:
                user_input = input("\n❓ Your question: ").strip()

                if user_input.lower() in ["exit", "quit", "q"]:
                    print("👋 Goodbye!")
                    break

                elif user_input.lower() == "help":
                    print("\n📋 Available commands:")
                    print("   • Ask any question about the indexed documents")
                    print("   • 'stats' - Show collection statistics")
                    print("   • 'help' - Show this help message")
                    print("   • 'exit' - Quit the program")
                    continue

                elif user_input.lower() == "stats":
                    stats = rag.get_collection_stats()
                    print("\n📊 Current Collection Stats:")
                    for key, value in stats.items():
                        print(f"   {key}: {value}")
                    continue

                elif not user_input:
                    print("Please enter a question or command.")
                    continue

                # Process the query
                result = rag.query(user_input, top_k=3)

                # Display results
                print(f"\n💡 Answer:")
                print("-" * 40)
                print(result["answer"])

                print(f"\n📚 Sources (from {result['source_count']} documents):")
                print("-" * 40)
                for i, context in enumerate(result["context"], 1):
                    source = context["source"]
                    score = context.get("score", 0)
                    preview = context["text"][:150].replace("\n", " ") + "..."
                    print(f"{i}. [{source}] (similarity: {score:.3f})")
                    print(f"   {preview}")

            except KeyboardInterrupt:
                print("\n\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"\n❌ An error occurred: {e}")
                print("Please try again or type 'exit' to quit.")

    except Exception as e:
        print(f"❌ Failed to initialize RAG system: {e}")
        print("\nTroubleshooting:")
        print("1. Check your OPENAI_API_KEY environment variable")
        print("2. Ensure Qdrant is running (if using local instance)")
        print("3. Check your internet connection")


if __name__ == "__main__":
    main()
