"""
Command-line interface for PythonRAG.
"""

import argparse
import logging
import sys

from . import __version__
from .core import RAGPipeline


def setup_logging(verbose: bool = False) -> None:
    """Set up logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )


def create_pipeline_command(args: argparse.Namespace) -> None:
    """Create and initialize a RAG pipeline."""
    print(f"Creating RAG pipeline with embedding_model={args.embedding_model}")

    try:
        rag = RAGPipeline(
            embedding_model=args.embedding_model,
            llm_model=args.llm_model,
            chunk_size=args.chunk_size,
            top_k=args.top_k,
        )

        stats = rag.get_stats()
        print("Pipeline created successfully!")
        print(f"Configuration: {stats}")

    except Exception as e:
        print(f"Error creating pipeline: {e}")
        sys.exit(1)


def add_documents_command(args: argparse.Namespace) -> None:
    """Add documents to a RAG pipeline."""
    print(f"Adding documents from: {args.documents}")

    # TODO: Implement document addition
    print("Document addition will be implemented in the next iteration")


def query_command(args: argparse.Namespace) -> None:
    """Query a RAG pipeline."""
    print(f"Querying: {args.question}")

    # TODO: Implement querying
    print("Querying will be implemented in the next iteration")


def main() -> None:
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="PythonRAG - A modern Python package for RAG workflows",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  pythonrag create --embedding-model sentence-transformers/all-MiniLM-L6-v2
  pythonrag add-docs document1.txt document2.pdf
  pythonrag query "What is the main topic of the documents?"
        """,
    )

    parser.add_argument(
        "--version", action="version", version=f"PythonRAG {__version__}"
    )

    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Enable verbose logging"
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Create pipeline command
    create_parser = subparsers.add_parser("create", help="Create a new RAG pipeline")
    create_parser.add_argument(
        "--embedding-model",
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="Embedding model to use",
    )
    create_parser.add_argument(
        "--llm-model", default="gpt-3.5-turbo", help="Language model to use"
    )
    create_parser.add_argument(
        "--chunk-size",
        type=int,
        default=1000,
        help="Chunk size for document processing",
    )
    create_parser.add_argument(
        "--top-k", type=int, default=5, help="Number of top results to retrieve"
    )
    create_parser.set_defaults(func=create_pipeline_command)

    # Add documents command
    docs_parser = subparsers.add_parser(
        "add-docs", help="Add documents to the pipeline"
    )
    docs_parser.add_argument("documents", nargs="+", help="Document files to add")
    docs_parser.set_defaults(func=add_documents_command)

    # Query command
    query_parser = subparsers.add_parser("query", help="Query the RAG system")
    query_parser.add_argument("question", help="Question to ask")
    query_parser.add_argument(
        "--top-k", type=int, help="Number of top results to retrieve"
    )
    query_parser.set_defaults(func=query_command)

    # Parse arguments
    args = parser.parse_args()

    # Set up logging
    setup_logging(args.verbose)

    # Execute command
    if hasattr(args, "func"):
        args.func(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
