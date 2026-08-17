"""Document processing module for loading and splitting PDF documents."""

from typing import List, Union
from pathlib import Path

from langchain_community.document_loaders import (
    PyPDFLoader,
    PyPDFDirectoryLoader,
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema import Document


class DocumentProcessor:
    """Handles PDF document loading and processing."""

    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50):
        """
        Initialize document processor.

        Args:
            chunk_size: Size of text chunks.
            chunk_overlap: Overlap between chunks.
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

    def load_from_pdf(self, file_path: Union[str, Path]) -> List[Document]:
        """Load a single PDF file."""
        loader = PyPDFLoader(str(file_path))
        return loader.load()

    def load_from_pdf_dir(self, directory: Union[str, Path]) -> List[Document]:
        """Load all PDF files from a directory."""
        loader = PyPDFDirectoryLoader(str(directory))
        return loader.load()

    def load_documents(
        self,
        source: Union[str, Path],
    ) -> List[Document]:
        """
        Load PDFs from a file or directory.

        Args:
            source: Path to a PDF file or directory containing PDFs.

        Returns:
            List of loaded documents.
        """
        path = Path(source)

        if path.is_dir():
            return self.load_from_pdf_dir(path)

        if path.is_file() and path.suffix.lower() == ".pdf":
            return self.load_from_pdf(path)

        raise ValueError(
            f"Invalid PDF source: {source}. "
            "Provide a PDF file or directory containing PDFs."
        )

    def split_documents(
        self,
        documents: List[Document],
    ) -> List[Document]:
        """Split documents into chunks."""
        return self.splitter.split_documents(documents)

    def process_pdfs(
        self,
        source: Union[str, Path],
    ) -> List[Document]:
        """
        Load and split PDF documents.

        Args:
            source: PDF file or directory containing PDFs.

        Returns:
            List of processed document chunks.
        """
        docs = self.load_documents(source)
        return self.split_documents(docs)