import logging
from typing import List
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from config import Config
from utils import get_logger

logger = get_logger(__name__)

class DocumentChunker:
    """
    Splits long documents into smaller chunks for embedding and retrieval.
    """

    def __init__(self, chunk_size: int = Config.CHUNK_SIZE, overlap: int = Config.CHUNK_OVERLAP):
        """
        Args:
            chunk_size (int): Max size of each chunk.
            overlap (int): Overlap between adjacent chunks.
        """
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )

    def split(self, documents: List[Document]) -> List[Document]:
        """
        Splits a list of documents into chunks.

        Args:
            documents (List[Document]): Original documents.

        Returns:
            List[Document]: Chunked documents.
        """
        try:
            logger.info(f"Splitting {len(documents)} documents into chunks.")
            chunks = self.splitter.split_documents(documents)
            logger.info(f"Generated {len(chunks)} chunks.")
            return chunks
        except Exception as e:
            logger.exception("Error during document splitting.")
            raise e
