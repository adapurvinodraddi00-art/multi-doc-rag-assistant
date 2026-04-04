import logging
import os
from typing import List
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from config import Config
from utils import get_logger
from utils.embedder import DocumentEmbedder

logger = get_logger(__name__)

class DocumentRetriever:
    """
    Manages vector storage and similarity search using FAISS.
    """

    def __init__(self, embedder: DocumentEmbedder):
        """
        Args:
            embedder (DocumentEmbedder): Instance of the embedder.
        """
        self.embedder = embedder
        self.vector_store = None

    def create_index(self, chunks: List[Document]) -> None:
        """
        Creates a new FAISS index from document chunks.

        Args:
            chunks (List[Document]): Processed text chunks.
        """
        try:
            logger.info(f"Creating FAISS index with {len(chunks)} chunks.")
            self.vector_store = FAISS.from_documents(chunks, self.embedder.get_embeddings())
            # Optionally save locally
            # self.vector_store.save_local(Config.VECTOR_DB_PATH)
        except Exception as e:
            logger.exception("Error creating FAISS index.")
            raise e

    def retrieve(self, query: str, k: int = 4) -> List[Document]:
        """
        Performs similarity search to find relevant document pieces.

        Args:
            query (str): User natural language query.
            k (int): Number of chunks to retrieve.

        Returns:
            List[Document]: Relevant chunks.
        """
        try:
            if not self.vector_store:
                logger.warning("Attempted retrieval without an active index.")
                return []
            
            logger.info(f"Retrieving top {k} documents for query: {query[:50]}...")
            return self.vector_store.similarity_search(query, k=k)
        except Exception as e:
            logger.exception(f"Error during retrieval for query: {query}")
            raise e
