import logging
from typing import List, Optional
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from utils import get_logger

logger = get_logger(__name__)

class DocumentLoader:
    """
    Handles loading documents from various file types.
    """
    def __init__(self, file_path: str):
        """
        Args:
            file_path (str): The absolute path to the document file.
        """
        self.file_path = file_path

    def load(self) -> List[Document]:
        """
        Loads the document content into a list of LangChain Document objects.

        Returns:
            List[Document]: List containing document pieces.
        
        Raises:
            Exception: If loading fails due to corrupted file or unsupported type.
        """
        try:
            logger.info(f"Loading document from {self.file_path}")
            if self.file_path.endswith(".pdf"):
                loader = PyPDFLoader(self.file_path)
                return loader.load()
            elif self.file_path.endswith(".txt"):
                # Manual text load for simpler debugging/testing
                with open(self.file_path, "r", encoding="utf-8") as f:
                    content = f.read()
                return [Document(page_content=content, metadata={"source": self.file_path})]
            else:
                logger.error(f"Unsupported file type: {self.file_path}")
                raise ValueError(f"Unsupported file type: {self.file_path}")

        except Exception as e:
            logger.exception(f"Error loading document {self.file_path}: {str(e)}")
            raise e
