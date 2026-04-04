import logging
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from config import Config
from utils import get_logger

logger = get_logger(__name__)

class DocumentEmbedder:
    """
    Manages embedding generation using Google Gemini.
    """

    def __init__(self, google_api_key: str = Config.GOOGLE_API_KEY):
        """
        Args:
            google_api_key (str): Key for Google Generative AI.
        """
        try:
            logger.info("Initializing Google Gemini Embeddings.")
            self.embeddings = GoogleGenerativeAIEmbeddings(
                model=Config.EMBEDDING_MODEL,
                google_api_key=google_api_key
            )
        except Exception as e:
            logger.exception("Failed to initialize embeddings.")
            raise e

    def get_embeddings(self) -> GoogleGenerativeAIEmbeddings:
        """
        Returns the embedding model instance.

        Returns:
            GoogleGenerativeAIEmbeddings: Embedding instance compatible with LangChain.
        """
        return self.embeddings
