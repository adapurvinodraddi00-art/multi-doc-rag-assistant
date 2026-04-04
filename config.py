import os
import logging
from dotenv import load_dotenv
from pydantic import ValidationError

# Load environment variables from .env file
load_dotenv()

class Config:
    """
    Configuration class to manage environment variables and constants.
    
    Attributes:
        GOOGLE_API_KEY (str): API key for Google Generative AI.
        EMBEDDING_MODEL (str): Model name for generating embeddings.
        LLM_MODEL (str): Model name for the LLM.
        CHUNK_SIZE (int): Size of text chunks for processing.
        CHUNK_OVERLAP (int): Overlap size between chunks.
    """
    GOOGLE_API_KEY: str = os.getenv("GOOGLE_API_KEY", "")
    EMBEDDING_MODEL: str = "models/gemini-embedding-001"
    LLM_MODEL: str = "models/gemini-3-flash-preview"
    CHUNK_SIZE: int = 1000
    CHUNK_OVERLAP: int = 200
    VECTOR_DB_PATH: str = "faiss_index"

    @classmethod
    def check_keys(cls) -> bool:
        """
        Validates that all required API keys are present.

        Returns:
            bool: True if keys are valid, False otherwise.
        
        Example:
            >>> Config.check_keys()
            True
        """
        if not cls.GOOGLE_API_KEY or cls.GOOGLE_API_KEY == "your_google_api_key_here":
            logging.error("GOOGLE_API_KEY is missing or invalid.")
            return False
        return True

# Initialize logging configuration
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
