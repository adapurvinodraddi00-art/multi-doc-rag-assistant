import logging
import re
from typing import Tuple, List
from utils import get_logger

logger = get_logger(__name__)

class ProjectsValidator:
    """
    Logic for validating inputs and outputs to ensure safety and correctness.
    """

    @staticmethod
    def validate_query(query: str) -> Tuple[bool, str]:
        """
        Ensures the query is not empty and meets basic safety criteria.

        Args:
            query (str): User query.

        Returns:
            Tuple[bool, str]: (is_valid, error_message)
        """
        if not query or len(query.strip()) < 3:
            return False, "Query must be at least 3 characters long."
        
        # Check for basic malicious patterns (example)
        if re.search(r"<script>|DROP TABLE|DELETE FROM", query, re.I):
            logger.warning(f"Potentially malicious query blocked: {query}")
            return False, "Suspicious activity detected in query."

        return True, ""

    @staticmethod
    def validate_file_type(filename: str) -> bool:
        """
        Validates allowed file extensions.

        Args:
            filename (str): Name of the uploaded file.

        Returns:
            bool: True if allowed.
        """
        allowed = [".pdf", ".txt"]
        return any(filename.lower().endswith(ext) for ext in allowed)
