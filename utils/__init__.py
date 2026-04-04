import os
import logging

# Ensure logs directory exists if logging to a file, but here we stay with console logging
# for ease of use in a Streamlit context.

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

def get_logger(name: str) -> logging.Logger:
    """
    Utility to get a logger with a specific name.
    
    Args:
        name (str): The name of the module/logger.
        
    Returns:
        logging.Logger: Configured logger instance.
    """
    return logging.getLogger(name)
