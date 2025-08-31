"""
Test configuration that bypasses validation for development testing
"""

import os
import logging

# Set test mode to bypass validation
os.environ["TEST_MODE"] = "true"
os.environ["GEMINI_API_KEY"] = "test_key"
os.environ["PINECONE_API_KEY"] = "test_key"
os.environ["PINECONE_ENVIRONMENT"] = "test_env"
os.environ["PINECONE_INDEX_NAME"] = "test_index"

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Suppress some verbose loggers
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)
