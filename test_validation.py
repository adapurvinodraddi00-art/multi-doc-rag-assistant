import unittest
import os
import logging
from config import Config
from utils.document_loader import DocumentLoader
from utils.chunker import DocumentChunker
from utils.embedder import DocumentEmbedder
from utils.retriever import DocumentRetriever
from utils.validator import ProjectsValidator
from langchain_core.documents import Document

# Set up logging for tests
logging.basicConfig(level=logging.INFO)

class TestRAGSystem(unittest.TestCase):
    """
    Comprehensive test suite for the Multi-Document RAG Research Assistant.
    """

    def setUp(self):
        """Setup temporary test data."""
        self.test_file_path = "test_doc.txt"
        with open(self.test_file_path, "w", encoding="utf-8") as f:
            f.write("Google Gemini is a powerful AI model. It supports RAG systems.")

    def tearDown(self):
        """Clean up temporary test data."""
        if os.path.exists(self.test_file_path):
            os.remove(self.test_file_path)

    def test_imports(self):
        """Test 1: Check if all core modules can be imported."""
        try:
            from config import Config
            from utils.document_loader import DocumentLoader
            self.assertTrue(True)
        except ImportError as e:
            self.fail(f"Import failed: {e}")

    def test_config(self):
        """Test 2: Check if config validation works (even if FAIL, it should return bool)."""
        is_valid = Config.check_keys()
        self.assertIsInstance(is_valid, bool)

    def test_chunking(self):
        """Test 3: Verify document chunking logic."""
        docs = [Document(page_content="A " * 2000, metadata={"source": "test"})]
        chunker = DocumentChunker(chunk_size=500, overlap=50)
        chunks = chunker.split(docs)
        self.assertGreater(len(chunks), 1)
        self.assertLessEqual(len(chunks[0].page_content), 1000) # Buffer allowed

    def test_validation_logic(self):
        """Test 4: Check if validator catches bad queries and file types."""
        v = ProjectsValidator()
        self.assertFalse(v.validate_query(" ")[0])
        self.assertFalse(v.validate_query("hi")[0])
        self.assertTrue(v.validate_file_type("paper.pdf"))
        self.assertFalse(v.validate_file_type("image.png"))

    def test_end_to_end_loading(self):
        """Test 5: Load a document and verify content exists."""
        loader = DocumentLoader(self.test_file_path)
        docs = loader.load()
        self.assertEqual(len(docs), 1)
        self.assertIn("Google Gemini", docs[0].page_content)

if __name__ == "__main__":
    unittest.main()
