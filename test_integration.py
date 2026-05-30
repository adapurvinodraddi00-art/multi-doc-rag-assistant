import unittest
import os
import logging
from config import Config
from utils.document_loader import DocumentLoader
from utils.chunker import DocumentChunker
from utils.embedder import DocumentEmbedder
from utils.retriever import DocumentRetriever
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_classic.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TestRAGIntegration(unittest.TestCase):
    """
    Integration test suite verifying the end-to-end RAG system pipeline
    with actual files, embedding, FAISS indexing, and LLM reasoning.
    """

    @classmethod
    def setUpClass(cls):
        """Verify API key and define file paths."""
        cls.api_key_valid = Config.check_keys()
        if not cls.api_key_valid:
            raise unittest.SkipTest("Skipping integration tests because GOOGLE_API_KEY is not configured or invalid.")

        cls.sample_dir = "samples"
        cls.sample_files = [
            os.path.join(cls.sample_dir, "gemini_overview.txt"),
            os.path.join(cls.sample_dir, "rag_explanation.txt"),
            os.path.join(cls.sample_dir, "project_summary.txt"),
        ]

        # Verify that sample files exist
        for file_path in cls.sample_files:
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"Required sample file not found: {file_path}")

    def test_end_to_end_rag_pipeline(self):
        """
        Verify that all sample documents can be loaded, chunked, embedded, indexed,
        and queried using the configured Gemini model.
        """
        logger.info("Step 1: Loading all sample documents.")
        all_docs = []
        for file_path in self.sample_files:
            loader = DocumentLoader(file_path)
            docs = loader.load()
            self.assertGreater(len(docs), 0, f"Failed to load content from {file_path}")
            all_docs.extend(docs)

        logger.info(f"Loaded {len(all_docs)} raw documents.")

        logger.info("Step 2: Chunking documents.")
        chunker = DocumentChunker(chunk_size=Config.CHUNK_SIZE, overlap=Config.CHUNK_OVERLAP)
        chunks = chunker.split(all_docs)
        self.assertGreater(len(chunks), 0, "No chunks were generated.")
        logger.info(f"Generated {len(chunks)} chunks.")

        logger.info("Step 3: Creating Embeddings and Indexing with FAISS.")
        embedder = DocumentEmbedder()
        retriever = DocumentRetriever(embedder)
        retriever.create_index(chunks)
        self.assertIsNotNone(retriever.vector_store, "FAISS vector store was not initialized.")

        # Test vector retrieval directly first
        test_query = "Google Gemini model sizes"
        retrieved_docs = retriever.retrieve(test_query, k=2)
        self.assertGreater(len(retrieved_docs), 0, "Retrieval returned no documents.")
        logger.info(f"Successfully retrieved {len(retrieved_docs)} documents for query: '{test_query}'")

        logger.info("Step 4: Setting up QA Chain and querying the LLM.")
        llm = ChatGoogleGenerativeAI(
            model=Config.LLM_MODEL,
            google_api_key=Config.GOOGLE_API_KEY,
            temperature=0.3
        )

        prompt_template = """Use the following pieces of context to answer the question at the end. 
        If you don't know the answer, just say that you don't know, don't try to make up an answer.
        Always provide a concise but comprehensive answer.

        {context}

        Question: {question}
        Helpful Answer:"""
        
        PROMPT = PromptTemplate(
            template=prompt_template, input_variables=["context", "question"]
        )

        chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever.vector_store.as_retriever(search_kwargs={"k": 3}),
            return_source_documents=True,
            chain_type_kwargs={"prompt": PROMPT}
        )

        # Query 1: Gemini sizes
        q1 = "What are the different sizes of Google Gemini models?"
        logger.info(f"Query 1: {q1}")
        res1 = chain({"query": q1})
        ans1 = res1["result"]
        logger.info(f"Answer 1: {ans1}")
        self.assertTrue(any(size in ans1.lower() for size in ["nano", "flash", "pro", "ultra"]),
                        f"Expected model size keyword in answer, got: {ans1}")

        # Query 2: What RAG stands for
        q2 = "What does RAG stand for and why is it used?"
        logger.info(f"Query 2: {q2}")
        res2 = chain({"query": q2})
        ans2 = res2["result"]
        logger.info(f"Answer 2: {ans2}")
        self.assertIn("retrieval-augmented generation", ans2.lower())

        # Query 3: Project technologies
        q3 = "What technologies are used in Project_DataScience to build the assistant?"
        logger.info(f"Query 3: {q3}")
        res3 = chain({"query": q3})
        ans3 = res3["result"]
        logger.info(f"Answer 3: {ans3}")
        self.assertTrue(any(tech in ans3.lower() for tech in ["streamlit", "gemini", "faiss"]),
                        f"Expected technology keyword in answer, got: {ans3}")

        logger.info("Integration test completed successfully with all assertions passing!")

if __name__ == "__main__":
    unittest.main()
