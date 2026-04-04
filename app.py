import streamlit as st
import os
import tempfile
import logging
from typing import List, Optional
from config import Config
from utils.document_loader import DocumentLoader
from utils.chunker import DocumentChunker
from utils.embedder import DocumentEmbedder
from utils.retriever import DocumentRetriever
from utils.validator import ProjectsValidator
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_classic.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate

# Configure page
st.set_page_config(page_title="Multi-Doc RAG Research Assistant", layout="wide")
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Custom CSS for premium look
st.markdown("""
    <style>
    .main {
        background-color: #0e1117;
        color: #ffffff;
    }
    .stButton>button {
        width: 100%;
        border-radius: 5px;
        height: 3em;
        background-color: #4CAF50;
        color: white;
    }
    .stTextInput>div>div>input {
        background-color: #262730;
        color: white;
    }
    </style>
    """, unsafe_allow_html=True)

def initialize_session_state():
    """Initializes streamlit session state variables."""
    if "retriever" not in st.session_state:
        st.session_state.retriever = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "processed_files" not in st.session_state:
        st.session_state.processed_files = []

def process_uploaded_files(uploaded_files):
    """
    Processes uploaded files: Load, Chunk, Embed, Index.
    
    Args:
        uploaded_files (List): List of uploaded streamlit files.
    """
    if not Config.check_keys():
        st.error("API Key missing! Please configure .env file.")
        return

    all_chunks = []
    chunker = DocumentChunker()
    
    progress_bar = st.progress(0)
    status_text = st.empty()

    for i, uploaded_file in enumerate(uploaded_files):
        status_text.text(f"Processing {uploaded_file.name}...")
        
        # Save to temp file
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_path = tmp_file.name

        try:
            loader = DocumentLoader(tmp_path)
            docs = loader.load()
            chunks = chunker.split(docs)
            all_chunks.extend(chunks)
            st.session_state.processed_files.append(uploaded_file.name)
        except Exception as e:
            st.error(f"Error processing {uploaded_file.name}: {e}")
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
        
        progress_bar.progress((i + 1) / len(uploaded_files))

    if all_chunks:
        try:
            status_text.text("Building Vector Index (FAISS)...")
            embedder = DocumentEmbedder()
            retriever = DocumentRetriever(embedder)
            retriever.create_index(all_chunks)
            st.session_state.retriever = retriever
            st.success(f"Indexed {len(all_chunks)} chunks from {len(uploaded_files)} files.")
        except Exception as e:
            st.error(f"Indexing Error: {e}")
    
    status_text.empty()
    progress_bar.empty()

def main():
    st.title("📄 Multi-Document RAG Research Assistant")
    st.markdown("### Powered by Google Gemini & FAISS")
    
    initialize_session_state()

    # Sidebar for configuration and uploads
    with st.sidebar:
        st.header("Project Configuration")
        uploaded_files = st.file_uploader(
            "Upload Research Papers (PDF/TXT)", 
            type=["pdf", "txt"], 
            accept_multiple_files=True
        )
        
        if st.button("Process Documents"):
            if uploaded_files:
                process_uploaded_files(uploaded_files)
            else:
                st.warning("Please upload files first.")

        st.divider()
        st.subheader("Loaded Files")
        for f in st.session_state.processed_files:
            st.write(f"- {f}")

    # Main Chat Interface
    if st.session_state.retriever:
        query = st.text_input("Ask a question about your documents:", placeholder="What are the key findings?")
        
        if st.button("Search & Answer"):
            is_valid, msg = ProjectsValidator.validate_query(query)
            if not is_valid:
                st.error(msg)
            else:
                with st.spinner("Analyzing documents..."):
                    try:
                        # Setup QA Chain
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
                            retriever=st.session_state.retriever.vector_store.as_retriever(),
                            return_source_documents=True,
                            chain_type_kwargs={"prompt": PROMPT}
                        )
                        
                        result = chain({"query": query})
                        answer = result["result"]
                        source_docs = result["source_documents"]

                        st.markdown("### Answer")
                        st.write(answer)
                        
                        with st.expander("View Citations"):
                            for i, doc in enumerate(source_docs):
                                st.markdown(f"**Source {i+1}:**")
                                st.write(doc.page_content[:300] + "...")
                                st.caption(f"Metadata: {doc.metadata}")

                        st.download_button(
                            label="Download Answer",
                            data=answer,
                            file_name="research_answer.txt",
                            mime="text/plain"
                        )

                    except Exception as e:
                        st.error(f"Reasoning Error: {e}")
                        logger.exception("QA Chain failure")
    else:
        st.info("👈 Upload and process documents in the sidebar to start.")

if __name__ == "__main__":
    main()
