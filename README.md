# Multi-Document RAG Research Assistant

A complete, production-ready college final year project implementing a Retrieval-Augmented Generation (RAG) system for research paper analysis.

## 🚀 Features
- **Multi-Document Support**: Upload multiple PDF or TXT files.
- **RAG Architecture**: Uses LangChain, FAISS (Vector DB), and Google Gemini.
- **Source Citations**: Displays exact document snippets used for the answer.
- **Clean UI**: Built with Streamlit for a premium user experience.
- **Validation**: Strict input and file type validation.

## 🛠️ Project Structure
```text
.
├── app.py                 # Streamlit UI & RAG Integration
├── config.py              # Configuration & API Key Validation
├── requirements.txt       # Project Dependencies
├── test_validation.py     # 5-Step Verification Suite
├── utils/
│   ├── document_loader.py # PDF/TXT Parser
│   ├── chunker.py         # Recursive Text Splitter
│   ├── embedder.py        # Gemini Embedding Logic
│   ├── retriever.py       # FAISS indexing & Retrieval
│   └── validator.py       # Security & Input Checks
└── .env.example           # Environment template
```

## 🏁 Run Guide

### Step 1: Set Up Environment
```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### Step 2: Configure API Keys
1. Copy `.env.example` to `.env`:
   ```bash
   cp .env.example .env
   ```
2. Open `.env` and paste your **GOOGLE_API_KEY**.

### Step 3: Run Validation Tests
Ensure everything is set up correctly:
```bash
python test_validation.py
```

### Step 4: Launch the App
```bash
streamlit run app.py
```
Open [http://localhost:8501](http://localhost:8501) in your browser.

## 📊 How it Works
1. **Load**: Documents are parsed into raw text.
2. **Chunk**: Text is split into overlapping snippets to preserve context.
3. **Embed**: Snippets are converted into high-dimensional vectors.
4. **Index**: Vectors are stored in FAISS for fast similarity search.
5. **Retrieve**: User queries are used to find relevant chunks.
6. **Reason**: Gemini generates a final answer based ONLY on the retrieved context.

---
**License**: MIT  
**Author**: College Final Year Student Project
