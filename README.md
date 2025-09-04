# Gemini RAG Chatbot

A Retrieval-Augmented Generation (RAG) chatbot powered by **Google Gemini** for answers grounded in your documents. Uses:
- **Gemini 1.5** (generation) via `google-generativeai`
- **text-embedding-004** (embeddings)
- **ChromaDB** (vector store)
- **Streamlit** (UI)
- **LangChain text splitters** (chunking)

## Quick Start

1. **Clone & install**
```bash
git clone <your-repo> rag-gemini
cd rag-gemini
python -m venv .venv && source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # put your real key in .env
python -m streamlit run app.py
