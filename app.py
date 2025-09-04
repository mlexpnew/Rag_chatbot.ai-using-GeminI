# app.py
import os, streamlit as st
for k, v in st.secrets.items():
    os.environ.setdefault(k, str(v))
import os
import io
import shutil
import traceback
import streamlit as st

from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from rag_core import (
    init_genai_clients,
    similarity_search,
    build_prompt,
    add_text_chunks,
)

# ------------- UI SETUP -------------
st.set_page_config(page_title="Gemini RAG Chatbot", page_icon="🤖", layout="wide")
st.title("Gemini RAG Chatbot")
st.caption("Ask questions grounded in your documents (PDF / TXT / MD).")

# ------------- SIDEBAR: DOC INGEST -------------
st.sidebar.header("Add Documents")

uploaded = st.sidebar.file_uploader(
    "Drag and drop files here",
    type=["pdf", "txt", "md", "markdown"],
    accept_multiple_files=True,
)

chunk_size = st.sidebar.number_input("Chunk size", min_value=200, max_value=4000, value=1200, step=50)
chunk_overlap = st.sidebar.number_input("Chunk overlap", min_value=0, max_value=2000, value=200, step=25)

# Clear vector DB
if st.sidebar.button("Clear vector DB"):
    shutil.rmtree("chroma_db", ignore_errors=True)
    os.makedirs("chroma_db", exist_ok=True)
    st.sidebar.success("Vector store cleared.")

# Helpers to read files
def _read_pdf_bytes(file_bytes: bytes) -> str:
    try:
        reader = PdfReader(io.BytesIO(file_bytes))
        return "\n".join([(p.extract_text() or "") for p in reader.pages])
    except Exception:
        return ""

def _read_text_bytes(file_bytes: bytes) -> str:
    try:
        return file_bytes.decode("utf-8", errors="ignore")
    except Exception:
        return ""

def _split(text: str):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=int(chunk_size),
        chunk_overlap=int(chunk_overlap),
        length_function=len,
        is_separator_regex=False,
    )
    return splitter.split_text(text)

def _ingest_uploaded_files(files) -> int:
    os.makedirs("data", exist_ok=True)
    total_added = 0
    chunks = []

    for f in files:
        name = f.name
        ext = os.path.splitext(name)[1].lower()
        raw = ""

        data = f.read()
        if ext == ".pdf":
            raw = _read_pdf_bytes(data)
        elif ext in (".txt", ".md", ".markdown"):
            raw = _read_text_bytes(data)

        if raw.strip():
            for piece in _split(raw):
                chunks.append((piece, {"source": name, "type": ext}))

        # Also save a copy to data/ for your records (optional)
        with open(os.path.join("data", name), "wb") as out:
            out.write(data)

    if chunks:
        total_added = add_text_chunks(chunks)
    return total_added

if st.sidebar.button("Ingest Uploaded Files") and uploaded:
    try:
        n = _ingest_uploaded_files(uploaded)
        st.sidebar.success(f"Ingested {n} new chunks.")
    except Exception as e:
        st.sidebar.error(f"Ingest failed: {e}")
        st.sidebar.code(traceback.format_exc())

st.sidebar.caption("Tip: you can also drop files into the `data/` folder and run `python ingest.py` for bulk indexing.")

# ------------- MAIN: QUESTION AREA -------------
q = st.text_input("Ask a question about your documents:", "")
top_k = st.slider("Top-K passages", 3, 12, 5)

ask = st.button("Ask")

if ask:
    if not q.strip():
        st.warning("Please enter a question.")
        st.stop()

    # Try generating with streaming; show nice errors
    try:
        # Make sure API key is present (rag_core will raise otherwise)
        model = init_genai_clients()

        # Retrieve relevant chunks
        ctx = similarity_search(q, k=top_k)
        msgs = build_prompt(ctx, q)

        # Stream the answer
        st.subheader("Answer")
        with st.chat_message("assistant"):
            ph = st.empty()
            buf = []
            resp = model.generate_content(msgs, stream=True)
            for chunk in resp:
                text = getattr(chunk, "text", "") or ""
                if text:
                    buf.append(text)
                    ph.markdown("".join(buf))
            # finalize streamed response
            resp.resolve()

        # Show sources
        if ctx:
            st.subheader("Sources")
            for i, c in enumerate(ctx, 1):
                md = c.get("metadata", {}) or {}
                src = md.get("source", "unknown")
                page = md.get("page", "")
                dist = c.get("score")
                line = f"- **{i}. {src}**"
                if page != "":
                    line += f" — page {page}"
                if dist is not None:
                    try:
                        line += f" — distance: `{float(dist):.4f}`"
                    except Exception:
                        pass
                st.markdown(line)
        else:
            st.info("No matching passages were retrieved. Try re-ingesting or asking a broader question.")

    except Exception as e:
        msg = str(e)
        # Helpful hint for API key issues
        if "API key not valid" in msg or "API_KEY_INVALID" in msg:
            st.error("Your Google AI Studio API key appears invalid. Check `.env` → `GOOGLE_API_KEY=AIza...` and restart.")
        else:
            st.error(f"Error: {msg}")
        st.caption("See terminal logs for a full traceback.")
