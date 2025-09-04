# rag_core.py

import os
import hashlib
from typing import List, Dict, Tuple

import google.generativeai as genai
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
import chromadb

# ---------- env ----------
# Ensure .env values override any stale shell vars
load_dotenv(override=False)

GENERATION_MODEL = os.getenv("GENERATION_MODEL", "gemini-1.5-flash")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-004")
CHROMA_DIR = os.getenv("CHROMA_DIR", "./chroma_db")
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1200"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))

# ---------- helpers ----------
def _require_api_key() -> str:
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError(
            "Missing GOOGLE_API_KEY. Add it to your .env or export it in the shell."
        )
    return api_key

def init_genai_clients():
    """Configure Gemini and return a GenerativeModel for chat/QA."""
    genai.configure(api_key=_require_api_key())
    return genai.GenerativeModel(GENERATION_MODEL)

def new_splitter():
    return RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
        is_separator_regex=False,
    )

def doc_id(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()

def get_chroma_collection(name: str = "docs"):
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    return client.get_or_create_collection(name=name, metadata={"hnsw:space": "cosine"})

# ---------- embeddings ----------
def embed_texts(texts: List[str]) -> List[List[float]]:
    """
    Robustly call Gemini embeddings and normalize response shapes to:
      [[float, float, ...], [float, ...], ...]
    Handles:
      - {"embedding": [...]}
      - {"embedding": {"values":[...]}}
      - {"embeddings": [[...], ...]}
      - {"embeddings": [{"values":[...]}, ...]}
      - (Some clients) {"embeddings": {"values": [[...], ...]}}  <-- tricky
    """
    genai.configure(api_key=_require_api_key())

    # Coerce to list for uniform handling
    if isinstance(texts, str):
        texts = [texts]

    resp = genai.embed_content(model=EMBEDDING_MODEL, content=texts)

    if not isinstance(resp, dict):
        raise RuntimeError(f"Unexpected embedding response type: {type(resp)}")

    # Pull out the embeddings payload (single or batch)
    embs = None
    if "embeddings" in resp:
        embs = resp["embeddings"]
    elif "embedding" in resp:
        embs = [resp["embedding"]]
    else:
        raise RuntimeError(f"Unexpected embedding response keys: {list(resp.keys())}")

    # Normalize a few odd shapes:
    # Case A: dict with 'values' at batch level: {"embeddings": {"values": [[...],[...]]}}
    if isinstance(embs, dict) and "values" in embs:
        embs = embs["values"]

    # Case B: list of dicts with 'values'
    if isinstance(embs, list) and embs and isinstance(embs[0], dict) and "values" in embs[0]:
        embs = [e["values"] for e in embs]

    # Case C: single vector (list of floats) when we expected batch -> wrap
    if isinstance(embs, list) and embs and isinstance(embs[0], (float, int)):
        embs = [embs]

    # Case D: extra nesting like [[[...],[...]]]
    if (
        isinstance(embs, list)
        and embs
        and isinstance(embs[0], list)
        and len(embs) == 1
        and embs[0]
        and isinstance(embs[0][0], list)
    ):
        # flatten one level
        embs = embs[0]

    # Final validation
    if not (isinstance(embs, list) and embs and isinstance(embs[0], list) and isinstance(embs[0][0], (float, int))):
        raise RuntimeError(
            f"Embeddings not in expected shape after normalization: type={type(embs)}, sample={str(embs)[:200]}"
        )

    return embs

# ---------- add/search/generate ----------
def add_text_chunks(chunks: List[Tuple[str, Dict]]) -> int:
    """
    chunks: list of (text, metadata) where metadata includes:
            source, page (optional), type (ext)
    """
    col = get_chroma_collection()
    docs = [c[0] for c in chunks]
    metas = [c[1] for c in chunks]
    ids = [doc_id(d) for d in docs]

    # filter out empties early
    keep = [(d, m, i) for d, m, i in zip(docs, metas, ids) if d and d.strip()]
    if not keep:
        return 0

    docs, metas, ids = zip(*keep)

    existing = set(col.get(ids=list(ids))["ids"])
    docs_to_add, metas_to_add, ids_to_add = [], [], []
    for d, m, i in zip(docs, metas, ids):
        if i not in existing:
            docs_to_add.append(d)
            metas_to_add.append(m)
            ids_to_add.append(i)

    if not docs_to_add:
        return 0

    vecs = embed_texts(list(docs_to_add))

    # Safety: if embeddings came back with an extra nesting, flatten once
    if vecs and isinstance(vecs[0], list) and len(vecs) == 1 and vecs[0] and isinstance(vecs[0][0], list):
        vecs = vecs[0]

    if len(vecs) != len(docs_to_add):
        # Try last-resort fix: if vecs is triple-nested (rare client quirk)
        if len(vecs) == 1 and isinstance(vecs[0], list) and len(vecs[0]) == len(docs_to_add):
            vecs = vecs[0]
        else:
            raise RuntimeError(f"Embedding/doc count mismatch: got {len(vecs)} vectors for {len(docs_to_add)} docs")

    col.add(
        documents=list(docs_to_add),
        metadatas=list(metas_to_add),
        ids=list(ids_to_add),
        embeddings=vecs,
    )
    return len(docs_to_add)

def similarity_search(query: str, k: int = 5) -> List[Dict]:
    col = get_chroma_collection()
    q_emb = embed_texts([query])[0]

    # Chroma >= 0.5: do NOT include "ids" here
    res = col.query(
        query_embeddings=[q_emb],
        n_results=k,
        include=["documents", "metadatas", "distances"],  # no "ids"
    )

    out = []
    if res and res.get("documents"):
        docs = res["documents"][0]
        metas = res["metadatas"][0]
        dists = res.get("distances", [[None]])[0]
        ids_list = res.get("ids", [[]])[0]  # Chroma still returns ids even if not requested
        for i, (doc, meta, dist) in enumerate(zip(docs, metas, dists)):
            item = {
                "text": doc,
                "metadata": meta or {},
                "score": None if dist is None else float(dist),
            }
            if i < len(ids_list):
                item["id"] = ids_list[i]
            out.append(item)
    return out

SYSTEM_PROMPT = """You are a helpful RAG assistant. Use the provided CONTEXT to answer the user.
- Cite facts from the context only.
- If something is missing in the context, say what's missing and answer briefly if safe.
- Keep answers concise and structured.
"""

def build_prompt(context_chunks: List[Dict], user_question: str) -> List[Dict]:
    context_block = ""
    for i, c in enumerate(context_chunks, 1):
        md = c.get("metadata", {}) or {}
        src = md.get("source", "unknown")
        page = md.get("page", "")
        tag = f"{src}" + (f" (page {page})" if page != "" else "")
        context_block += f"\n[Chunk {i} — {tag}]\n{c['text']}\n"

    return [
        {
            "role": "user",
            "parts": (
                SYSTEM_PROMPT
                + "\n\nCONTEXT:\n"
                + context_block
                + f"\n\nQUESTION: {user_question}\n\nReturn a clear, step-by-step answer with any relevant bullet points."
            ),
        }
    ]

def answer_query(model, question: str, top_k: int = 5):
    """
    Runs retrieval + generation. Returns (answer_text, context_chunks).
    """
    context = similarity_search(question, k=top_k)
    msgs = build_prompt(context, question)
    resp = model.generate_content(msgs)
    answer = getattr(resp, "text", "") or ""
    return answer, context
