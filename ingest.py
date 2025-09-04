import os
from typing import List, Tuple, Dict

from dotenv import load_dotenv
from tqdm import tqdm
from pypdf import PdfReader
import markdown

from rag_core import new_splitter, add_text_chunks

load_dotenv()
DATA_DIR = os.getenv("DATA_DIR", "./data")

def read_txt(path: str) -> str:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()

def read_md(path: str) -> str:
    # Convert MD to plain-ish text by stripping HTML tags from rendered HTML
    raw = read_txt(path)
    html = markdown.markdown(raw)
    # crude strip
    return "".join(html.split("<")[0::2])

def read_pdf(path: str) -> str:
    reader = PdfReader(path)
    texts = []
    for p in reader.pages:
        txt = p.extract_text() or ""
        texts.append(txt)
    return "\n".join(texts)

def load_files() -> List[Tuple[str, Dict]]:
    splitter = new_splitter()
    chunks = []

    for root, _, files in os.walk(DATA_DIR):
        for fname in files:
            path = os.path.join(root, fname)
            ext = os.path.splitext(fname)[1].lower()

            if ext in [".txt"]:
                raw = read_txt(path)
            elif ext in [".md", ".markdown"]:
                raw = read_md(path)
            elif ext in [".pdf"]:
                raw = read_pdf(path)
            else:
                continue

            pieces = splitter.split_text(raw)
            for idx, piece in enumerate(pieces, start=1):
                meta = {
                    "source": os.path.relpath(path, DATA_DIR),
                    "page": idx if ext == ".pdf" else "",
                    "type": ext
                }
                chunks.append((piece, meta))

    return chunks

if __name__ == "__main__":
    os.makedirs(DATA_DIR, exist_ok=True)
    chunks = load_files()
    if not chunks:
        print(f"No supported files found in {DATA_DIR}. Add PDFs/TXT/MD and re-run.")
    else:
        print(f"Loaded {len(chunks)} chunks. Embedding and adding to vector store...")
        added = add_text_chunks(chunks)
        print(f"✅ Added {added} new chunks.")
