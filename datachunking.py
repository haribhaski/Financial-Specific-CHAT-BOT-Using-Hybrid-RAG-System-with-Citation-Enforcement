import os
from pathlib import Path
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from typing import List
import re

def _split_paragraphs(text: str) -> List[str]:
    # ✅ Handle both \n\n and \n \n (PyPDF injects spaces between newlines)
    parts = re.split(r'\n\s*\n', text)
    paragraphs = [p.strip() for p in parts if p.strip() and len(p.strip()) >= 40]
    return paragraphs

def load_all_docs(base_dir: str = "data/"):
    all_docs = []
    base = Path(base_dir)

    for root, _, files in os.walk(base):
        root_path = Path(root)
        rel_parts = root_path.relative_to(base).parts

        for filename in files:
            filepath = root_path / filename
            ext = filepath.suffix.lower()

            # ✅ Correct metadata mapping per README
            top = rel_parts[0] if rel_parts else ""
            if top == "sec-edgar-filings":
                category    = "SEC"
                institution = rel_parts[1] if len(rel_parts) > 1 else "UNKNOWN"
            elif top in {"RBI", "FED"}:
                category    = top        # "RBI" or "FED"
                institution = top
            else:
                category    = "UNKNOWN"
                institution = "UNKNOWN"

            try:
                if ext == ".txt":
                    loader = TextLoader(str(filepath), encoding="utf-8")
                elif ext == ".pdf":
                    loader = PyPDFLoader(str(filepath))
                else:
                    print(f"⏭️ Skipped: {filename}")
                    continue

                docs = loader.load()

                for doc in docs:
                    doc.metadata["source"]      = filename   # e.g. jpm_10k_2023.txt
                    doc.metadata["category"]    = category   # SEC / RBI / FED
                    doc.metadata["institution"] = institution # BAC / JPM / GS / RBI / FED

                all_docs.extend(docs)
                print(f"✅ Loaded: {filepath}")

            except Exception as e:
                print(f"❌ Failed: {filepath} — {e}")

    print(f"\n📦 Total documents loaded: {len(all_docs)}")
    return all_docs

import tiktoken

def _token_len(text: str) -> int:
    """Count tokens using GPT-2 tokenizer (fast, no API needed)."""
    enc = tiktoken.get_encoding("cl100k_base")
    return len(enc.encode(text))

def chunk_documents(docs: List[Document]) -> List[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=650,       # tokens — middle of 500-800 range
        chunk_overlap=100,    # tokens — per README spec
        length_function=_token_len,   # ✅ token-aware, not character-aware
        separators=["\n\n", "\n", ".", " ", ""]
    )

    all_chunks = []

    for doc in docs:
        source      = doc.metadata.get("source", "unknown")
        category    = doc.metadata.get("category", "unknown")
        institution = doc.metadata.get("institution", "unknown")
        page        = doc.metadata.get("page", None)

        # ✅ Split page text into real paragraphs by blank lines
        paragraphs = _split_paragraphs(doc.page_content)

        # Fallback: if no paragraph breaks found, treat whole page as one paragraph
        if not paragraphs:
            paragraphs = [doc.page_content.strip()] if doc.page_content.strip() else []
            
        for para_idx, para_text in enumerate(paragraphs):
            para_doc = Document(
                page_content=para_text,
                metadata={
                    "source"            : source,
                    "category"          : category,
                    "institution"       : institution,
                    "page"              : page,
                    "paragraph_index"   : para_idx,
                    "paragraph_preview" : para_text[:180]
                },
            )

            para_chunks = splitter.split_documents([para_doc])

            for sub_idx, chunk in enumerate(para_chunks):
                chunk.metadata["chunk_index"]    = sub_idx
                chunk.metadata["citation_label"] = (
                    f"{source}"
                    + (f", page {page + 1}" if isinstance(page, int) else "")
                    + f", paragraph {para_idx + 1}"
                )

            all_chunks.extend(para_chunks)

    # Filter noise
    before      = len(all_chunks)
    all_chunks  = [c for c in all_chunks if len(c.page_content.strip()) >= 100]
    removed     = before - len(all_chunks)

    print(f"📄 Original docs : {len(docs)}")
    print(f"🔪 Total chunks  : {len(all_chunks)}")
    print(f"🧹 Removed noise : {removed} short chunks")
    avg = len(all_chunks) // len(docs) if docs else 0
    print(f"📊 Avg chunks/doc: {avg}")

    return all_chunks

def inspect_chunks(chunks: List[Document], n: int = 3):
    print(f"\n🔍 Sample Chunks\n{'='*60}")
    for chunk in chunks[:n]:
        print(f"\n📌 Source     : {chunk.metadata['source']}")
        print(f"   Category   : {chunk.metadata['category']}")
        print(f"   Institution: {chunk.metadata['institution']}")
        page = chunk.metadata.get("page", None)
        print(f"   Page       : {page + 1 if isinstance(page, int) else 'N/A'}")
        print(f"   Paragraph  : {chunk.metadata.get('paragraph_index', -1) + 1}")
        print(f"   Chunk Index: {chunk.metadata['chunk_index']}")
        print(f"   Citation   : {chunk.metadata.get('citation_label', 'N/A')}")
        print(f"   Length     : {len(chunk.page_content)} chars")
        print(f"   Preview    : {chunk.page_content[:150]}...")
        print("-"*60)
        
if __name__ == "__main__":
    # Step 1 — Load
    docs = load_all_docs("data/")

    # Step 2 — Chunk
    chunks = chunk_documents(docs)

    # Step 3 — Inspect
    inspect_chunks(chunks, n=3)
    # Add to bottom of datachunking.py temporarily
    from collections import Counter

    sources = Counter(c.metadata["source"] for c in chunks)
    print("\n📊 Chunks per source:")
    for src, count in sources.most_common():
        print(f"  {src}: {count}")

    # Check for garbage chunks (too short)
    short = [c for c in chunks if len(c.page_content) < 100]
    print(f"\n⚠️  Short chunks (<100 chars): {len(short)}")
    print(f"   Example: {short[0].page_content if short else 'None'}")
    print("\n🏷️  Sample citations:")
    for c in chunks[10:13]:
        print(f"  {c.metadata.get('citation_label', 'MISSING')}")