#pip install sec-edgar-downloader
import os
from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from bs4 import BeautifulSoup

def clean_sec_filing(filepath: str) -> str:
    with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
        raw = f.read()
    
    # Strip all HTML/XML tags
    soup = BeautifulSoup(raw, "html.parser")
    clean_text = soup.get_text(separator="\n")
    
    # Remove excessive whitespace
    lines = [line.strip() for line in clean_text.splitlines() if line.strip()]
    return "\n".join(lines)

def load_all_docs(base_dir: str = "data/"):
    all_docs = []
    base = Path(base_dir)

    for root, _, files in os.walk(base):
        root_path = Path(root)
        rel_parts = root_path.relative_to(base).parts

        for filename in files:
            filepath = root_path / filename
            ext = filepath.suffix.lower()

            # Metadata mapping
            top = rel_parts[0] if rel_parts else ""
            if top == "sec-edgar-filings":
                category    = "SEC"
                institution = rel_parts[1] if len(rel_parts) > 1 else "UNKNOWN"
            elif top in {"RBI", "FED"}:
                category    = top
                institution = top
            else:
                category    = "UNKNOWN"
                institution = "UNKNOWN"

            try:
                if ext == ".txt":
                    # ✅ Clean SEC filings before loading
                    clean_text = clean_sec_filing(str(filepath))
                    doc = Document(
                        page_content=clean_text,
                        metadata={
                            "source"     : filename,
                            "category"   : category,
                            "institution": institution
                        }
                    )
                    all_docs.append(doc)
                    print(f"✅ Loaded (cleaned): {filepath}")

                elif ext in {".pdf", ".PDF"}:
                    loader = PyPDFLoader(str(filepath))
                    docs = loader.load()
                    for doc in docs:
                        doc.metadata["source"]      = filename
                        doc.metadata["category"]    = category
                        doc.metadata["institution"] = institution
                    all_docs.extend(docs)
                    print(f"✅ Loaded (pdf): {filepath}")

                else:
                    print(f"⏭️  Skipped: {filename}")
                    continue

            except Exception as e:
                print(f"❌ Failed: {filepath} — {e}")

    print(f"\n📦 Total documents loaded: {len(all_docs)}")
    return all_docs

if __name__ == "__main__":
    x = load_all_docs()