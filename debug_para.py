# Add this as a temp debug script: debug_paragraphs.py
from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader("data/RBI/PR6DFA5AD53D2D0414FAAB8D898975C40AA.PDF")
docs = loader.load()

for i, doc in enumerate(docs):
    print(f"\n=== Page {i+1} ===")
    print(repr(doc.page_content[:500]))  # repr shows actual \n characters