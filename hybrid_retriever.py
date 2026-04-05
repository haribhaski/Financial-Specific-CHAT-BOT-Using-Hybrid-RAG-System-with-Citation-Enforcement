import os
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_cohere import CohereRerank
from langchain_core.documents import Document
from rank_bm25 import BM25Okapi
from datachunking import load_all_docs, chunk_documents
import numpy as np
from typing import List, Optional

CHROMA_DIR = "chroma_db"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


# ── BM25 Index ────────────────────────────────────────────────────────────────
class BM25Index:
    def __init__(self, chunks: List[Document]):
        self.chunks = chunks
        tokenized = [self._tokenize(c.page_content) for c in chunks]
        self.bm25 = BM25Okapi(tokenized)
        print(f"✅ BM25 index built over {len(chunks)} chunks")

    def _tokenize(self, text: str) -> List[str]:
        return [
            w.lower() for w in text.split()
            if w.isalnum() or "." in w
        ]

    def search(self, query: str, k: int = 20,
               institution_filter: Optional[str] = None) -> List[Document]:
        tokens = self._tokenize(query)
        scores = self.bm25.get_scores(tokens)

        if institution_filter:
            for i, chunk in enumerate(self.chunks):
                if chunk.metadata.get("institution") != institution_filter:
                    scores[i] = 0.0

        top_indices = np.argsort(scores)[::-1][:k]
        return [self.chunks[idx] for idx in top_indices if scores[idx] > 0]


# ── Hybrid Retriever ──────────────────────────────────────────────────────────
class HybridRetriever:
    def __init__(self, vectorstore: Chroma, bm25_index: BM25Index,
                 institution_filter: Optional[str] = None, k: int = 6):
        self.vectorstore = vectorstore
        self.bm25_index = bm25_index
        self.institution_filter = institution_filter
        self.k = k

        # ✅ Cohere reranker — replaces HuggingFace cross-encoder
        self.reranker = CohereRerank(
            cohere_api_key=os.environ["COHERE_API_KEY"],
            model="rerank-english-v3.0",
            top_n=k
        )

    def invoke(self, query: str) -> List[Document]:
        fetch_k = self.k * 3

        # ── 1. BM25 retrieval ─────────────────────────────────────────────────
        bm25_results = self.bm25_index.search(
            query, k=fetch_k,
            institution_filter=self.institution_filter
        )

        # ── 2. Vector retrieval ───────────────────────────────────────────────
        search_kwargs = {"k": fetch_k}
        if self.institution_filter:
            search_kwargs["filter"] = {"institution": self.institution_filter}
        vector_results = self.vectorstore.similarity_search(query, **search_kwargs)

        # ── 3. Merge + deduplicate ────────────────────────────────────────────
        seen = set()
        merged = []
        for doc in bm25_results + vector_results:
            key = doc.page_content[:100]
            if key not in seen:
                seen.add(key)
                merged.append(doc)

        print(f"   BM25: {len(bm25_results)} | Vector: {len(vector_results)} "
              f"| Merged: {len(merged)} | After rerank → {self.k}")

        # ── 4. Cohere rerank ──────────────────────────────────────────────────
        reranked = self.reranker.compress_documents(merged, query)
        return reranked


# ── Factory functions ─────────────────────────────────────────────────────────
def load_vectorstore():
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBED_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )
    return Chroma(persist_directory=CHROMA_DIR, embedding_function=embeddings)


def build_bm25_index() -> BM25Index:
    docs = load_all_docs("data/")
    chunks = chunk_documents(docs)
    return BM25Index(chunks)


def build_hybrid_retriever(vectorstore: Chroma, bm25_index: BM25Index,
                           institution_filter: Optional[str] = None,
                           k: int = 6) -> HybridRetriever:
    return HybridRetriever(
        vectorstore=vectorstore,
        bm25_index=bm25_index,
        institution_filter=institution_filter,
        k=k
    )


# ── Quick test ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    vs = load_vectorstore()
    bm25 = build_bm25_index()

    test_cases = [
        ("What is JPM net income?",           "JPM"),
        ("What did the Fed decide on rates?",  "FED"),
        ("What is Goldman Sachs revenue?",     "GS"),
        ("What happened to 2000 rupee note?",  "RBI"),
    ]

    for query, institution in test_cases:
        print(f"\n{'='*60}")
        print(f"❓ {query}  [filter: {institution}]")
        print("="*60)
        retriever = build_hybrid_retriever(
            vs, bm25, institution_filter=institution, k=3
        )
        results = retriever.invoke(query)
        for doc in results:
            print(f"\n  📌 {doc.metadata.get('citation_label', 'N/A')}")
            print(f"     {doc.page_content[:200]}")