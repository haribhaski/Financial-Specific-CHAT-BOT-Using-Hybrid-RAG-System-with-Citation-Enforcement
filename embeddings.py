import os
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from datachunking import load_all_docs, chunk_documents

CHROMA_DIR = "chroma_db"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

def build_vectorstore(chunks, persist_dir: str = CHROMA_DIR):
    print(f"\n🤖 Loading embedding model: {EMBED_MODEL}")
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBED_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )

    print(f"📥 Embedding {len(chunks)} chunks into Chroma...")
    print("   (this may take 5–15 mins for 6k chunks on CPU)")

    # Batch to avoid memory spikes
    BATCH_SIZE = 500
    vectorstore = None

    for i in range(0, len(chunks), BATCH_SIZE):
        batch = chunks[i : i + BATCH_SIZE]
        batch_num = i // BATCH_SIZE + 1
        total_batches = (len(chunks) + BATCH_SIZE - 1) // BATCH_SIZE
        print(f"   Batch {batch_num}/{total_batches} — chunks {i} to {i + len(batch)}")

        if vectorstore is None:
            vectorstore = Chroma.from_documents(
                documents=batch,
                embedding=embeddings,
                persist_directory=persist_dir
            )
        else:
            vectorstore.add_documents(batch)

    print(f"\n✅ Vectorstore saved to: {persist_dir}/")
    print(f"   Total vectors stored: {vectorstore._collection.count()}")
    return vectorstore


def load_vectorstore(persist_dir: str = CHROMA_DIR):
    print(f"📂 Loading existing vectorstore from {persist_dir}/")
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBED_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )
    vectorstore = Chroma(
        persist_directory=persist_dir,
        embedding_function=embeddings
    )
    print(f"   Vectors loaded: {vectorstore._collection.count()}")
    return vectorstore


if __name__ == "__main__":
    # Skip rebuilding if DB already exists
    if os.path.exists(CHROMA_DIR) and os.listdir(CHROMA_DIR):
        print("⚡ Chroma DB already exists — loading instead of rebuilding.")
        vs = load_vectorstore()
    else:
        docs = load_all_docs("data/")
        chunks = chunk_documents(docs)
        vs = build_vectorstore(chunks)

    # Quick sanity check
    print("\n🔍 Test query: 'What is JPM net income?'")
    results = vs.similarity_search("What is JPM net income?", k=3)
    for r in results:
        print(f"\n  📌 {r.metadata.get('citation_label', 'N/A')}")
        print(f"     {r.page_content[:200]}...")