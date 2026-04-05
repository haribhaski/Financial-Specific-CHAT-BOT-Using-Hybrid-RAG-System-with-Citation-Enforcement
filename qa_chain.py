import os
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from hybrid_retriever import load_vectorstore, build_bm25_index, build_hybrid_retriever

# ── Prompt ────────────────────────────────────────────────────────────────────
PROMPT_TEMPLATE = """You are a financial analyst assistant. Answer the question
using ONLY the context provided. Be precise and cite your sources.

For each key fact in your answer, append the citation in brackets like:
[monetary20260128a1.pdf, page 1, paragraph 1]

If the context does not contain enough information, say:
"I don't have sufficient information in the provided documents to answer this."

Context:
{context}

Question: {question}

Answer:"""

PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template=PROMPT_TEMPLATE
)


def format_context(docs):
    """Concatenate chunks with their citation labels."""
    parts = []
    for doc in docs:
        citation = doc.metadata.get("citation_label", "unknown source")
        parts.append(f"[{citation}]\n{doc.page_content}")
    return "\n\n---\n\n".join(parts)


def build_qa_chain(institution_filter: str = None, vectorstore=None,
                   bm25_index=None, k: int = 6):
    if vectorstore is None:
        vectorstore = load_vectorstore()
    if bm25_index is None:
        bm25_index = build_bm25_index()

    retriever = build_hybrid_retriever(
        vectorstore=vectorstore,
        bm25_index=bm25_index,
        institution_filter=institution_filter,
        k=k
    )

    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0,
        api_key=os.environ["GROQ_API_KEY"]
    )

    # ✅ Use RunnableLambda since HybridRetriever is not a LangChain Runnable
    chain = (
        {
            "context": RunnableLambda(retriever.invoke) | format_context,
            "question": RunnablePassthrough()
        }
        | PROMPT
        | llm
        | StrOutputParser()
    )

    return chain


# ── CLI test ──────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Load once, reuse across all questions
    vectorstore = load_vectorstore()
    bm25_index  = build_bm25_index()

    test_cases = [
        ("What is JPM's net income?",         "JPM"),
        ("What is Goldman Sachs revenue?",     "GS"),
        ("What did the Fed decide on rates?",  "FED"),
        ("What is RBI's policy stance?",       "RBI"),
    ]

    for question, institution in test_cases:
        print(f"\n{'='*60}")
        print(f"❓ Q: {question}  [filter: {institution}]")
        print("="*60)
        chain = build_qa_chain(
            institution_filter=institution,
            vectorstore=vectorstore,
            bm25_index=bm25_index,
            k=6
        )
        answer = chain.invoke(question)
        print(answer)