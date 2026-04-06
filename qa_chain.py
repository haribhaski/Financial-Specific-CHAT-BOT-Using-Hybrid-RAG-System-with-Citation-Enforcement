import os
import asyncio
import hashlib
from pathlib import Path
from collections import OrderedDict
from typing import Any, Dict, List, Optional
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from hybrid_retriever import load_vectorstore, build_bm25_index, build_hybrid_retriever

try:
    from langsmith import traceable
except Exception:  # pragma: no cover
    def traceable(*args, **kwargs):
        def _decorator(func):
            return func
        return _decorator

_ANSWER_CACHE: "OrderedDict[str, str]" = OrderedDict()
_CACHE_MAX_ITEMS = 256


def _get_env_var(name: str) -> str:
    value = os.getenv(name)
    if value:
        return value

    env_path = Path(__file__).resolve().parent / ".env"
    if env_path.exists():
        for line in env_path.read_text(encoding="utf-8", errors="ignore").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, raw = line.split("=", 1)
            if key.strip() == name:
                return raw.strip().strip('"').strip("'")

    raise RuntimeError(
        f"Missing required environment variable: {name}. "
        f"Set it in your shell or add it to .env in the project root."
    )


def _get_optional_env_var(name: str) -> Optional[str]:
    value = os.getenv(name)
    if value:
        return value

    env_path = Path(__file__).resolve().parent / ".env"
    if env_path.exists():
        for line in env_path.read_text(encoding="utf-8", errors="ignore").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, raw = line.split("=", 1)
            if key.strip() == name:
                return raw.strip().strip('"').strip("'")
    return None


def _configure_langsmith_for_qa() -> None:
    """Enable LangSmith tracing for QA when a LangSmith key is available."""
    ls_key = _get_optional_env_var("LANGSMITH_API_KEY") or _get_optional_env_var("LANGCHAIN_API_KEY")
    if not ls_key:
        return

    os.environ["LANGSMITH_API_KEY"] = ls_key
    os.environ.setdefault("LANGSMITH_TRACING", "true")
    os.environ.setdefault("LANGCHAIN_TRACING_V2", "true")
    os.environ.setdefault("LANGSMITH_PROJECT", _get_optional_env_var("LANGSMITH_PROJECT") or "FinRAG-QA")

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


@traceable
def format_prompt(question: str, context: str) -> str:
    return PROMPT_TEMPLATE.format(context=context, question=question)


@traceable(run_type="llm")
async def invoke_llm_async(llm: ChatGroq, prompt_text: str) -> Any:
    try:
        return await llm.ainvoke(prompt_text)
    except Exception:
        return await asyncio.to_thread(llm.invoke, prompt_text)


@traceable
def parse_output(response: Any) -> str:
    return response.content if hasattr(response, "content") else str(response)


@traceable
async def run_pipeline(question: str, docs: List[Any], llm: ChatGroq) -> str:
    context = format_context(docs)
    prompt_text = format_prompt(question, context)
    response = await invoke_llm_async(llm, prompt_text)
    return parse_output(response)


def _cache_get(key: str) -> Optional[str]:
    value = _ANSWER_CACHE.get(key)
    if value is not None:
        _ANSWER_CACHE.move_to_end(key)
    return value


def _cache_set(key: str, value: str) -> None:
    _ANSWER_CACHE[key] = value
    _ANSWER_CACHE.move_to_end(key)
    while len(_ANSWER_CACHE) > _CACHE_MAX_ITEMS:
        _ANSWER_CACHE.popitem(last=False)


def _hash_query_context(question: str, context: str) -> str:
    payload = f"{question}\n---\n{context}".encode("utf-8", errors="ignore")
    return hashlib.sha256(payload).hexdigest()


async def async_retrieve_documents(
    question: str,
    institution_filter: str = None,
    vectorstore=None,
    bm25_index=None,
    k: int = 6,
) -> List[Any]:
    if vectorstore is None:
        vectorstore = load_vectorstore()
    if bm25_index is None:
        bm25_index = build_bm25_index()

    retriever = build_hybrid_retriever(
        vectorstore=vectorstore,
        bm25_index=bm25_index,
        institution_filter=institution_filter,
        k=k,
    )
    return await asyncio.to_thread(retriever.invoke, question)


async def async_answer_with_cache(
    question: str,
    institution_filter: str = None,
    vectorstore=None,
    bm25_index=None,
    k: int = 6,
) -> Dict[str, Any]:
    _configure_langsmith_for_qa()

    docs = await async_retrieve_documents(
        question=question,
        institution_filter=institution_filter,
        vectorstore=vectorstore,
        bm25_index=bm25_index,
        k=k,
    )

    context = format_context(docs)
    cache_key = _hash_query_context(question, context)
    cached = _cache_get(cache_key)
    if cached is not None:
        return {"answer": cached, "docs": docs, "cache_hit": True}

    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0,
        api_key=_get_env_var("GROQ_API_KEY"),
    )

    answer = await run_pipeline(question=question, docs=docs, llm=llm)
    _cache_set(cache_key, answer)
    return {"answer": answer, "docs": docs, "cache_hit": False}


def build_qa_chain(institution_filter: str = None, vectorstore=None,
                   bm25_index=None, k: int = 6):
    _configure_langsmith_for_qa()

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
        api_key=_get_env_var("GROQ_API_KEY")
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