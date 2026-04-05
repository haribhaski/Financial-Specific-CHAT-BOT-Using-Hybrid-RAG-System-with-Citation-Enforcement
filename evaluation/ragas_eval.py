import os
import json
import sys
import time
from pathlib import Path

# ── Allow imports from project root ──────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from datasets import Dataset
from ragas import evaluate
from ragas.metrics import Faithfulness, AnswerRelevancy, ContextPrecision
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from qa_chain import format_context, PROMPT_TEMPLATE
from hybrid_retriever import load_vectorstore, build_bm25_index, build_hybrid_retriever

GOLDEN_PATH            = Path(__file__).parent / "golden_dataset.json"
FAITHFULNESS_THRESHOLD = 0.7
COHERE_RATE_LIMIT_SECS = 7


def run_evaluation():
    print("🔄 Loading vectorstore and BM25 index...")
    vectorstore = load_vectorstore()
    bm25_index  = build_bm25_index()

    print("📂 Loading golden dataset...")
    with open(GOLDEN_PATH) as f:
        golden = json.load(f)

    questions     = []
    ground_truths = []
    answers       = []
    contexts      = []

    # ── LLM for generation ────────────────────────────────────────────────────
    groq_llm  = ChatGroq(
        model="llama-3.1-8b-instant",
        temperature=0,
        api_key=os.environ["GROQ_API_KEY"]
    )

    # ── RAGAS judge LLM + embeddings ──────────────────────────────────────────
    ragas_llm  = LangchainLLMWrapper(groq_llm)
    ragas_emb  = LangchainEmbeddingsWrapper(
        HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    )

    print(f"🧪 Running {len(golden)} test cases...\n")

    for i, item in enumerate(golden):
        question    = item["question"]
        institution = item["institution"]
        gt          = item["ground_truth"]

        print(f"  [{i+1}/{len(golden)}] {question[:60]}...")

        if i > 0:
            print(f"   ⏳ Waiting {COHERE_RATE_LIMIT_SECS}s for Cohere rate limit...")
            time.sleep(COHERE_RATE_LIMIT_SECS)

        # ── Retrieve once ─────────────────────────────────────────────────────
        retriever = build_hybrid_retriever(
            vectorstore=vectorstore,
            bm25_index=bm25_index,
            institution_filter=institution,
            k=6
        )
        docs          = retriever.invoke(question)
        context_texts = [doc.page_content for doc in docs]

        # ── Generate answer ───────────────────────────────────────────────────
        formatted   = format_context(docs)
        prompt_text = PROMPT_TEMPLATE.format(context=formatted, question=question)
        answer      = groq_llm.invoke(prompt_text).content

        questions.append(question)
        ground_truths.append(gt)
        answers.append(answer)
        contexts.append(context_texts)

    # ── Build RAGAS dataset ───────────────────────────────────────────────────
    dataset = Dataset.from_dict({
        "question"    : questions,
        "answer"      : answers,
        "contexts"    : contexts,
        "ground_truth": ground_truths,
    })

    print("\n📊 Running RAGAS evaluation...")
    print("⏳ Waiting 30s before RAGAS scoring to avoid rate limits...")
    time.sleep(30)
    results = evaluate(
        dataset,
        metrics=[
            Faithfulness(llm=ragas_llm),
            AnswerRelevancy(llm=ragas_llm, embeddings=ragas_emb),
            ContextPrecision(llm=ragas_llm),
        ],
    )

    print("\n" + "="*50)
    print("📈 RAGAS Evaluation Results")
    print("="*50)
    df = results.to_pandas()
    print(df.to_string())

    mean_faithfulness = df["faithfulness"].mean()
    print(f"\n✅ Mean Faithfulness : {mean_faithfulness:.3f}")
    print(f"   Threshold         : {FAITHFULNESS_THRESHOLD}")

    if mean_faithfulness < FAITHFULNESS_THRESHOLD:
        print(f"\n❌ FAILED — faithfulness {mean_faithfulness:.3f} below threshold {FAITHFULNESS_THRESHOLD}")
        sys.exit(1)
    else:
        print(f"\n✅ PASSED — faithfulness {mean_faithfulness:.3f} meets threshold")
        sys.exit(0)


if __name__ == "__main__":
    run_evaluation()