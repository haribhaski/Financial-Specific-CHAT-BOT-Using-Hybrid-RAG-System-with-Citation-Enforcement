import os
import streamlit as st
from hybrid_retriever import load_vectorstore, build_bm25_index, build_hybrid_retriever
from qa_chain import build_qa_chain

st.set_page_config(
    page_title="FinRAG — Financial Document QA",
    page_icon="📊",
    layout="centered"
)

st.title("📊 FinRAG — Financial Document QA")
st.caption("Ask questions about SEC filings, Fed statements, and RBI reports.")

# ── Load once, cache forever ──────────────────────────────────────────────────
@st.cache_resource
def get_vectorstore():
    return load_vectorstore()

@st.cache_resource
def get_bm25_index():
    return build_bm25_index()

vectorstore = get_vectorstore()
bm25_index  = get_bm25_index()

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("🔧 Settings")
    institution = st.selectbox(
        "Filter by institution",
        options=["All", "JPM", "GS", "BAC", "FED", "RBI"],
        index=0
    )
    k = st.slider("Chunks to retrieve", min_value=3, max_value=10, value=6)
    st.markdown("---")
    st.markdown("**Sources loaded:**")
    st.markdown("- 🏦 JPM 10-K\n- 🏦 GS 10-K\n- 🏦 BAC 10-K\n- 🏛️ Fed Statement\n- 🇮🇳 RBI Reports")

# ── Chat history ──────────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ── Input ─────────────────────────────────────────────────────────────────────
if question := st.chat_input("Ask a financial question..."):

    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Retrieving and reasoning..."):
            try:
                filter_val = None if institution == "All" else institution
                chain = build_qa_chain(
                    institution_filter=filter_val,
                    vectorstore=vectorstore,
                    bm25_index=bm25_index,
                    k=k
                )
                answer = chain.invoke(question)
                st.markdown(answer)
                st.session_state.messages.append({"role": "assistant", "content": answer})

            except Exception as e:
                st.error(f"❌ Error: {str(e)}")

# ── Empty state ───────────────────────────────────────────────────────────────
if not st.session_state.messages:
    st.markdown("### 💡 Try asking:")
    cols = st.columns(2)
    examples = [
        ("What did the Fed decide on rates?", "FED"),
        ("What is Goldman Sachs revenue?",     "GS"),
        ("What are BAC's risk factors?",       "BAC"),
        ("What is JPMorgan's total assets?",   "JPM"),
    ]
    for i, (q, inst) in enumerate(examples):
        with cols[i % 2]:
            st.code(f"{q}\n[Filter: {inst}]", language=None)