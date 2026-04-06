import os
import json
import re
import io
import csv
import asyncio
from collections import defaultdict
import streamlit as st
import streamlit.components.v1 as components
from streamlit.errors import StreamlitSecretNotFoundError
from pypdf import PdfReader
from langchain_groq import ChatGroq
from neo4j import GraphDatabase
from pyvis.network import Network
from hybrid_retriever import load_vectorstore, build_bm25_index, build_hybrid_retriever
from qa_chain import async_answer_with_cache, async_retrieve_documents

st.set_page_config(
    page_title="FinRAG — Financial Document QA",
    page_icon="📊",
    layout="centered"
)

st.title("📊 FinRAG — Financial Document QA")
st.caption("Ask questions about SEC filings, Fed statements, and RBI reports.")


# ── Helpers ───────────────────────────────────────────────────────────────────

def _run_async(coro):
    try:
        return asyncio.run(coro)
    except RuntimeError:
        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(coro)
        finally:
            loop.close()


def _read_uploaded_text(uploaded_file) -> str:
    if uploaded_file.name.lower().endswith(".txt"):
        return uploaded_file.getvalue().decode("utf-8", errors="ignore")
    if uploaded_file.name.lower().endswith(".pdf"):
        reader = PdfReader(uploaded_file)
        pages = [page.extract_text() or "" for page in reader.pages]
        return "\n".join(pages)
    return ""


def _extract_json_object(raw_text: str) -> dict:
    match = re.search(r"\{.*\}", raw_text, flags=re.DOTALL)
    if not match:
        return {"nodes": [], "edges": []}
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return {"nodes": [], "edges": []}


def _safe_secret(path: str, default=None):
    """Read Streamlit secrets without crashing when secrets.toml is absent."""
    try:
        cur = st.secrets
        for key in path.split("."):
            if key not in cur:
                return default
            cur = cur[key]
        return cur
    except StreamlitSecretNotFoundError:
        return default


# ── Neo4j ─────────────────────────────────────────────────────────────────────

@st.cache_resource
def get_neo4j_driver():
    """
    Reads connection details from Streamlit secrets or environment variables.
    In secrets.toml:
        [neo4j]
        uri      = "neo4j+s://xxxxxxxx.databases.neo4j.io"
        username = "neo4j"
        password = "your-aura-password"
    Or set NEO4J_URI / NEO4J_USERNAME / NEO4J_PASSWORD as env vars.
    """
    uri = _safe_secret("neo4j.uri") or os.environ.get("NEO4J_URI")
    username = _safe_secret("neo4j.username") or os.environ.get("NEO4J_USERNAME", "neo4j")
    password = _safe_secret("neo4j.password") or os.environ.get("NEO4J_PASSWORD")

    if not uri or not password:
        return None

    try:
        driver = GraphDatabase.driver(uri, auth=(username, password))
        driver.verify_connectivity()
        return driver
    except Exception as e:
        st.warning(f"⚠️ Neo4j connection failed: {e}")
        return None


def _push_graph_to_neo4j(driver, graph_data: dict, source_label: str) -> bool:
    """
    Upserts nodes and relationships into Neo4j.
    Each node gets a :Entity label plus a source property so graphs from
    different uploads don't collide but can be queried together.
    """
    if not driver:
        return False

    try:
        with driver.session() as session:
            # Clear previous graph for this source so re-uploads are clean
            session.run(
                "MATCH (n:Entity {source: $src}) DETACH DELETE n",
                src=source_label,
            )

            # Create nodes
            for node_name in graph_data.get("nodes", []):
                session.run(
                    """
                    MERGE (n:Entity {name: $name, source: $src})
                    SET n.label = $name
                    """,
                    name=node_name,
                    src=source_label,
                )

            # Create relationships
            for edge in graph_data.get("edges", []):
                rel_type = re.sub(r"[^A-Za-z0-9_]", "_", edge["relation"].upper())
                session.run(
                    f"""
                    MATCH (a:Entity {{name: $source, source: $src}})
                    MATCH (b:Entity {{name: $target, source: $src}})
                    MERGE (a)-[r:{rel_type}]->(b)
                    SET r.label = $relation
                    """,
                    source=edge["source"],
                    target=edge["target"],
                    relation=edge["relation"],
                    src=source_label,
                )
        return True
    except Exception as e:
        st.error(f"Neo4j write error: {e}")
        return False


def _load_graph_from_neo4j(driver, source_label: str) -> dict:
    """Reads back the graph stored under a given source label."""
    if not driver:
        return {"nodes": [], "edges": []}

    try:
        with driver.session() as session:
            result = session.run(
                """
                MATCH (a:Entity {source: $src})-[r]->(b:Entity {source: $src})
                RETURN a.name AS source, b.name AS target, r.label AS relation
                """,
                src=source_label,
            )
            edges = [{"source": r["source"], "target": r["target"], "relation": r["relation"]}
                     for r in result]

            node_result = session.run(
                "MATCH (n:Entity {source: $src}) RETURN n.name AS name",
                src=source_label,
            )
            nodes = [r["name"] for r in node_result]

        return {"nodes": nodes, "edges": edges}
    except Exception as e:
        st.error(f"Neo4j read error: {e}")
        return {"nodes": [], "edges": []}


# ── LLM extraction ────────────────────────────────────────────────────────────

# Finance-specific entity types and relation vocabulary.
# This is the core fix for "generic entities" and "meaningless relations".
_EXTRACTION_PROMPT = """You are a financial knowledge graph extractor. Your job is to extract
precise, meaningful entities and relationships from financial documents.

ENTITY TYPES to extract (be specific, not generic):
- Institutions: banks, central banks, regulators, exchanges (e.g. "JPMorgan Chase", "Federal Reserve", "SEBI")
- Financial metrics: revenue figures, rates, ratios with their values (e.g. "Net Revenue $158B", "Fed Funds Rate 5.25%")
- Products & instruments: loan types, securities, indices (e.g. "10-Year Treasury", "HDFC Home Loan", "S&P 500")
- Policies & regulations: named acts, frameworks, decisions (e.g. "Basel III", "Dodd-Frank Act", "Rate Hike Decision")
- Time periods: quarters, fiscal years (e.g. "Q3 2024", "FY2023")
- People: named executives, officials (e.g. "Jerome Powell", "Shaktikanta Das")
- Economic concepts with context: not just "inflation" but "CPI Inflation 3.2%"

RELATION TYPES to use (choose the most specific):
reported_revenue_of, acquired, regulates, issued_guidance_on, raised_rate_to,
cut_rate_to, approved, rejected, invested_in, divested_from, partnered_with,
subsidiary_of, reported_loss_of, increased_by, decreased_by, set_target_of,
warned_about, responded_to, enforced_by, linked_to, impacted_by

Return JSON ONLY — no markdown, no explanation:
{{
    "nodes": ["entity1", "entity2", ...],
    "edges": [
        {{"source": "entity1", "target": "entity2", "relation": "relation_from_list_above"}}
    ]
}}

Rules:
- 5-25 nodes, 10-35 edges. (Flexible)
- Every node in edges must also appear in nodes list.
- Prefer specific named entities over generic category labels.
- Use exact figures where mentioned (e.g. "$54.9B" not just "revenue").
- No duplicate edges.
- Never output stopword-like nodes such as: The, This, That, From, Further, Thus.
- Do not output standalone month names as entities unless tied to an event name.
- This is a fincance based Graph so ensure key financial terms and make the main node as the company

Document text:
"""


_NOISY_SINGLE_TOKENS = {
    "the", "this", "that", "from", "further", "thus", "issue", "it", "they", "we",
    "april", "march", "may", "june", "july", "august", "september", "october", "november", "december",
    "january", "february",
}

_KEEP_SINGLE_TOKEN = {"rbi", "fed", "jpm", "gs", "bac", "india", "sebi"}


def _normalize_entity(raw: str) -> str:
    value = re.sub(r"\s+", " ", str(raw or "").strip())
    value = value.strip(" ,.;:-_()[]{}\"'")
    if not value:
        return ""

    token_count = len(value.split())
    low = value.lower()

    if token_count == 1 and low in _NOISY_SINGLE_TOKENS:
        return ""
    if token_count == 1 and low not in _KEEP_SINGLE_TOKEN:
        # Keep meaningful single-word financial tokens only when alphanumeric and not too short.
        if len(value) < 4 or not re.search(r"[A-Za-z]", value):
            return ""

    # Drop entities that are mostly punctuation/numbers without semantic tag.
    if not re.search(r"[A-Za-z]", value):
        return ""

    return value


def _sanitize_graph(graph_data: dict) -> dict:
    nodes = graph_data.get("nodes", [])
    edges = graph_data.get("edges", [])

    cleaned_nodes = {
        _normalize_entity(n) for n in nodes
        if _normalize_entity(n)
    }

    cleaned_edges = []
    seen = set()
    degree = defaultdict(int)

    for edge in edges:
        if not isinstance(edge, dict):
            continue
        source = _normalize_entity(edge.get("source", ""))
        target = _normalize_entity(edge.get("target", ""))
        relation = re.sub(r"\s+", "_", str(edge.get("relation", "linked_to")).strip().lower()) or "linked_to"

        if not source or not target or source == target:
            continue

        key = (source, relation, target)
        if key in seen:
            continue
        seen.add(key)

        cleaned_nodes.add(source)
        cleaned_nodes.add(target)
        degree[source] += 1
        degree[target] += 1

        cleaned_edges.append({"source": source, "target": target, "relation": relation})

    main_node = ""
    if degree:
        main_node = max(degree.items(), key=lambda x: x[1])[0]

    return {
        "nodes": sorted(cleaned_nodes),
        "edges": cleaned_edges,
        "main_node": main_node,
    }


def _extract_graph_with_llm(text: str) -> dict:
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        return {"nodes": [], "edges": []}

    llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0, api_key=api_key)
    raw = llm.invoke(_EXTRACTION_PROMPT + text[:14000]).content
    data = _extract_json_object(raw)

    nodes = data.get("nodes", [])
    edges = data.get("edges", [])
    if not isinstance(nodes, list) or not isinstance(edges, list):
        return {"nodes": [], "edges": []}

    cleaned_edges = []
    seen = set()
    for edge in edges:
        if not isinstance(edge, dict):
            continue
        source   = str(edge.get("source", "")).strip()
        target   = str(edge.get("target", "")).strip()
        relation = str(edge.get("relation", "linked_to")).strip() or "linked_to"
        key = (source, target, relation)
        if source and target and key not in seen:
            cleaned_edges.append({"source": source, "target": target, "relation": relation})
            seen.add(key)

    unique_nodes = sorted({str(n).strip() for n in nodes if str(n).strip()})
    if not unique_nodes:
        unique_nodes = sorted(
            {e["source"] for e in cleaned_edges} | {e["target"] for e in cleaned_edges}
        )

    return _sanitize_graph({"nodes": unique_nodes, "edges": cleaned_edges})


def _extract_graph_heuristic(text: str) -> dict:
    sentences = re.split(r"(?<=[.!?])\s+", text[:8000])
    relation_terms = ["increased", "decreased", "acquired", "announced", "approved",
                      "reported", "raised", "cut", "regulates", "invested", "warned"]
    edges = []
    seen = set()

    for sent in sentences:
        # Prefer 2+ token named entities, then fallback to known single-token entities.
        entities = re.findall(r"\b[A-Z][A-Za-z&.\-]{1,}(?:\s+[A-Z][A-Za-z&.\-]{1,})+\b", sent)
        singles = re.findall(r"\b[A-Z][A-Za-z&.\-]{2,}\b", sent)
        singles = [s for s in singles if s.lower() in _KEEP_SINGLE_TOKEN]
        entities.extend(singles)
        if len(entities) < 2:
            continue
        lower_sent = sent.lower()
        relation = next((t for t in relation_terms if t in lower_sent), "linked_to")
        source, target = entities[0], entities[1]
        key = (source, target)
        if source != target and key not in seen:
            edges.append({"source": source, "target": target, "relation": relation})
            seen.add(key)
        if len(edges) >= 35:
            break

    nodes = sorted({e["source"] for e in edges} | {e["target"] for e in edges})
    return _sanitize_graph({"nodes": nodes[:25], "edges": edges[:35]})


# ── Pyvis visualisation ───────────────────────────────────────────────────────

# Colour palette: nodes get one of these based on a simple hash so the same
# entity always gets the same colour across renders.
_NODE_COLOURS = [
    "#4E9AF1", "#F16B4E", "#4EF1A0", "#F1D44E",
    "#A04EF1", "#F14E9A", "#4EE0F1", "#F1A04E",
]


def _render_pyvis(graph_data: dict, height: int = 600) -> str:
    """
    Builds an interactive pyvis graph and returns the raw HTML string.
    Physics is tuned for finance graphs: spread out, legible labels.
    """
    net = Network(
        height=f"{height}px",
        width="100%",
        bgcolor="#0f1117",
        font_color="#e0e0e0",
        directed=True,
    )

    net.set_options("""
    {
      "physics": {
        "enabled": true,
        "forceAtlas2Based": {
          "gravitationalConstant": -80,
          "centralGravity": 0.01,
          "springLength": 160,
          "springConstant": 0.06,
          "damping": 0.4,
          "avoidOverlap": 0.8
        },
        "solver": "forceAtlas2Based",
        "stabilization": { "iterations": 150 }
      },
      "edges": {
        "arrows": { "to": { "enabled": true, "scaleFactor": 0.6 } },
        "color": { "color": "#555577", "highlight": "#aaaaff" },
        "font": { "size": 10, "color": "#aaaaaa", "strokeWidth": 0 },
        "smooth": { "type": "curvedCW", "roundness": 0.15 }
      },
      "nodes": {
        "font": { "size": 12, "bold": true },
        "borderWidth": 2,
        "shadow": true
      },
      "interaction": {
        "hover": true,
        "tooltipDelay": 100,
        "navigationButtons": true
      }
    }
    """)

    node_set = {e["source"] for e in graph_data.get("edges", [])} | \
               {e["target"] for e in graph_data.get("edges", [])}

    main_node = graph_data.get("main_node", "")
    for node in node_set:
        colour = _NODE_COLOURS[hash(node) % len(_NODE_COLOURS)]
        size = 30 if node == main_node else 22
        net.add_node(
            node,
            label=node,
            color={"background": colour, "border": "#ffffff33",
                   "highlight": {"background": colour, "border": "#ffffff"}},
            size=size,
            title=node,
        )

    for edge in graph_data.get("edges", []):
        net.add_edge(
            edge["source"],
            edge["target"],
            label=edge["relation"],
            title=edge["relation"],
        )

    return net.generate_html()


def _filter_graph(graph_data: dict, selected_nodes: list) -> dict:
    if not selected_nodes:
        return graph_data
    selected = set(selected_nodes)
    filtered_edges = [
        e for e in graph_data.get("edges", [])
        if e.get("source") in selected and e.get("target") in selected
    ]
    nodes_in_edges = sorted(
        {e["source"] for e in filtered_edges} | {e["target"] for e in filtered_edges}
    )
    return {"nodes": nodes_in_edges, "edges": filtered_edges}


def _edges_to_csv(edges: list) -> str:
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=["source", "relation", "target"])
    writer.writeheader()
    for e in edges:
        writer.writerow({
            "source": e.get("source", ""),
            "relation": e.get("relation", ""),
            "target": e.get("target", ""),
        })
    return buffer.getvalue()


# ── Load once, cache forever ──────────────────────────────────────────────────

@st.cache_resource
def get_vectorstore():
    return load_vectorstore()

@st.cache_resource
def get_bm25_index():
    return build_bm25_index()

vectorstore = get_vectorstore()
bm25_index  = get_bm25_index()
neo4j_driver = get_neo4j_driver()

# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.header("🔧 Settings")
    institution = st.selectbox(
        "Filter by institution",
        options=["All", "JPM", "GS", "BAC", "FED", "RBI"],
        index=0,
    )
    k = st.slider("Chunks to retrieve", min_value=3, max_value=10, value=6)
    st.markdown("---")
    st.markdown("**Sources loaded:**")
    st.markdown("- 🏦 JPM 10-K\n- 🏦 GS 10-K\n- 🏦 BAC 10-K\n- 🏛️ Fed Statement\n- 🇮🇳 RBI Reports")

    st.markdown("---")
    if neo4j_driver:
        st.success("✅ Neo4j Aura connected")
    else:
        st.warning("⚠️ Neo4j not connected\nSet NEO4J_URI / NEO4J_USERNAME / NEO4J_PASSWORD")

tab_chat, tab_graph = st.tabs(["💬 Q&A", "🕸️ Knowledge Graph"])

# ── Chat tab ──────────────────────────────────────────────────────────────────

with tab_chat:
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if question := st.chat_input("Ask a financial question..."):
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            with st.spinner("Retrieving and reasoning..."):
                try:
                    filter_val = None if institution == "All" else institution
                    result = _run_async(async_answer_with_cache(
                        question=question,
                        institution_filter=filter_val,
                        vectorstore=vectorstore,
                        bm25_index=bm25_index,
                        k=k,
                    ))
                    answer = result["answer"]
                    st.markdown(answer)
                    if result.get("cache_hit"):
                        st.caption("⚡ Served from in-memory query/context cache.")
                    st.session_state.messages.append({"role": "assistant", "content": answer})
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")

    if not st.session_state.messages:
        st.markdown("### 💡 Try asking:")
        cols = st.columns(2)
        examples = [
            ("What did the Fed decide on rates?", "FED"),
            ("What is Goldman Sachs revenue?",    "GS"),
            ("What are BAC's risk factors?",      "BAC"),
            ("What is JPMorgan's total assets?",  "JPM"),
        ]
        for i, (q, inst) in enumerate(examples):
            with cols[i % 2]:
                st.code(f"{q}\n[Filter: {inst}]", language=None)

# ── Graph tab ─────────────────────────────────────────────────────────────────

with tab_graph:
    st.subheader("Knowledge Graph — Neo4j + Interactive Visualiser")
    st.caption(
        "Upload a PDF/TXT to extract a finance-specific knowledge graph. "
        "The graph is stored in Neo4j Aura and rendered interactively."
    )

    uploaded   = st.file_uploader("Upload file", type=["pdf", "txt"], key="kg_upload")
    use_llm    = st.toggle("Use LLM extraction (recommended)", value=True)

    if "uploaded_graph" not in st.session_state:
        st.session_state.uploaded_graph  = {"nodes": [], "edges": []}
        st.session_state.graph_source    = ""

    # ── Generate ──────────────────────────────────────────────────────────────
    if uploaded is not None and st.button("Generate Knowledge Graph"):
        with st.spinner("Extracting entities and relations..."):
            text = _read_uploaded_text(uploaded)
            if not text.strip():
                st.error("Could not extract text from the uploaded file.")
            else:
                if use_llm:
                    graph_data = _extract_graph_with_llm(text)

                # Fall back to heuristic if LLM returned nothing
                if not use_llm or not graph_data.get("edges"):
                    graph_data = _extract_graph_heuristic(text)

                # Final cleanup regardless of extraction mode
                graph_data = _sanitize_graph(graph_data)

                source_label = re.sub(r"[^A-Za-z0-9_]", "_", uploaded.name)
                st.session_state.uploaded_graph = graph_data
                st.session_state.graph_source   = source_label

                # Push to Neo4j
                if neo4j_driver:
                    ok = _push_graph_to_neo4j(neo4j_driver, graph_data, source_label)
                    if ok:
                        st.success(f"✅ Graph stored in Neo4j Aura ({len(graph_data['nodes'])} nodes, {len(graph_data['edges'])} edges)")
                    else:
                        st.warning("Graph extracted but could not be saved to Neo4j.")
                else:
                    st.info("Neo4j not connected — graph will not be persisted.")

    # ── Load from Neo4j if session is empty ───────────────────────────────────
    if not st.session_state.uploaded_graph.get("edges") and neo4j_driver:
        with st.expander("📂 Load a previously stored graph from Neo4j"):
            source_to_load = st.text_input("Enter the source label (filename without extension):")
            if st.button("Load from Neo4j") and source_to_load:
                loaded = _load_graph_from_neo4j(neo4j_driver, source_to_load)
                if loaded.get("edges"):
                    st.session_state.uploaded_graph = loaded
                    st.session_state.graph_source   = source_to_load
                    st.success(f"Loaded {len(loaded['nodes'])} nodes, {len(loaded['edges'])} edges.")
                else:
                    st.warning("No graph found for that source label.")

    graph_data = st.session_state.uploaded_graph

    # ── Render ────────────────────────────────────────────────────────────────
    if graph_data.get("edges"):
        st.markdown("---")

        # Node filter
        st.markdown("### 🔍 Filter Nodes")
        graph_filter_nodes = st.multiselect(
            "Select nodes to focus on (leave empty for full graph)",
            options=graph_data.get("nodes", []),
            default=[],
            help="Only edges between selected nodes will be shown.",
        )

        filtered_graph = _filter_graph(graph_data, graph_filter_nodes) if graph_filter_nodes else graph_data
        if graph_filter_nodes and not filtered_graph["edges"]:
            st.info("No edges between selected nodes — showing full graph.")
            filtered_graph = graph_data

        # Interactive pyvis graph
        st.markdown("### 🕸️ Interactive Graph")
        st.caption("Drag nodes • Scroll to zoom • Hover for labels • Click to highlight")
        html_content = _render_pyvis(filtered_graph, height=620)
        components.html(html_content, height=640, scrolling=False)

        # Stats
        st.caption(
            f"Nodes: **{len(filtered_graph['nodes'])}** | "
            f"Edges: **{len(filtered_graph['edges'])}** | "
            f"Source: `{st.session_state.graph_source or 'in-memory'}`"
        )
        if graph_data.get("main_node"):
            st.caption(f"Main node detected: **{graph_data['main_node']}**")

        # Downloads
        st.markdown("### ⬇️ Download")
        col_json, col_csv = st.columns(2)
        with col_json:
            st.download_button(
                label="Download JSON",
                data=json.dumps(graph_data, indent=2),
                file_name="knowledge_graph.json",
                mime="application/json",
            )
        with col_csv:
            st.download_button(
                label="Download CSV (edges)",
                data=_edges_to_csv(graph_data.get("edges", [])),
                file_name="knowledge_graph_edges.csv",
                mime="text/csv",
            )

        # Triples table
        with st.expander("📋 View extracted triples"):
            st.dataframe(filtered_graph["edges"], use_container_width=True)

        # RAG linkage
        st.markdown("---")
        st.markdown("### 🔗 Link Graph Node to RAG Corpus")
        node_choice   = st.selectbox("Select a graph node", options=graph_data.get("nodes", []))
        node_question = st.text_input(
            "Ask a node-specific question",
            value=f"What does the corpus say about {node_choice}?",
        )

        if st.button("Retrieve linked chunks and answer"):
            with st.spinner("Retrieving linked RAG chunks..."):
                docs = _run_async(async_retrieve_documents(
                    question=node_choice,
                    institution_filter=None,
                    vectorstore=vectorstore,
                    bm25_index=bm25_index,
                    k=5,
                ))
                if not docs:
                    st.warning("No linked chunks found for this node.")
                else:
                    for idx, doc in enumerate(docs, start=1):
                        citation = doc.metadata.get("citation_label", "unknown source")
                        st.markdown(f"**Chunk {idx}:** `{citation}`")
                        st.write(doc.page_content[:320] + ("..." if len(doc.page_content) > 320 else ""))

            with st.spinner("Generating node-linked answer..."):
                answer_result = _run_async(async_answer_with_cache(
                    question=node_question,
                    institution_filter=None,
                    vectorstore=vectorstore,
                    bm25_index=bm25_index,
                    k=6,
                ))
                st.markdown("#### Node-Linked Answer")
                st.write(answer_result["answer"])
                if answer_result.get("cache_hit"):
                    st.caption("⚡ Served from in-memory query/context cache.")

    elif uploaded is not None:
        st.warning("No relationships detected. Try a different file or enable LLM extraction.")