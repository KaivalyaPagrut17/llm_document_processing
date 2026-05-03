"""
InsureQ — Streamlit UI for LLM Document Processing System
Run: streamlit run app.py  (requires Phase 4 API at localhost:8000)
"""
import streamlit as st
import requests

API_BASE = "http://localhost:8000"

st.set_page_config(page_title="InsureQ", page_icon="🛡️", layout="wide", initial_sidebar_state="expanded")

# ── CSS ──────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.stApp { background: #0b0f1a; }
section[data-testid="stSidebar"] {
    background: #0d1117;
    border-right: 1px solid #1e2535;
}
section[data-testid="stSidebar"] * { color: #c9d1d9; }

/* ── Brand header ── */
.brand { display:flex; align-items:center; gap:10px; padding:4px 0 16px; }
.brand-icon { font-size:28px; }
.brand-name { font-size:20px; font-weight:700; color:#e6edf3; letter-spacing:-0.3px; }
.brand-sub  { font-size:11px; color:#6e7681; margin-top:1px; }

/* ── Status pill ── */
.status-pill {
    display:inline-flex; align-items:center; gap:6px;
    padding:5px 12px; border-radius:20px; font-size:12px; font-weight:600;
}
.status-online  { background:#0d2818; color:#3fb950; border:1px solid #238636; }
.status-offline { background:#2d1117; color:#f85149; border:1px solid #6e2a2a; }

/* ── Sidebar metric ── */
.sb-metric {
    background:#161b22; border:1px solid #21262d; border-radius:10px;
    padding:14px 16px; text-align:center; margin:8px 0;
}
.sb-metric-val { font-size:26px; font-weight:700; color:#58a6ff; }
.sb-metric-lbl { font-size:11px; color:#6e7681; margin-top:2px; }

/* ── Page title ── */
.page-title { font-size:22px; font-weight:700; color:#e6edf3; margin-bottom:2px; }
.page-sub   { font-size:13px; color:#6e7681; margin-bottom:20px; }

/* ── Chat bubbles ── */
.msg-row-user { display:flex; justify-content:flex-end; margin:10px 0; }
.msg-row-ai   { display:flex; justify-content:flex-start; margin:10px 0; }
.bubble-user {
    background: linear-gradient(135deg, #1f6feb, #388bfd);
    color:#fff; border-radius:18px 18px 4px 18px;
    padding:12px 18px; max-width:75%; font-size:14px; line-height:1.6;
    box-shadow: 0 2px 8px rgba(31,111,235,0.3);
}
.bubble-ai {
    background:#161b22; border:1px solid #21262d;
    color:#c9d1d9; border-radius:18px 18px 18px 4px;
    padding:14px 18px; max-width:80%; font-size:14px; line-height:1.7;
    box-shadow: 0 2px 8px rgba(0,0,0,0.3);
}
.intent-tag {
    display:inline-block; background:#0d2818; color:#3fb950;
    border:1px solid #238636; border-radius:20px;
    padding:2px 10px; font-size:11px; font-weight:600;
    text-transform:uppercase; letter-spacing:0.6px; margin-bottom:8px;
}
.latency-tag { font-size:11px; color:#484f58; margin-top:8px; text-align:right; }

/* ── Source card ── */
.source-card {
    background:#0d1117; border:1px solid #21262d;
    border-left:3px solid #388bfd; border-radius:8px;
    padding:10px 14px; margin:6px 0; font-size:13px; color:#8b949e;
}
.source-card b { color:#58a6ff; }

/* ── Search result ── */
.result-card {
    background:#161b22; border:1px solid #21262d; border-radius:12px;
    padding:16px; margin:10px 0;
}
.result-header { display:flex; justify-content:space-between; align-items:center; margin-bottom:8px; }
.result-title  { font-size:14px; font-weight:600; color:#58a6ff; }
.result-meta   { font-size:11px; color:#484f58; }
.score-track   { background:#21262d; border-radius:4px; height:5px; margin:6px 0 10px; }
.score-fill    { height:5px; border-radius:4px; background:linear-gradient(90deg,#1f6feb,#3fb950); }
.result-text   { font-size:13px; color:#8b949e; line-height:1.6; }

/* ── Upload / reindex cards ── */
.action-card {
    background:#161b22; border:1px solid #21262d; border-radius:12px; padding:20px;
}
.action-title { font-size:15px; font-weight:600; color:#e6edf3; margin-bottom:6px; }
.action-sub   { font-size:12px; color:#6e7681; margin-bottom:14px; }

/* ── Stats row ── */
.stat-box {
    background:#161b22; border:1px solid #21262d; border-radius:10px;
    padding:16px; text-align:center;
}
.stat-val { font-size:22px; font-weight:700; color:#58a6ff; }
.stat-lbl { font-size:11px; color:#6e7681; margin-top:3px; }

/* ── Inputs ── */
.stTextInput input, .stTextArea textarea {
    background:#161b22 !important; color:#c9d1d9 !important;
    border:1px solid #30363d !important; border-radius:8px !important;
}
.stTextInput input:focus, .stTextArea textarea:focus {
    border-color:#388bfd !important; box-shadow:0 0 0 3px rgba(56,139,253,0.15) !important;
}
div[data-testid="stChatInput"] textarea {
    background:#161b22 !important; color:#c9d1d9 !important;
    border:1px solid #30363d !important; border-radius:12px !important;
}
/* Buttons */
.stButton > button {
    border-radius:8px !important; font-weight:600 !important; font-size:13px !important;
    transition: all .15s ease !important;
}
.stButton > button[kind="primary"] {
    background:linear-gradient(135deg,#1f6feb,#388bfd) !important;
    border:none !important; color:#fff !important;
}
.stButton > button[kind="primary"]:hover { opacity:.9 !important; }
/* Divider */
hr { border-color:#21262d !important; }
/* Expander */
details { background:#161b22 !important; border:1px solid #21262d !important; border-radius:8px !important; }
summary { color:#8b949e !important; font-size:13px !important; }
</style>
""", unsafe_allow_html=True)

# ── API helpers ───────────────────────────────────────────────────────────────
def _get(path, timeout=3):
    try:
        r = requests.get(f"{API_BASE}{path}", timeout=timeout)
        return r.json() if r.ok else None
    except Exception:
        return None

def _post(path, **kwargs):
    r = requests.post(f"{API_BASE}{path}", timeout=kwargs.pop("timeout", 60), **kwargs)
    r.raise_for_status()
    return r.json()

health = _get("/health")
stats  = _get("/stats")

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div class="brand">
        <div class="brand-icon">🛡️</div>
        <div><div class="brand-name">InsureQ</div>
             <div class="brand-sub">Insurance Document AI</div></div>
    </div>""", unsafe_allow_html=True)

    if health:
        model_name = health.get("model", "?").split("/")[-1]
        emb_name   = health.get("embedding_model", "?").split("/")[-1]
        st.markdown('<span class="status-pill status-online">● API Online</span>', unsafe_allow_html=True)
        st.markdown(f"<div style='font-size:11px;color:#6e7681;margin-top:6px'>Model: {model_name}<br>Embeddings: {emb_name}</div>", unsafe_allow_html=True)
    else:
        st.markdown('<span class="status-pill status-offline">● API Offline</span>', unsafe_allow_html=True)
        st.code("python src/phase4_api.py", language="bash")

    st.markdown("<hr>", unsafe_allow_html=True)

    if stats:
        st.markdown(f"""
        <div class="sb-metric">
            <div class="sb-metric-val">{stats.get('total_chunks', 0):,}</div>
            <div class="sb-metric-lbl">Indexed Chunks</div>
        </div>""", unsafe_allow_html=True)

    top_k = st.slider("Results (Top K)", 3, 15, 7)
    st.markdown("<hr>", unsafe_allow_html=True)
    page = st.radio("", ["💬 Chat Q&A", "🔍 Semantic Search", "📁 Documents"], label_visibility="collapsed")
    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown("<div style='font-size:11px;color:#484f58;text-align:center'>Phase 1–4 Complete</div>", unsafe_allow_html=True)

# ── Chat Q&A ──────────────────────────────────────────────────────────────────
if page == "💬 Chat Q&A":
    st.markdown('<div class="page-title">💬 Document Q&A</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-sub">Ask anything about your insurance documents — powered by Qwen2.5-72B.</div>', unsafe_allow_html=True)

    if "messages" not in st.session_state:
        st.session_state.messages = []

    chat_container = st.container()
    with chat_container:
        for msg in st.session_state.messages:
            if msg["role"] == "user":
                st.markdown(f'<div class="msg-row-user"><div class="bubble-user">{msg["content"]}</div></div>', unsafe_allow_html=True)
            else:
                intent  = msg.get("intent", "")
                latency = msg.get("latency_ms", "")
                st.markdown(f"""
                <div class="msg-row-ai"><div class="bubble-ai">
                    <span class="intent-tag">⚡ {intent}</span><br>
                    {msg["content"]}
                    <div class="latency-tag">⏱ {latency} ms</div>
                </div></div>""", unsafe_allow_html=True)

                sources = msg.get("sources", [])
                if sources:
                    with st.expander(f"📚 {len(sources)} source{'s' if len(sources)>1 else ''}"):
                        for s in sources:
                            meta = " · ".join(filter(None, [
                                f"Page {s['page']}" if s.get("page") else "",
                                s.get("section", "")
                            ]))
                            st.markdown(f"""
                            <div class="source-card">
                                <b>{s['document']}</b>
                                {"<br><span style='font-size:11px'>" + meta + "</span>" if meta else ""}
                                <br><span style='font-size:11px'>Score: {s['similarity_score']}</span>
                            </div>""", unsafe_allow_html=True)

    user_input = st.chat_input("Ask a question about your documents…")

    if user_input:
        if not health:
            st.error("API is offline. Start it with `python src/phase4_api.py`.")
        else:
            st.session_state.messages.append({"role": "user", "content": user_input})
            with st.spinner("Thinking…"):
                try:
                    res = _post("/query", json={"query": user_input, "top_k": top_k})
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": res["answer"],
                        "intent": res["intent"],
                        "sources": res["sources"],
                        "latency_ms": res["latency_ms"],
                    })
                except Exception as e:
                    st.error(f"Query failed: {e}")
            st.rerun()

    if st.session_state.messages:
        if st.button("🗑 Clear conversation"):
            st.session_state.messages = []
            st.rerun()

# ── Semantic Search ───────────────────────────────────────────────────────────
elif page == "🔍 Semantic Search":
    st.markdown('<div class="page-title">🔍 Semantic Search</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-sub">Search document chunks directly — no LLM, just vector similarity.</div>', unsafe_allow_html=True)

    query = st.text_input("", placeholder="e.g. waiting period for pre-existing diseases", label_visibility="collapsed")

    if st.button("Search", type="primary", use_container_width=True) and query:
        if not health:
            st.error("API is offline.")
        else:
            with st.spinner("Searching…"):
                try:
                    res     = _post("/search", json={"query": query, "top_k": top_k}, timeout=30)
                    results = res["results"]
                    st.markdown(f"<div style='font-size:13px;color:#6e7681;margin:8px 0'>{len(results)} results · {res['latency_ms']} ms</div>", unsafe_allow_html=True)
                    for r in results:
                        pct  = int(r["similarity_score"] * 100)
                        meta = f"Page {r['page']} · " if r.get("page") else ""
                        st.markdown(f"""
                        <div class="result-card">
                            <div class="result-header">
                                <span class="result-title">#{r['rank']} — {r['file_name']}</span>
                                <span class="result-meta">{meta}Chunk {r['chunk_id']}</span>
                            </div>
                            <div class="score-track"><div class="score-fill" style="width:{pct}%"></div></div>
                            <div style="font-size:11px;color:#484f58;margin-bottom:8px">Similarity: {r['similarity_score']}</div>
                            <div class="result-text">{r['text']}</div>
                        </div>""", unsafe_allow_html=True)
                except Exception as e:
                    st.error(f"Search failed: {e}")

# ── Documents ─────────────────────────────────────────────────────────────────
elif page == "📁 Documents":
    st.markdown('<div class="page-title">📁 Document Management</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-sub">Upload documents and manage the vector index.</div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2, gap="large")

    with col1:
        st.markdown('<div class="action-card">', unsafe_allow_html=True)
        st.markdown('<div class="action-title">📤 Upload Document</div>', unsafe_allow_html=True)
        st.markdown('<div class="action-sub">Supported formats: PDF, DOCX. File is processed and indexed automatically.</div>', unsafe_allow_html=True)
        uploaded = st.file_uploader("", type=["pdf", "docx"], label_visibility="collapsed")
        if uploaded and st.button("Upload & Process", type="primary", use_container_width=True):
            if not health:
                st.error("API is offline.")
            else:
                with st.spinner(f"Processing {uploaded.name}…"):
                    try:
                        res = _post("/upload", files={"file": (uploaded.name, uploaded.read())}, timeout=120)
                        st.success(f"✅ **{res['filename']}** — {res['chunks_created']} chunks created")
                    except Exception as e:
                        st.error(f"Upload failed: {e}")
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="action-card">', unsafe_allow_html=True)
        st.markdown('<div class="action-title">🔄 Rebuild Index</div>', unsafe_allow_html=True)
        st.markdown('<div class="action-sub">Rebuilds the ChromaDB vector index from all processed documents. Run after uploading new files.</div>', unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("🔄 Reindex All Documents", use_container_width=True):
            if not health:
                st.error("API is offline.")
            else:
                with st.spinner("Rebuilding index… this may take a few minutes."):
                    try:
                        res = _post("/reindex", timeout=300)
                        st.success(f"✅ {res['message']}")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Reindex failed: {e}")
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown('<div class="action-title">📊 Index Stats</div>', unsafe_allow_html=True)

    if stats:
        c1, c2, c3 = st.columns(3)
        for col, val, lbl in [
            (c1, f"{stats.get('total_chunks', 0):,}", "Total Chunks"),
            (c2, stats.get("collection_name", "—"), "Collection"),
            (c3, stats.get("similarity_metric", "—"), "Similarity Metric"),
        ]:
            col.markdown(f'<div class="stat-box"><div class="stat-val">{val}</div><div class="stat-lbl">{lbl}</div></div>', unsafe_allow_html=True)
        st.markdown(f"<div style='font-size:12px;color:#6e7681;margin-top:12px'>Embedding model: <code>{stats.get('embedding_model','?')}</code></div>", unsafe_allow_html=True)
    else:
        st.warning("Could not fetch stats — is the API running?")
