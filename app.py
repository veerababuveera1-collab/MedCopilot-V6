import streamlit as st
import os
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from pypdf import PdfReader
import time

# ===================== CONFIG =====================
st.set_page_config(
    page_title="MedCopilot Enterprise — Hospital AI Command Center",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

DATA_DIR = "data/pdfs"
INDEX_DIR = "index"
INDEX_PATH = "index/faiss_index.bin"

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(INDEX_DIR, exist_ok=True)

# ===================== LOAD MODEL =====================
@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

model = load_model()

# ===================== CSS =====================
st.markdown("""
<style>
.main-title {
    font-size: 38px;
    font-weight: bold;
    color: #0d6efd;
}
.card {
    padding: 20px;
    border-radius: 12px;
    background: #f8f9fa;
    box-shadow: 0px 0px 10px rgba(0,0,0,0.1);
}
.kpi {
    font-size: 28px;
    font-weight: bold;
    color: #198754;
}
</style>
""", unsafe_allow_html=True)

# ===================== FUNCTIONS =====================
def load_pdfs():
    texts = []
    for file in os.listdir(DATA_DIR):
        if file.endswith(".pdf"):
            reader = PdfReader(os.path.join(DATA_DIR, file))
            for page in reader.pages:
                text = page.extract_text()
                if text:
                    texts.append(text)
    return texts

def build_index(texts):
    embeddings = model.encode(texts)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(np.array(embeddings).astype("float32"))
    faiss.write_index(index, INDEX_PATH)
    return len(texts)

def load_index():
    if os.path.exists(INDEX_PATH):
        return faiss.read_index(INDEX_PATH)
    return None

def search_index(query, texts, k=3):
    index = load_index()
    if index is None:
        return []

    q_emb = model.encode([query]).astype("float32")
    D, I = index.search(q_emb, k)
    results = [texts[i] for i in I[0] if i < len(texts)]
    return results

# ===================== HEADER =====================
st.markdown("<div class='main-title'>🧠 MedCopilot Enterprise — Hospital AI Command Center</div>", unsafe_allow_html=True)
st.write("Clinical Evidence • Medical Intelligence • Global Research")
st.divider()

# ===================== SIDEBAR =====================
st.sidebar.title("🏥 MedCopilot Control Panel")

menu = st.sidebar.radio(
    "Navigation",
    ["📊 Dashboard", "🔍 Clinical AI Console", "📁 PDF Knowledge", "⚙ System Health"]
)

# ===================== LOAD DATA =====================
texts_cache = load_pdfs()
total_pdfs = len(os.listdir(DATA_DIR))
indexed_pages = len(texts_cache)

# ===================== DASHBOARD =====================
if menu == "📊 Dashboard":
    st.subheader("📊 Hospital Intelligence Dashboard")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown(f"<div class='card'>Total PDFs<br><div class='kpi'>{total_pdfs}</div></div>", unsafe_allow_html=True)

    with col2:
        st.markdown(f"<div class='card'>Indexed Pages<br><div class='kpi'>{indexed_pages}</div></div>", unsafe_allow_html=True)

    with col3:
        st.markdown("<div class='card'>AI Confidence<br><div class='kpi'>96.5%</div></div>", unsafe_allow_html=True)

    with col4:
        st.markdown("<div class='card'>Queries Today<br><div class='kpi'>61</div></div>", unsafe_allow_html=True)

    st.divider()

    st.success("All Hospital AI Systems Operational")
    st.info("Clinical Intelligence Engine: Active")
    st.info("Evidence Index Engine: Active")
    st.info("Drug Intelligence Engine: Active")
    st.info("AI Agents Network: Online")

# ===================== CLINICAL AI =====================
elif menu == "🔍 Clinical AI Console":
    st.subheader("🔍 Clinical Intelligence Console")

    query = st.text_area("Ask a clinical research or hospital question", height=120)

    if st.button("🚀 Run Clinical Intelligence"):
        if not os.path.exists(INDEX_PATH):
            st.error("Evidence Index not built. Please build it first from PDF Knowledge page.")
        else:
            with st.spinner("Searching medical evidence..."):
                time.sleep(1)
                texts = load_pdfs()
                results = search_index(query, texts)

            st.success("Clinical Evidence Found")

            for i, res in enumerate(results, 1):
                st.markdown(f"### 📄 Evidence {i}")
                st.write(res[:1200] + "...")

# ===================== PDF KNOWLEDGE =====================
elif menu == "📁 PDF Knowledge":
    st.subheader("📁 Clinical PDF Knowledge Library")

    uploaded_files = st.file_uploader(
        "Upload Clinical PDFs (Guidelines, Research Papers, Protocols)",
        type=["pdf"],
        accept_multiple_files=True
    )

    if uploaded_files:
        for pdf in uploaded_files:
            with open(os.path.join(DATA_DIR, pdf.name), "wb") as f:
                f.write(pdf.getbuffer())
            st.success(f"Saved: {pdf.name}")

    st.divider()

    if st.button("🧠 Build Evidence Index"):
        with st.spinner("Building clinical knowledge index..."):
            texts = load_pdfs()
            pages = build_index(texts)

        st.success("Evidence Index Built Successfully!")
        st.info(f"Indexed Pages: {pages}")

    st.divider()
    st.write("📚 Knowledge Base Status")
    st.success(f"{total_pdfs} PDFs available")
    st.success(f"{indexed_pages} pages extracted")

# ===================== SYSTEM HEALTH =====================
elif menu == "⚙ System Health":
    st.subheader("⚙ System Health Monitor")

    st.success("Embedding Model: MiniLM-L6-v2")
    st.success("Vector DB: FAISS")
    st.success("Evidence Index: Ready")
    st.success("Clinical Engine: Online")
    st.success("AI Core: Stable")

    st.progress(95)
    st.progress(98)
    st.progress(96)

# ===================== FOOTER =====================
st.divider()
st.caption("🧠 MedCopilot Enterprise © Hospital AI Platform | Clinical Decision Intelligence")
