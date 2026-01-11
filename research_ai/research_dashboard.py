import streamlit as st
import faiss
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer

VECTOR_DIR = "research_ai/vector_db"
INDEX_FILE = f"{VECTOR_DIR}/pubmed.index"
CACHE_FILE = f"{VECTOR_DIR}/pubmed.pkl"

st.set_page_config(page_title="MedCopilot Research AI", page_icon="🔬", layout="wide")

st.title("🔬 MedCopilot Research AI — PubMed Intelligence")
st.caption("Medical Literature • Research Intelligence • Evidence Copilot")

@st.cache_resource
def load_embedder():
    return SentenceTransformer("all-MiniLM-L6-v2")

embedder = load_embedder()


def load_index():
    index = faiss.read_index(INDEX_FILE)
    with open(CACHE_FILE, "rb") as f:
        data = pickle.load(f)
    return index, data["documents"], data["sources"]


try:
    index, documents, sources = load_index()
    st.sidebar.success("🟢 Research Knowledge Base Loaded")
except:
    st.sidebar.error("❌ Research index not found. Run research_indexer.py")
    st.stop()


query = st.text_input("Ask a medical research question (PubMed)")

if st.button("🚀 Run Research Intelligence") and query:
    q_emb = embedder.encode([query])
    D, I = index.search(np.array(q_emb), 5)

    st.subheader("📚 Research Evidence")

    for i in I[0]:
        st.markdown(documents[i][:1200])
        st.info(sources[i])
