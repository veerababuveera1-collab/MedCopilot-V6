import streamlit as st
import faiss
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer

VECTOR_DIR = "research_ai/vector_trials"
INDEX_FILE = f"{VECTOR_DIR}/trials.index"
CACHE_FILE = f"{VECTOR_DIR}/trials.pkl"

st.set_page_config(page_title="MedCopilot Clinical Trials AI", page_icon="🧪", layout="wide")

st.title("🧪 MedCopilot Clinical Trials Intelligence")
st.caption("Clinical Trials • Research Evidence • Drug Development")

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
    st.sidebar.success("🟢 Clinical Trials Knowledge Base Loaded")
except:
    st.sidebar.error("❌ Clinical trials index not found. Run trials_indexer.py")
    st.stop()


query = st.text_input("Search clinical trials (condition, drug, phase, outcome)")

if st.button("🚀 Run Trials Intelligence") and query:
    q_emb = embedder.encode([query])
    D, I = index.search(np.array(q_emb), 5)

    st.subheader("🧪 Clinical Trials Evidence")

    for i in I[0]:
        st.markdown(documents[i])
        st.info(sources[i])

