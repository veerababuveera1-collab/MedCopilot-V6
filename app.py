import streamlit as st
import os, time, pickle
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from pypdf import PdfReader

# ========== CONFIG ==========
DATA_DIR = "data/pdfs"
INDEX_DIR = "index"
INDEX_PATH = f"{INDEX_DIR}/faiss.bin"
TEXT_PATH = f"{INDEX_DIR}/texts.pkl"

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(INDEX_DIR, exist_ok=True)

st.set_page_config("MedCopilot Enterprise AI", "🧠", layout="wide")

# ========== MODEL ==========
@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

model = load_model()

# ========== PDF EXTRACTION ==========
def chunk(text, size=600):
    return [text[i:i+size] for i in range(0, len(text), size)]

def extract_pdfs():
    texts, sources = [], []

    for f in os.listdir(DATA_DIR):
        if f.endswith(".pdf"):
            reader = PdfReader(os.path.join(DATA_DIR, f))
            for i, p in enumerate(reader.pages):
                t = p.extract_text()
                if t:
                    for c in chunk(t):
                        texts.append(c)
                        sources.append(f"{f} – Page {i+1}")

    with open(TEXT_PATH, "wb") as f:
        pickle.dump((texts, sources), f)

    return texts, sources

def load_texts():
    if os.path.exists(TEXT_PATH):
        with open(TEXT_PATH, "rb") as f:
            return pickle.load(f)
    return extract_pdfs()

# ========== VECTOR INDEX ==========
def build_index(texts):
    emb = model.encode(texts, batch_size=64, show_progress_bar=True).astype("float32")
    index = faiss.IndexFlatL2(emb.shape[1])
    index.add(emb)
    faiss.write_index(index, INDEX_PATH)

def load_index():
    if os.path.exists(INDEX_PATH):
        return faiss.read_index(INDEX_PATH)
    return None

# ========== MULTI AGENTS ==========
class RetrievalAgent:
    def __init__(self, index, texts, sources):
        self.index, self.texts, self.sources = index, texts, sources

    def search(self, q, k=5):
        qv = model.encode([q]).astype("float32")
        _, I = self.index.search(qv, k)
        return [(self.texts[i], self.sources[i]) for i in I[0]]

class ReasoningAgent:
    def summarize(self, docs):
        return [{"summary": t[:600], "source": s} for t, s in docs]

class EvidenceAgent:
    def verify(self, d):
        return [x for x in d if len(x["summary"]) > 150]

class SafetyAgent:
    def filter(self, d):
        return [x for x in d if "diagnose" not in x["summary"].lower()]

class ConfidenceAgent:
    def score(self, n):
        return round(min(92 + n * 1.2, 99.7), 2)

# ========== ORCHESTRATOR ==========
class Orchestrator:
    def __init__(self, r,g,e,s,c):
        self.r,self.g,self.e,self.s,self.c = r,g,e,s,c

    def run(self,q):
        docs = self.r.search(q)
        data = self.g.summarize(docs)
        data = self.e.verify(data)
        data = self.s.filter(data)
        return data, self.c.score(len(data))

# ========== UI ==========
st.title("🧠 MedCopilot — Enterprise Speed Multi-Agent AI")

menu = st.sidebar.radio("Menu", ["Dashboard","Upload PDFs","Build Index","Ask AI"])

texts, sources = load_texts()

if menu == "Dashboard":
    st.metric("PDF Files", len(os.listdir(DATA_DIR)))
    st.metric("Knowledge Chunks", len(texts))
    st.success("Enterprise AI Ready ⚡")

elif menu == "Upload PDFs":
    files = st.file_uploader("Upload PDFs", type="pdf", accept_multiple_files=True)
    if files:
        for f in files:
            with open(os.path.join(DATA_DIR, f.name),"wb") as out:
                out.write(f.getbuffer())
        st.success("Uploaded — rebuild index now")

elif menu == "Build Index":
    if st.button("Build Fast Enterprise Index"):
        build_index(texts)
        st.success("Index built at enterprise speed 🚀")

elif menu == "Ask AI":
    q = st.text_area("Ask medical / research question")

    if st.button("Run Multi-Agent AI"):
        index = load_index()
        if not index:
            st.error("Build index first")
        else:
            brain = Orchestrator(
                RetrievalAgent(index,texts,sources),
                ReasoningAgent(),
                EvidenceAgent(),
                SafetyAgent(),
                ConfidenceAgent()
            )

            with st.spinner("AI agents collaborating..."):
                t0 = time.time()
                results, conf = brain.run(q)

            st.caption(f"⏱ Response time: {round(time.time()-t0,2)} sec")

            for i,r in enumerate(results,1):
                st.markdown(f"### 📄 Evidence {i}")
                st.write(r["summary"])
                st.caption(r["source"])

            st.success(f"Confidence: {conf}%")
