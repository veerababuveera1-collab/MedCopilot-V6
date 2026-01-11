import os
import pickle
import faiss
import numpy as np
import xml.etree.ElementTree as ET
from sentence_transformers import SentenceTransformer

DATA_DIR = "research_ai/data/pubmed"
VECTOR_DIR = "research_ai/vector_db"
INDEX_FILE = os.path.join(VECTOR_DIR, "pubmed.index")
CACHE_FILE = os.path.join(VECTOR_DIR, "pubmed.pkl")

os.makedirs(VECTOR_DIR, exist_ok=True)

model = SentenceTransformer("all-MiniLM-L6-v2")


def parse_pubmed_xml(file_path):
    tree = ET.parse(file_path)
    root = tree.getroot()

    docs = []
    sources = []

    for article in root.findall(".//PubmedArticle"):
        title = article.findtext(".//ArticleTitle", default="")
        abstract = article.findtext(".//AbstractText", default="")
        pmid = article.findtext(".//PMID", default="")

        text = f"{title}\n\n{abstract}"
        if len(text.strip()) > 200:
            docs.append(text)
            sources.append(f"PMID: {pmid}")

    return docs, sources


def build_index():
    documents = []
    sources = []

    for file in os.listdir(DATA_DIR):
        if file.endswith(".xml"):
            d, s = parse_pubmed_xml(os.path.join(DATA_DIR, file))
            documents.extend(d)
            sources.extend(s)

    if not documents:
        print("❌ No research papers found.")
        return

    print(f"🧠 Indexing {len(documents)} research papers...")

    embeddings = model.encode(documents, batch_size=16)
    dim = embeddings.shape[1]

    index = faiss.IndexFlatL2(dim)
    index.add(np.array(embeddings))

    faiss.write_index(index, INDEX_FILE)

    with open(CACHE_FILE, "wb") as f:
        pickle.dump({"documents": documents, "sources": sources}, f)

    print("✅ Research knowledge index built successfully.")


if __name__ == "__main__":
    build_index()
