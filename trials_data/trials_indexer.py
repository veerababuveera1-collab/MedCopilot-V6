import os
import json
import pickle
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

DATA_DIR = "research_ai/trials_data"
VECTOR_DIR = "research_ai/vector_trials"
INDEX_FILE = os.path.join(VECTOR_DIR, "trials.index")
CACHE_FILE = os.path.join(VECTOR_DIR, "trials.pkl")

os.makedirs(VECTOR_DIR, exist_ok=True)

model = SentenceTransformer("all-MiniLM-L6-v2")


def parse_trials(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    docs = []
    sources = []

    studies = data["StudyFieldsResponse"]["StudyFields"]

    for study in studies:
        nct = study.get("NCTId", [""])[0]
        title = study.get("BriefTitle", [""])[0]
        condition = study.get("Condition", [""])[0]
        phase = study.get("Phase", [""])[0]
        status = study.get("OverallStatus", [""])[0]

        text = f"""
Trial ID: {nct}
Title: {title}
Condition: {condition}
Phase: {phase}
Status: {status}
"""

        if len(text.strip()) > 100:
            docs.append(text)
            sources.append(f"NCT ID: {nct}")

    return docs, sources


def build_trials_index():
    documents = []
    sources = []

    for file in os.listdir(DATA_DIR):
        if file.endswith(".json"):
            d, s = parse_trials(os.path.join(DATA_DIR, file))
            documents.extend(d)
            sources.extend(s)

    if not documents:
        print("❌ No trials found.")
        return

    print(f"🧠 Indexing {len(documents)} clinical trials...")

    embeddings = model.encode(documents, batch_size=16)
    dim = embeddings.shape[1]

    index = faiss.IndexFlatL2(dim)
    index.add(np.array(embeddings))

    faiss.write_index(index, INDEX_FILE)

    with open(CACHE_FILE, "wb") as f:
        pickle.dump({"documents": documents, "sources": sources}, f)

    print("✅ Clinical trials knowledge index built successfully.")


if __name__ == "__main__":
    build_trials_index()

