import os
import requests
import xml.etree.ElementTree as ET
from datetime import datetime

SAVE_DIR = "research_ai/data/pubmed"
os.makedirs(SAVE_DIR, exist_ok=True)

PUBMED_SEARCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
PUBMED_FETCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"


def fetch_pubmed(query, max_results=50):
    print(f"🔍 Searching PubMed for: {query}")

    params = {
        "db": "pubmed",
        "term": query,
        "retmax": max_results,
        "retmode": "xml"
    }

    search = requests.get(PUBMED_SEARCH_URL, params=params)
    root = ET.fromstring(search.text)

    ids = [id.text for id in root.findall(".//Id")]
    print(f"📄 Found {len(ids)} papers")

    if not ids:
        return

    fetch_params = {
        "db": "pubmed",
        "id": ",".join(ids),
        "retmode": "xml"
    }

    fetch = requests.get(PUBMED_FETCH_URL, params=fetch_params)

    filename = f"pubmed_{query.replace(' ', '_')}_{datetime.now().strftime('%Y%m%d')}.xml"
    path = os.path.join(SAVE_DIR, filename)

    with open(path, "w", encoding="utf-8") as f:
        f.write(fetch.text)

    print(f"✅ Saved: {path}")


if __name__ == "__main__":
    topic = input("Enter medical topic: ")
    fetch_pubmed(topic)
