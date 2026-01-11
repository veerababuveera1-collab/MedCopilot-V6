import os
import requests
import json
from datetime import datetime

SAVE_DIR = "research_ai/trials_data"
os.makedirs(SAVE_DIR, exist_ok=True)

BASE_URL = "https://clinicaltrials.gov/api/query/study_fields"


def fetch_trials(condition, max_results=50):
    print(f"🔍 Searching ClinicalTrials.gov for: {condition}")

    params = {
        "expr": condition,
        "fields": "NCTId,BriefTitle,Condition,Phase,EnrollmentCount,OverallStatus",
        "min_rnk": 1,
        "max_rnk": max_results,
        "fmt": "json"
    }

    response = requests.get(BASE_URL, params=params)
    data = response.json()

    filename = f"trials_{condition.replace(' ', '_')}_{datetime.now().strftime('%Y%m%d')}.json"
    path = os.path.join(SAVE_DIR, filename)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

    print(f"✅ Saved: {path}")


if __name__ == "__main__":
    condition = input("Enter disease/condition: ")
    fetch_trials(condition)

