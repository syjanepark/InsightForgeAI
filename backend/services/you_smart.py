import os
import requests
import gzip
import json
from dotenv import load_dotenv
from io import BytesIO

load_dotenv()

API_KEY = os.getenv("YDC_API_KEY")
SMART_URL = "https://api.you.com/api/v1/smart"

def call_smart(prompt: str):
    """Call You.com Smart API for reasoning/summarization."""
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
        "Accept-Encoding": "gzip, deflate",
        "User-Agent": "InsightForgeAI/1.0"
    }

    body = {
        "model": "smart-large",
        "input": prompt,
        "temperature": 0.3,
    }

    r = requests.post(SMART_URL, headers=headers, json=body, timeout=30)

    # handle gzip safely
    if r.headers.get("Content-Encoding") == "gzip":
        buf = BytesIO(r.content)
        with gzip.GzipFile(fileobj=buf) as f:
            response = json.loads(f.read().decode("utf-8"))
    else:
        response = r.json()

    return response.get("output", "")