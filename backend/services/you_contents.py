import os
import requests
from typing import List, Dict
from dotenv import load_dotenv

# Ensure environment variables from .env are loaded when this module is imported
load_dotenv()

YOU_API_KEY = os.getenv("YDC_API_KEY") or os.getenv("X_API_KEY")

def fetch_contents(urls: List[str]) -> List[Dict]:
    """Fetch full-text content for a small list of URLs using You Contents API.
    Returns list of {url, text, title} dicts. Falls back to empty list on error.
    """
    if not urls:
        return []
    if not YOU_API_KEY:
        return []
    try:
        endpoint = "https://api.ydc-index.io/v1/contents"
        headers = {"X-API-Key": YOU_API_KEY, "Content-Type": "application/json"}
        payload = {"urls": urls[:5], "format": "html"}
        resp = requests.post(endpoint, json=payload, headers=headers, timeout=20)
        if resp.status_code != 200:
            return []
        data = resp.json() or {}
        items = data.get("results") or data.get("pages") or data.get("contents") or []
        out = []
        for it in items:
            out.append({
                "url": it.get("url"),
                "title": it.get("title"),
                "text": it.get("html") or it.get("text") or it.get("content") or it.get("summary")
            })
        return out
    except Exception:
        return []


