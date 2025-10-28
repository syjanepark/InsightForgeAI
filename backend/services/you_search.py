import os
import requests
import gzip
import json
from dotenv import load_dotenv
from io import BytesIO

load_dotenv()

BASE_URL = "https://api.ydc-index.io/v1/search"
API_KEY = os.getenv("YDC_API_KEY") or os.getenv("X-API-Key") or os.getenv("X_API_KEY")

def search_web(query: str, count: int = 3):
    """Call You.com Web Search API and return simplified results."""
    
    # Return empty results if API key is missing
    if not API_KEY:
        print(f"⚠️  YDC_API_KEY not found - skipping search for: {query}")
        return []
    
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Accept": "application/json",
        "Accept-Encoding": "gzip, deflate",
        "User-Agent": "InsightForgeAI/1.0"
    }
    params = {
        "q": query,
        "num_web_results": count,
        "safe_search": "Off",
        "include_domains": "",
        "exclude_domains": "",
    }

    try:
        r = requests.get(BASE_URL, headers=headers, params=params, timeout=5)  # Reduced timeout
        r.raise_for_status()  # Raise exception for bad status codes
    except Exception as e:
        print(f"⚠️  Search API failed for '{query}': {e}")
        return []

    try:
        # handle gzip response
        if r.headers.get("Content-Encoding") == "gzip":
            buf = BytesIO(r.content)
            with gzip.GzipFile(fileobj=buf) as f:
                data = json.loads(f.read().decode("utf-8"))
        else:
            data = r.json()

        results = []
        for item in data.get("web_results", []):
            results.append({
                "title": item.get("title"),
                "url": item.get("url"),
                "snippet": item.get("snippet", "")
            })
        return results
    except Exception as e:
        print(f"⚠️  Failed to parse search results for '{query}': {e}")
        return []