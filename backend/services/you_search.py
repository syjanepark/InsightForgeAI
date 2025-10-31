import os
import requests
import gzip
import json
from dotenv import load_dotenv
from services.you_news import call_news
from io import BytesIO

# Ensure environment variables from .env are loaded when this module is imported
load_dotenv()

BASE_URL = "https://api.ydc-index.io/v1/search"

def search_web(query: str, count: int = 3):
    """Call You.com Web Search API and return simplified results."""
    
    # Reload env vars on each call to ensure fresh key
    # Try loading from backend/.env explicitly (assuming backend/services/you_search.py)
    backend_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    env_path = os.path.join(backend_dir, '.env')
    load_dotenv(env_path)  # backend/.env
    load_dotenv()  # Also try current working directory and default locations
    
    api_key = os.getenv("YDC_API_KEY") or os.getenv("X_API_KEY")
    
    # Debug: log all env var possibilities
    ydc_key = os.getenv("YDC_API_KEY", "")
    x_key = os.getenv("X_API_KEY", "")
    print(f"🔍 Env check - YDC_API_KEY len: {len(ydc_key)}, X_API_KEY len: {len(x_key)}")
    
    # Return empty results if API key is missing
    if not api_key:
        print(f"⚠️  YDC_API_KEY not found - skipping search for: {query}")
        print(f"⚠️  Checked .env at: {env_path}")
        return []
    
    # Debug: log key length (first 8 and last 4 chars for verification)
    key_preview = f"{api_key[:8]}...{api_key[-4:]}" if len(api_key) > 12 else "***"
    print(f"🔐 Using API key: {key_preview} (len={len(api_key)})")
    
    # Warn if key looks like a placeholder
    if len(api_key) < 50 or "YOUR" in api_key.upper() or "PLACEHOLDER" in api_key.upper():
        print(f"⚠️  WARNING: API key looks like a placeholder! Expected ~93 chars, got {len(api_key)}")
        print(f"⚠️  Make sure YDC_API_KEY=ydc-sk-... is in backend/.env")
    
    headers = {
        "X-API-Key": api_key,
        "Accept": "application/json",
        "Accept-Encoding": "gzip, deflate",
        "User-Agent": "InsightForgeAI/1.0"
    }
    # Official params for Web Search
    params = {"query": query, "count": count}
    
    # Debug: log the request being made
    print(f"🌐 Request: GET {BASE_URL}?query={query[:50]}...&count={count}")

    try:
        r = requests.get(BASE_URL, headers=headers, params=params, timeout=8)
        if r.status_code != 200:
            print(f"⚠️  Search API non-200: {r.status_code} body={r.text[:200]}")
            print(f"🔍 Response headers: {dict(r.headers)}")
            raise requests.HTTPError(f"status={r.status_code}")
    except Exception as e:
        print(f"⚠️  Search API failed for '{query}': {e}")
        # Fallback to news if search is blocked/missing scopes
        try:
            news = call_news(query, count=count)
            if news:
                return news
        except Exception:
            pass
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
        web_results = data.get("results", {}).get("web", [])
        # Also check for news results in the same response
        news_results = data.get("results", {}).get("news", [])
        
        for item in web_results:
            # Handle both "description" and "snippets" fields
            snippet = item.get("description", "")
            if not snippet and item.get("snippets"):
                snippet = " ".join(item.get("snippets", []))
            results.append({
                "title": item.get("title"),
                "url": item.get("url"),
                "snippet": snippet
            })
        
        # Add news items too if available
        for item in news_results:
            snippet = item.get("description", "")
            if not snippet and item.get("snippets"):
                snippet = " ".join(item.get("snippets", []))
            results.append({
                "title": item.get("title"),
                "url": item.get("url"),
                "snippet": snippet
            })
            
        if results:
            print(f"✅ Parsed {len(results)} results from Search API")
            return results
        # If no results, attempt news fallback
        news = call_news(query, count=count)
        if not news:
            print("ℹ️  Search yielded no web results; News fallback also empty.")
        return news
    except Exception as e:
        print(f"⚠️  Failed to parse search results for '{query}': {e}")
        return []