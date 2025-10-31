import os
import requests
from typing import List, Dict
from dotenv import load_dotenv

# Ensure environment variables from .env are loaded when this module is imported
load_dotenv()

YOU_API_KEY = os.getenv("YDC_API_KEY") or os.getenv("X_API_KEY")

def call_news(query: str, count: int = 5) -> List[Dict]:
    """Fetch recent news snippets for a query using You.com News API.
    Returns a list of {title, url, snippet} dicts. Sends query/count to the API.
    Falls back to empty list if no key or error.
    """
    try:
        if not YOU_API_KEY:
            return []
        url = "https://api.ydc-index.io/livenews"
        headers = {"X-API-Key": YOU_API_KEY}
        # Official params for Live News
        params = {"q": (query or ""), "limit": count}
        resp = requests.get(url, headers=headers, params=params, timeout=15)
        # Some deployments of livenews reject params; retry without params on 4xx
        if resp.status_code != 200:
            try:
                body = (resp.text or "")[:200].lower()
            except Exception:
                body = ""
            if resp.status_code == 400 or "invalid request parameter" in body:
                # Retry without params
                resp = requests.get(url, headers=headers, timeout=15)
                if resp.status_code != 200:
                    return []
                data = resp.json() or {}
                items = (
                    data.get("results")
                    or data.get("news")
                    or data.get("items")
                    or data.get("articles")
                    or data.get("data")
                    or (data.get("result", {}) or {}).get("news")
                    or []
                )
                # If we fetched without params, do a simple keyword filter client-side
                q = (query or "").strip().lower()
                if q:
                    filtered = [
                        it for it in items
                        if q in (str(it.get("title",""))+" "+str(it.get("snippet",""))+" "+str(it.get("summary",""))).lower()
                    ]
                    # If the strict match yields nothing, fall back to top items
                    items = filtered if filtered else items
            else:
                return []
        else:
            data = resp.json() or {}
            # Try multiple common shapes
            items = (
                data.get("results")
                or data.get("news")
                or data.get("items")
                or data.get("articles")
                or data.get("data")
                or (data.get("results", {}) or {}).get("news")
                or (data.get("result", {}) or {}).get("news")
                or []
            )
        out = []
        for it in items[:count]:
            out.append({
                "title": it.get("title") or it.get("name") or "",
                "url": it.get("url") or "",
                "snippet": it.get("snippet") or it.get("summary") or ""
            })
        return out
    except Exception:
        return []


