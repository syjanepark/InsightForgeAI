import asyncio
from services.you_search import search_web

class ResearchAgent:
    async def query(self, keyword: str):
        queries = [
            f"{keyword} market trends 2025",
            f"{keyword} growth forecast",
            f"{keyword} industry report 2024"
        ]
        # Run the synchronous search_web function in thread pool
        loop = asyncio.get_event_loop()
        tasks = [loop.run_in_executor(None, search_web, q) for q in queries]
        results = await asyncio.gather(*tasks)
        # flatten + dedupe by title
        seen, combined = set(), []
        for group in results:
            for r in group:
                if r["title"] not in seen:
                    seen.add(r["title"])
                    combined.append(r)
        return combined[:5]