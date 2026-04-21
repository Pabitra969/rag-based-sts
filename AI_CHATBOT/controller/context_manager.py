import asyncio
from typing import List, Dict, Any, Optional

class ContextManager:
    def __init__(self, embedder, faiss_retriever, local_retriever, sessions, top_k:int = 4, debug:bool=False):
        self.embedder = embedder
        self.faiss_retriever = faiss_retriever
        self.local_retriever = local_retriever
        self.sessions = sessions
        self.top_k = top_k
        self.debug = debug

    async def fetch_retrieval(self, query: str) -> List[Dict[str, Any]]:
        # Embed then call chroma retriever; fallback to local retriever on error
        try:
            emb_tuple = self.embedder.encode_cached(query)
            emb = list(emb_tuple)
            items = await self.faiss_retriever.search_with_embedding(emb, top_k=self.top_k)
            if self.debug:
                print("ContextManager: Faiss retrieved", len(items))
            return items
        except Exception as e:
            if self.debug:
                print("ContextManager: Faiss retrieval failed, falling back to local:", e)
            return await self.local_retriever.search(query, top_k=self.top_k)

    def build_sources_text(self, results: List[Dict[str, Any]], max_chars: int = 1400, recent_bot_short: str = "") -> str:
        """
        Builds a cleaned, category-aware retrieval context string for model input.
        Automatically avoids mixing unrelated product categories (reduces hallucination).
        """
        if not results:
            return ""

        lines = []
        categories = set()

        for r in results:
            meta = r.get("metadata") or {}
            title = meta.get("title") or meta.get("name") or meta.get("id", "")
            price = meta.get("price", "")
            desc = meta.get("description", "") or r.get("content", "")
            category = meta.get("category", "")
            if not desc:
                continue
            if recent_bot_short and recent_bot_short.strip() in desc:
                continue

            categories.add(category)
            line = f"- {title} ₹{price} [{category}]: {desc}".strip()
            lines.append(line)

        # === Category filtering ===
        # Prevent cross-category confusion (like saree + electronics in same context)
        if len(categories) > 1:
            # Keep majority category only (most frequent among retrieved results)
            category_list = [r.get("metadata", {}).get("category", "") for r in results]
            dominant = max(set(category_list), key=category_list.count)
            lines = [l for l in lines if f"[{dominant}]" in l]
            if self.debug:
                print(f"[ContextManager] Filtered to dominant category: {dominant}")

        ctx = "\n".join(lines)

        # Limit context size to avoid excessive token load
        if len(ctx) > max_chars:
            ctx = ctx[:max_chars].rsplit("\n", 1)[0] + "\n..."

        return ctx

    def get_short_history(self, user_id: str) -> str:
        return self.sessions.get_recent_short(user_id)
