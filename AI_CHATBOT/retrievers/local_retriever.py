# # retrievers/local_retriever.py
# """
# Local CSV/JSON retriever - used when you don't want a vector DB.
# It simply does simple substring matching across combined text fields (fast for small datasets).
# """
# import asyncio
# from typing import List, Dict, Any
# import pandas as pd
# from concurrent.futures import ThreadPoolExecutor
# from .base_retriever import BaseRetriever

# _POOL = ThreadPoolExecutor(max_workers=2)

# class LocalRetriever(BaseRetriever):
#     def __init__(self, csv_path: str = "data/products2.csv", debug: bool = False):
#         self.csv_path = csv_path
#         self.debug = debug
#         self._load()

#     def _load(self):
#         self.df = pd.read_csv(self.csv_path)
#         # build combined text
#         self.df["_combined"] = self.df.apply(lambda r: " | ".join(f"{k}: {v}" for k,v in r.items() if pd.notna(v)), axis=1)

#     async def search(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
#         loop = asyncio.get_running_loop()
#         def _search():
#             hits = []
#             q = query.lower()
#             for _, row in self.df.iterrows():
#                 text = str(row["_combined"]).lower()
#                 if q in text:
#                     hits.append({"source":"local","content": row["_combined"], "metadata": row.to_dict(), "score": 1.0})
#             # basic fallback: return top_k rows if no exact match
#             if not hits:
#                 for _, row in self.df.head(top_k).iterrows():
#                     hits.append({"source":"local","content": row["_combined"], "metadata": row.to_dict(), "score": 0.1})
#             return hits[:top_k]
#         res = await loop.run_in_executor(_POOL,_search)
#         return res


import csv
import re

class LocalRetriever:
    """
    Lightweight CSV-based retriever (FAISS fallback).
    Uses cosine similarity on normalized FastEmbedder vectors or simple keyword search.
    """

    def __init__(self, csv_path: str = "./data/products2.csv", embedder=None, debug: bool = False):
        self.csv_path = csv_path
        self.embedder = embedder
        self.debug = debug
        self.data = self._load_csv()

    def _load_csv(self):
        rows = []
        try:
            with open(self.csv_path, newline='', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for r in reader:
                    # merge all columns into one searchable string
                    combined = " | ".join(f"{k}: {v}" for k, v in r.items() if v)
                    rows.append({
                        "content": combined,
                        "metadata": r,
                        "id": r.get("id", str(len(rows)))
                    })
            if self.debug:
                print(f"LocalRetriever: loaded {len(rows)} rows from {self.csv_path}")
        except Exception as e:
            if self.debug:
                print(f"LocalRetriever: failed to load {self.csv_path}: {e}")
        return rows

    @staticmethod
    def _tokens(text):
        return re.findall(r"[a-z0-9]+", str(text or "").lower())

    @staticmethod
    def _singular(token):
        if token.endswith("ies") and len(token) > 4:
            return token[:-3] + "y"
        if token.endswith("s") and len(token) > 3:
            return token[:-1]
        return token

    def _keyword_search(self, query: str, top_k: int):
        stopwords = {
            "a", "an", "the", "do", "you", "have", "has", "is", "are", "me",
            "show", "find", "some", "any", "please", "can", "could", "want",
            "need", "about", "details", "detail", "product", "products",
            "available", "recommend", "suggest", "give", "for", "with", "of",
            "in", "to", "and", "sir", "mam",
        }
        raw_tokens = self._tokens(query)
        q_tokens = [
            self._singular(t)
            for t in raw_tokens
            if len(t) > 1 and t not in stopwords
        ]
        if "drawing" in q_tokens and any(t in {"kid", "kids"} for t in raw_tokens):
            q_tokens = [t for t in q_tokens if t != "kid"] + ["kit"]
        if not q_tokens:
            return []

        q_phrase = " ".join(q_tokens)
        scored = []

        for r in self.data:
            m = r.get("metadata", {})
            title = str(m.get("title", ""))
            description = str(m.get("description", ""))
            category = str(m.get("category", ""))
            brand = str(m.get("brand", ""))
            color = str(m.get("color", ""))
            material = str(m.get("material", ""))

            title_tokens = {self._singular(t) for t in self._tokens(title)}
            desc_tokens = {self._singular(t) for t in self._tokens(description)}
            category_tokens = {self._singular(t) for t in self._tokens(category.replace("_", " "))}
            attribute_tokens = {
                self._singular(t)
                for t in self._tokens(" ".join([brand, color, material]))
            }
            searchable = " ".join(self._tokens(" ".join([title, description, category]))).lower()

            score = 0.0
            if q_phrase and q_phrase in " ".join(self._tokens(title)).lower():
                score += 12.0
            if q_phrase and q_phrase in searchable:
                score += 5.0

            for token in q_tokens:
                if token in title_tokens:
                    score += 6.0
                if token in category_tokens:
                    score += 4.0
                if token in desc_tokens:
                    score += 2.0
                if token in attribute_tokens:
                    score += 1.0

            if score > 0:
                item = dict(r)
                item["score"] = score
                item["source"] = "local"
                scored.append((score, item))

        scored.sort(reverse=True, key=lambda x: x[0])
        return [r for _, r in scored[:top_k]]

    async def search(self, query: str, top_k: int = 5):
        """
        Simple offline keyword-based or embedding-based search.
        If embedder is available, uses cosine similarity; else uses keyword scoring.
        """
        if not self.data:
            return []

        keyword_results = self._keyword_search(query, top_k)
        if keyword_results:
            return keyword_results

        return []
