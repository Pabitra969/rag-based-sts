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
import json
import numpy as np

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

    async def search(self, query: str, top_k: int = 5):
        """
        Simple offline keyword-based or embedding-based search.
        If embedder is available, uses cosine similarity; else uses keyword scoring.
        """
        if not self.data:
            return []

        # Use embeddings if embedder exists
        if self.embedder:
            try:
                query_emb = np.array(self.embedder.encode([query])[0], dtype=np.float32)
                scores = []
                for r in self.data:
                    emb_text = r["content"]
                    emb_vec = np.array(self.embedder.encode([emb_text])[0], dtype=np.float32)
                    sim = float(np.dot(query_emb, emb_vec))
                    scores.append((sim, r))
                scores.sort(reverse=True, key=lambda x: x[0])
                results = [r for _, r in scores[:top_k]]
                if self.debug:
                    print(f"LocalRetriever: embed search returned {len(results)} results")
                return results
            except Exception as e:
                if self.debug:
                    print(f"LocalRetriever embed fallback: {e}")

        # fallback: simple keyword matching
        q_words = query.lower().split()
        scored = []
        for r in self.data:
            text = r["content"].lower()
            score = sum(text.count(w) for w in q_words)
            scored.append((score, r))
        scored.sort(reverse=True, key=lambda x: x[0])
        return [r for _, r in scored[:top_k]]
