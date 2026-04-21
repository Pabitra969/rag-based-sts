# retrievers/faiss_retriever.py
import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import List, Dict, Any
import numpy as np
import json
from .base_retriever import BaseRetriever

# Try to import faiss safely
try:
    import faiss  # type: ignore
    _FAISS_AVAILABLE = True
except Exception:
    faiss = None  # type: ignore
    _FAISS_AVAILABLE = False

_POOL = ThreadPoolExecutor(max_workers=2)

FAISS_INDEX_PATH = "./faiss_index.index"
FAISS_META_PATH = "./faiss_meta.json"

class FaissRetriever(BaseRetriever):
    def __init__(self, index_path: str = FAISS_INDEX_PATH, meta_path: str = FAISS_META_PATH, debug: bool = False):
        self.debug = debug
        self.metadata = []
        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                self.metadata = json.load(f)
        except Exception:
            self.metadata = []
        if _FAISS_AVAILABLE:
            try:
                self.index = faiss.read_index(index_path)  # type: ignore
            except Exception as e:
                if self.debug:
                    print(f"FaissRetriever init warning: {e}")
                self.index = None
        else:
            if self.debug:
                print("FaissRetriever: faiss not available, will fallback to local retriever")
            self.index = None

    async def search_with_embedding(self, embedding: list, top_k: int = 3) -> List[Dict[str, Any]]:
        if not _FAISS_AVAILABLE or self.index is None:
            raise RuntimeError("Faiss not available")
        emb = np.array(embedding, dtype=np.float32).reshape(1, -1)
        emb /= np.linalg.norm(emb, axis=1, keepdims=True) + 1e-9

        loop = asyncio.get_running_loop()

        def _q():
            D, I = self.index.search(emb, top_k)
            out = []
            for i, dist in zip(I[0], D[0]):
                if i == -1: continue
                meta = self.metadata[i]
                out.append({
                    "source": "faiss",
                    "content": meta.get("content", ""),
                    "metadata": meta,
                    "score": float(1 / (1 + dist)),
                    "raw_distance": float(dist)
                })
            return out

        results = await loop.run_in_executor(_POOL, _q)
        if self.debug:
            print(f"🔍 FaissRetriever: {len(results)} results")
        return results

    async def search(self, query: str, top_k: int = 3):
        raise RuntimeError("Use search_with_embedding(embedding, top_k).")
