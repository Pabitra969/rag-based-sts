# fast_embedder.py
from functools import lru_cache
from typing import List
import numpy as np, os, json, hashlib


try:
    from fastembed import TextEmbedding
    _FASTEMBED_AVAILABLE = True
except Exception:
    _FASTEMBED_AVAILABLE = False

#self.local_model_path = from_root("local_embedder")
class FastEmbedder:
    def __init__(self, local_model_path = "./local_embedder", cache_path: str = "./cache/embed_cache.json", persist_cache: bool = False):
        self.local_model_path = os.path.abspath(local_model_path)
        self.cache_path = cache_path
        self.persist_cache = persist_cache
        self.model = None
        self.cache = {}

        # Only load on-disk cache if persistence is enabled
        if self.persist_cache and os.path.exists(cache_path):
            try:
                self.cache = json.load(open(cache_path, "r", encoding="utf-8"))
            except Exception:
                self.cache = {}

        if not os.path.exists(self.local_model_path):
            raise RuntimeError(f"❌ Local embedder path not found: {self.local_model_path}")

        if not _FASTEMBED_AVAILABLE:
            raise RuntimeError("❌ fastembed not installed — run `pip install fastembed`")

        # FastEmbed 0.7.x uses 'model_name' and 'cache_dir'
        # We point cache_dir to our local folder, and specify the model name.
        # FastEmbed will look for [cache_dir]/fastembed/[slugified_model_name]
        try:
            self.model = TextEmbedding(model_name="BAAI/bge-small-en-v1.5", 
                                       cache_dir=self.local_model_path, 
                                       local_files_only=True)
        except TypeError:
            # Fallback (unlikely needed if upgraded)
            self.model = TextEmbedding(model_name_or_path=self.local_model_path,
                                       trust_remote_code=False, local_files_only=True)
        print(f"✅ FastEmbedder loaded offline from {self.local_model_path}")

    def _hash(self, text: str) -> str:
        return hashlib.sha1(text.strip().lower().encode()).hexdigest()

    def _save_cache(self):
        if not self.persist_cache:
            return
        os.makedirs(os.path.dirname(self.cache_path), exist_ok=True)
        json.dump(self.cache, open(self.cache_path, "w", encoding="utf-8"))

    @lru_cache(maxsize=2048)
    def encode_cached(self, text: str) -> tuple:
        # Prefer in-memory LRU; optionally mirror to disk if enabled
        h = self._hash(text)
        if h in self.cache:
            return tuple(self.cache[h])
        arr = self.encode([text])[0]
        # mirror to optional disk cache map (bounded by use)
        self.cache[h] = arr.tolist()
        self._save_cache()
        return tuple(arr.tolist())

    def encode(self, texts: List[str]) -> np.ndarray:
        """Encode list of texts into normalized float32 vectors."""
        raw_output = self.model.embed(texts)
        if isinstance(raw_output, np.ndarray):
            arr = raw_output.astype(np.float32)
        else:
            arr = np.array([getattr(e, "embedding", e) for e in raw_output], dtype=np.float32)
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return arr / norms

    def clear_disk_cache(self):
        """Remove on-disk cache file if present."""
        try:
            if os.path.exists(self.cache_path):
                os.remove(self.cache_path)
        except Exception:
            pass
