# faiss_db_setup.py
import pandas as pd
import numpy as np
import faiss
import json
from fast_embedder import FastEmbedder

CSV_PATH = "data/products2.csv"
FAISS_INDEX_PATH = "./faiss_index.index"
FAISS_META_PATH = "./faiss_meta.json"

def build_faiss_db(csv_path=CSV_PATH, index_path=FAISS_INDEX_PATH, meta_path=FAISS_META_PATH):
    df = pd.read_csv(csv_path)
    texts, metas = [], []

    for _, r in df.iterrows():
        combined = " | ".join(f"{k}: {v}" for k, v in r.items() if pd.notna(v))
        texts.append(combined)
        metas.append({"id": str(r.get("id", _)), "content": combined, **r.to_dict()})

    embedder = FastEmbedder()
    embeddings = embedder.encode(texts)
    dim = embeddings.shape[1]

    # normalize and build FAISS index
    embeddings = embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-9)
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings.astype(np.float32))
    faiss.write_index(index, index_path)

    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metas, f, ensure_ascii=False, indent=2)

    print(f"✅ Built FAISS DB with {len(df)} items at {index_path}")

if __name__ == "__main__":
    build_faiss_db()
