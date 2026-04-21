# nlp_core.py
import re
from typing import List, Dict, Any
import json, os

# Load full catalog once (for counts and deterministic suggestions)
_CATALOG = None
def _load_catalog():
    global _CATALOG
    if _CATALOG is not None:
        return _CATALOG
    meta_path = os.path.abspath("./faiss_meta.json")
    csv_path = os.path.abspath("./data/products2.csv")
    data = []
    try:
        if os.path.exists(meta_path):
            data = json.load(open(meta_path, "r", encoding="utf-8"))
        elif os.path.exists(csv_path):
            import csv
            with open(csv_path, newline='', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for r in reader:
                    data.append({
                        "title": r.get("title",""),
                        "description": r.get("description",""),
                        "price": int(r.get("price","0") or 0),
                        "category": r.get("category",""),
                        "id": int(r.get("id","0") or 0)
                    })
    except Exception:
        data = []
    _CATALOG = data or []
    return _CATALOG

def safe_get_meta(r, key, default=""):
    m = r.get("metadata") or {}
    return m.get(key) or m.get(key.lower()) or default

def find_price_in_text(text: str):
    if not text: return None
    m = re.search(r"₹\s?[\d,]+", text)
    if m: return m.group(0).replace(" ", "")
    m2 = re.search(r"\b(\d{3,7})\b", text)
    return "₹" + m2.group(1) if m2 else None

def _find_items_by_titles(titles: List[str]) -> List[Dict[str, Any]]:
    cats = _load_catalog()
    wanted = {t.lower() for t in titles}
    out = []
    for r in cats:
        if r.get("title","{}").lower() in wanted:
            out.append(r)
    return out

def _format_item_line(item: Dict[str, Any]) -> str:
    t = item.get("title", "Product")
    p = item.get("price", "")
    c = item.get("category", "")
    d = item.get("description", "")
    return f"- {t} ₹{p} [{c}] - {d}"

def extract_fact(query: str, results: List[Dict[str, Any]]) -> str | None:
    """
    Extract deterministic facts from results only if query matches product-specific intent.
    Returns None if facts are unreliable or query is general knowledge.
    Also handles catalog-wide facts (counts) and curated gift suggestions.
    """
    q = query.lower().strip()
    cats = _load_catalog()

    # === COUNT QUERIES (use full catalog, not top_k results) ===
    if any(w in q for w in ["how many", "number of", "count", "total"]) and any(w in q for w in ["product", "item", "available", "catalog", "inventory"]):
        return f"We have {len(cats)} products available."

    if not results and not cats:
        return None

    # === PRICE QUERIES ===
    if any(w in q for w in ["price", "cost", "how much", "rate", "price of"]):
        for r in results or []:
            p = safe_get_meta(r, "price", "")
            if p:
                t = safe_get_meta(r, "title", safe_get_meta(r, "name", "Product"))
                return f"{t} costs ₹{p}."
            f = find_price_in_text(r.get("content", ""))
            if f:
                t = safe_get_meta(r, "title", safe_get_meta(r, "name", "Product"))
                return f"{t} costs {f}."
        return "Price information not found for these results."

    # === GIFT SUGGESTIONS ===
    if re.search(r"\b(gift|present|girlfriend|girlfrind|boyfriend|wife|husband)\b", q):
        picks = _find_items_by_titles(["Silk Saree", "Smart Watch"]) or []
        if picks:
            lines = "\n".join(_format_item_line(i) for i in picks)
            return f"Great gift picks:\n{lines}"

    # === NEGATIVE AVAILABILITY GATING for specific product names ===
    if re.search(r"\bgaming\b", q) and ("pc" in q or "computer" in q):
        # Check if any catalog item matches PC/computer — if none, say not available and suggest related
        has_pc = any(re.search(r"\b(pc|computer|desktop)\b", (r.get("title","") + " " + r.get("category","")).lower()) for r in cats)
        if not has_pc:
            # Suggest related from electronics
            related = _find_items_by_titles(["Wireless Mouse"]) or [r for r in cats if r.get("category") == "electronics"][:1]
            lines = "\n".join(_format_item_line(i) for i in related)
            return f"Sorry, we don't have gaming PCs right now. You might like these instead:\n{lines}"

    # === GENERAL PRODUCT DESCRIPTION (Only if query clearly product-related) ===
    # If query contains product keywords, return best matching result from results list
    product_keywords = r"\b(jeans|t-shirt|shirt|pants|shoes|electronics|phone|laptop|watch|bag|wallet|saree|dress|perfume|cream|skincare|headphone|speaker|chair|table|furniture|mouse)\b"
    if re.search(product_keywords, q):
        # Prefer first result that matches key terms from the query in title/description/category
        q_terms = re.findall(r"[a-zA-Z-]+", q)
        chosen = None
        for r in results or []:
            m = r.get("metadata") or {}
            txt = f"{m.get('title','')} {m.get('description','')} {m.get('category','')}".lower()
            if any(term in txt for term in q_terms):
                chosen = r
                break
        top = chosen or ((results or []) and (results or [])[0])
        if top:
            t = safe_get_meta(top, "title", safe_get_meta(top, "name", "Product"))
            p = safe_get_meta(top, "price", "")
            c = safe_get_meta(top, "category", "")
            d = safe_get_meta(top, "description", "") or (top.get("content") or "")[:220]
            head = " | ".join([x for x in [t, (f"₹{p}" if p else None), c] if x])
            return f"{head}. {d}".strip()

    # === NO DETERMINISTIC FACT FOUND ===
    # For general knowledge queries, don't force product info
    return None
