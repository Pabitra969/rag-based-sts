# nlp_core.py
import re
from typing import List, Dict, Any, Optional
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
        if os.path.exists(csv_path):
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
        elif os.path.exists(meta_path):
            data = json.load(open(meta_path, "r", encoding="utf-8"))
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

def extract_fact(query: str, results: List[Dict[str, Any]], voice_mode: bool = False) -> Optional[str]:
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

    # === NEGATIVE AVAILABILITY GATING for specific unavailable product names ===
    if re.search(r"\bgaming\b", q) and re.search(r"\bchair\b", q):
        has_gaming_chair = any(
            "gaming" in f"{r.get('title','')} {r.get('description','')}".lower()
            and "chair" in f"{r.get('title','')} {r.get('description','')}".lower()
            for r in cats
        )
        if not has_gaming_chair:
            related = _find_items_by_titles(["Office Chair"])
            lines = "\n".join(_format_item_line(i) for i in related)
            return f"I don't have a gaming chair in the current catalog. Related option:\n{lines}" if lines else "I don't have a gaming chair in the current catalog."

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



    # === GENERAL PRODUCT LISTING (Zero-latency fallback) ===
    # If the user just wants to browse products (not asking for details), return a fast deterministic list.
    product_keywords = (
        r"\b("
        r"jeans|t[- ]?shirt|shirt|pants|trouser|shorts|shoes|sneakers|"
        r"electronics|phone|laptop|watch|bag|wallet|belt|saree|dress|kurti|"
        r"kurta|sweater|jacket|hoodie|leggings|perfume|cream|skincare|trimmer|"
        r"headphones?|headset|earphones?|earbuds|speaker|chair|table|furniture|"
        r"mouse|keyboard|webcam|ssd|tablet|drawing|graphic|plug|charger|router|"
        r"cooler|purifier|heater|fan|geyser|refrigerator|microwave|dishwasher|"
        r"toothbrush|screen protector|"
        r"blender|mixer|kettle|cooker|pressure cooker|frying pan|knife|"
        r"chopping board|tiffin|casserole|idli|grater|rolling pin|roti maker|"
        r"spice rack|mixing bowl|oil dispenser|food storage|container|"
        r"stool|bar stool|recliner|bookshelf|sofa|mattress|pillow|bedsheet|curtain|"
        r"iron|washing machine|vacuum|air conditioner|ac|tv|television|grinder|mixer grinder|"
        r"product|item|options"
        r")\b"
    )
    # Only trigger detail mode when user EXPLICITLY asks for details/description
    detail_keywords = r"\b(detail|details|describe|explain|information|features?|specification|specs?)\b"
    # Also trigger detail mode when user says "tell me about X" or "more about X"
    detail_phrase = r"\b(tell me about|more about|know about|info about|details? of|details? about)\b"

    is_detail_query = bool(re.search(detail_keywords, q)) or bool(re.search(detail_phrase, q))

    if re.search(product_keywords, q) and not is_detail_query:
        if not results:
            return "I don't have that product in the current catalog."
        
        if voice_mode:
            count = min(len(results), 3)
            cat = safe_get_meta(results[0], "category", "products")
            names = [safe_get_meta(r, "title", safe_get_meta(r, "name", "product")) for r in results[:count]]
            if count == 1:
                return f"I found the {names[0]} in the {cat} section."
            return f"I found {count} {cat} items for you: {', '.join(names[:-1])} and {names[-1]}."

        lines = ["Here are some options for you:"]
        for r in results[:3]:
            t = safe_get_meta(r, "title", safe_get_meta(r, "name", "Product"))
            p = safe_get_meta(r, "price", "")
            c = safe_get_meta(r, "category", "")
            d = safe_get_meta(r, "description", "") or (r.get("content") or "")[:120]
            head = " | ".join([x for x in [t, (f"₹{p}" if p else None), c] if x])
            lines.append(f"{head}. {d}")
        return "\n".join(lines)

    # === NO DETERMINISTIC FACT FOUND ===
    # For general knowledge queries, don't force product info
    return None
