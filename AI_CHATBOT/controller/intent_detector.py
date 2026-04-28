# controller/intent_detector.py
import re
from typing import Tuple
import numpy as np
from fast_embedder import FastEmbedder


# =========================
# INTENT DEFINITIONS
# =========================
INTENT_BANK = {
    "greeting": ["hi", "hello", "hey", "good morning", "good evening"],
    "thanks": ["thanks", "thank you", "appreciate", "grateful"],
    "farewell": ["bye", "goodbye", "see you", "take care"],

    # GENERAL DOMAIN
    "general_knowledge": [
        "what is black hole",
        "who invented telephone",
        "explain photosynthesis",
        "why is sky blue",
        "capital of india",
        "difference between ai and ml",
    ],

    # PRODUCT DOMAIN
    "product_search": [
        "find product", "show items", "search clothing",
        "display products", "recommend items",
        "need jeans", "want shirt"
    ],
    "price_filter": [
        "under price", "below price",
        "cheap products", "affordable items",
        "budget friendly"
    ],
    "meta_count": [
        "how many products", "count items",
        "number of products", "total items available"
    ],

    # USER
    "personal_query": [
        "my order", "order id", "track my order",
        "my profile", "my account", "order status"
    ],
}


# =========================
# EMBEDDINGS CACHE
# =========================
_embedder = FastEmbedder(local_model_path="./local_embedder")

_INTENT_VECS = {
    k: np.mean(_embedder.encode(v), axis=0)
    for k, v in INTENT_BANK.items()
}


# =========================
# STAGE 1: PRODUCT DETECTOR (HARD GATE)
# =========================
def is_product_query(q: str) -> bool:
    product_keywords = r"\b(jeans|t[- ]?shirts?|shirt|pants|shoes|phone|laptop|watch|bag|wallet|saree|dress|clothes|perfume|skincare|hair|cosmetics|chair|table|furniture|mouse|speaker|backpack)\b"

    buying_intent = r"\b(buy|purchase|get|order|want|need|looking for|recommend|suggest|show|find)\b"

    gift_context = r"\b(gift|present|for girlfriend|for boyfriend|for mom|for dad|for wife|for husband)\b"

    vague_product = r"\b(something|anything|options|ideas)\b"

    # Strong signals
    if re.search(product_keywords, q):
        return True

    if re.search(gift_context, q):
        return True

    # Buying intent + vague object
    if re.search(buying_intent, q) and re.search(vague_product, q):
        return True

    # Suggestion-style queries
    if re.search(r"\b(suggest|recommend|show|find)\b", q) and not re.search(r"\b(explain|what is|why)\b", q):
        return True

    return False


# =========================
# MAIN FUNCTION
# =========================
def detect_intent(query: str) -> Tuple[str, float]:
    q = query.lower().strip()

    # =========================
    # STAGE 1: DOMAIN SPLIT
    # =========================
    if is_product_query(q):
        domain = "product"
    else:
        domain = "general"

    # =========================
    # HIGH PRIORITY RULES
    # =========================

    # Greetings
    if re.match(r"^(hi|hello|hey|hlw)\b", q):
        return ("greeting", 0.95)

    if re.match(r"^(thanks|thank you)\b", q):
        return ("thanks", 0.95)

    if re.fullmatch(r"(bye|goodbye|see you|take care)", q):
        return ("farewell", 0.95)

    # Personal queries (always override)
    if re.search(r"\b(my order|order id|track|profile|account|purchase)\b", q):
        return ("personal_query", 0.90)

    # =========================
    # PRODUCT DOMAIN LOGIC
    # =========================
    if domain == "product":

        # Price queries
        if re.search(r"\b(price|cost|under|below|less than|within|cheap|affordable)\b", q):
            return ("price_filter", 0.90)

        # Count queries
        if re.search(r"\b(how many|count|number of)\b", q):
            return ("meta_count", 0.90)

        # Default product intent
        return ("product_search", 0.85)

    # =========================
    # GENERAL DOMAIN LOGIC
    # =========================
    else:

        # Math queries
        if re.search(r"\b\d+\s*[+\-*/]\s*\d+\b", q):
            return ("general_knowledge", 0.95)

        # Live info and date/time
        if re.search(r"\b(temperature|weather|forecast|rain|humidity|wind|news|headline|stock|traffic|score)\b", q):
            return ("general_knowledge", 0.95)

        if re.search(r"\b(what\s+date|current\s+date|today'?s\s+date|what\s+day\s+is\s+it|which\s+day\s+is\s+it|time now|what time|current\s+time)\b", q):
            return ("general_knowledge", 0.95)

        # WH questions ONLY if not product
        if re.search(r"\b(what|why|how|when|where|which|explain|define|meaning)\b", q):
            return ("general_knowledge", 0.85)

    # =========================
    # STAGE 2: EMBEDDING FALLBACK
    # =========================
    q_vec = _embedder.encode([q])[0]

    sims = {
        k: float(np.dot(q_vec, v) / (np.linalg.norm(v) + 1e-9))
        for k, v in _INTENT_VECS.items()
    }

    # Restrict intents by domain
    if domain == "product":
        allowed = ["product_search", "price_filter", "meta_count"]
    else:
        allowed = ["general_knowledge", "greeting", "thanks"]

    sims = {k: v for k, v in sims.items() if k in allowed}

    best_intent = max(sims, key=sims.get)
    best_score = sims[best_intent]

    # Strict threshold
    if best_score < 0.65:
        return ("general_knowledge", best_score)

    return (best_intent, best_score)
