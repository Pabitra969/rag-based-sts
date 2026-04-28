# controller/intent_detector.py
import re
from typing import Tuple
import numpy as np
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
        "need jeans", "want shirt",
        "track pants", "sports shoes", "smart watch", "leather jacket",
        "summer dress", "formal shirt", "running shoes", "denim jeans",
        "silk scarf", "women's bag", "men's belt",
    ],
    "price_filter": [
        "under price", "below price",
        "cheap products", "affordable items",
        "budget friendly",
        "price less than", "cost under",
    ],
    "meta_count": [
        "how many products", "count items",
        "number of products", "total items available",
        "how many shirts", "count jackets",
    ],

    # USER PERSONAL QUERIES
    "personal_query": [
        "my order", "order id", "track order", "track my order",
        "order status", "my profile", "my account",
        "purchase history", "recent orders",
        "track my purchase",
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
    product_keywords = (
        r"\b("
        r"jeans|t[- ]?shirts?|shirt|pants|trouser|shorts|shoes|sneakers|"
        r"phone|laptop|watch|bag|wallet|belt|saree|dress|kurti|kurta|"
        r"sweater|jacket|hoodie|leggings|clothes|clothing|perfume|skincare|"
        r"hair|cosmetics|toothbrush|trimmer|chair|table|furniture|mouse|speaker|"
        r"headphones?|headset|earphones?|earbuds|backpack|keyboard|webcam|"
        r"ssd|tablet|drawing|graphic|plug|charger|router|screen protector|cooler|purifier|"
        r"heater|fan|geyser|refrigerator|microwave|dishwasher|product|item|catalog"
        r")\b"
    )

    buying_intent = r"\b(buy|purchase|get|order|want|need|looking for|recommend|suggest|show|find|details?)\b"

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
    # HIGH PRIORITY RULES (Regex)
    # =========================

    # Greetings
    if re.match(r"^(hi|hello|hey|hlw)\b", q):
        regex_intent, regex_conf = "greeting", 0.95
    elif re.match(r"^(thanks|thank you)\b", q):
        regex_intent, regex_conf = "thanks", 0.95
    elif re.fullmatch(r"(bye|goodbye|see you|take care)", q):
        regex_intent, regex_conf = "farewell", 0.95
    elif re.search(r"\b(my order|order id|track order|track my order|order status|profile|account|purchase)\b", q):
        regex_intent, regex_conf = "personal_query", 0.90
    elif domain == "product" and re.search(r"\b(price|cost|under|below|less than|within|cheap|affordable)\b", q):
        regex_intent, regex_conf = "price_filter", 0.90
    elif domain == "product" and re.search(r"\b(how many|count|number of)\b", q):
        regex_intent, regex_conf = "meta_count", 0.90
    elif domain == "product":
        regex_intent, regex_conf = "product_search", 0.85
    else:
        # General domain heuristics
        if re.search(r"\b\d+\s*[+\-*/]\s*\d+\b", q):
            regex_intent, regex_conf = "general_knowledge", 0.95
        elif re.search(r"\b(temperature|weather|forecast|rain|humidity|wind|news|headline|stock|traffic|score)\b", q):
            regex_intent, regex_conf = "general_knowledge", 0.95
        elif re.search(r"\b(what\s+date|current\s+date|today'?s\s+date|what\s+day\s+is\s+it|which\s+day\s+is\s+it|time now|what time|current\s+time)\b", q):
            regex_intent, regex_conf = "general_knowledge", 0.95
        elif re.search(r"\b(what|why|how|when|where|which|explain|define|meaning)\b", q):
            regex_intent, regex_conf = "general_knowledge", 0.85
        else:
            regex_intent, regex_conf = "general_knowledge", 0.80

    # =========================
    # STAGE 2: EMBEDDING FALLBACK (Always run)
    # =========================
    q_vec = _embedder.encode([q])[0]
    sims = {
        k: float(np.dot(q_vec, v) / (np.linalg.norm(v) + 1e-9))
        for k, v in _INTENT_VECS.items()
    }

    # Restrict intents by domain for safety
    if domain == "product":
        allowed = ["product_search", "price_filter", "meta_count"]
    else:
        allowed = ["general_knowledge", "greeting", "thanks", "farewell"]
    sims = {k: v for k, v in sims.items() if k in allowed}

    embed_intent = max(sims, key=sims.get) if sims else "general_knowledge"
    embed_score = sims.get(embed_intent, 0.0)

    # Choose the intent with higher confidence between regex and embedding
    if embed_score > regex_conf:
        return (embed_intent, embed_score)
    else:
        return (regex_intent, regex_conf)
