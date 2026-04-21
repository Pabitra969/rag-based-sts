# controller/intent_detector.py
import re
from typing import Tuple
import numpy as np
from fast_embedder import FastEmbedder

# Predefined intents and sample trigger phrases
INTENT_BANK = {
    "greeting": ["hi", "hello", "hey", "good morning", "good evening"],
    "thanks": ["thanks", "thank you", "appreciate", "grateful"],
    "farewell": ["bye", "goodbye", "see you", "take care"],
    "price_filter": ["under price", "below price", "cheap products", "affordable items", "budget friendly"],
    "product_search": ["find product", "show items", "search clothing", "display products", "recommend items", "need jeans", "want shirt"],
    "personal_query": ["my order", "order id", "track my order", "my profile", "my account", "order status"],
    "meta_count": ["how many products", "count items", "number of products", "total items available"],
}

# Cache embeddings for intent phrases
_embedder = FastEmbedder(local_model_path="./local_embedder")
_INTENT_VECS = {
    k: np.mean(_embedder.encode(v), axis=0) for k, v in INTENT_BANK.items()
}

def detect_intent(query: str) -> Tuple[str, float]:
    """Hybrid intent detection using regex + embedding similarity."""
    q = query.lower().strip()

    # === RULE-BASED PRE-FILTERS (High Priority) ===
    
    # Quick greetings
    if re.match(r"^(hi|hello|hey|hlw)\b", q):
        return ("greeting", 0.95)
    if re.match(r"^(thanks|thank you|bye|goodbye)\b", q):
        return ("thanks", 0.95)
    
    # Math/calculation queries (definitely general knowledge)
    if re.search(r"\b\d+\s*[+\-*/]\s*\d+\b", q) or re.search(r"\bwhat\s+is\s+\d+", q):
        return ("general_knowledge", 0.90)
    
    # Date/time queries
    if re.search(r"\b(what\s+date|what\s+day|today|current\s+date|what\s+time)\b", q):
        return ("general_knowledge", 0.90)
    
    # Famous people/factual queries ("who is", "what is")
    if re.search(r"\bwho\s+is\s+\w+\b", q) and not re.search(r"\b(product|item|seller|owner)\b", q):
        return ("general_knowledge", 0.85)
    
    # Clear product keywords (boost product detection)
    product_keywords = r"\b(jeans|t[- ]?shirts?|tshirts?|shirt|pants|shoes|electronics|phone|laptop|pc|computer|desktop|watch|bag|wallet|saree|dress|clothes|clothing|perfume|cream|skincare|skin|face|hair|beauty|cosmetics?|chair|table|furniture|mouse|speaker|backpack)\b"
    # Gift-style queries should also be treated as product intent
    gift_keywords = r"\b(gift|present|girlfriend|girlfrind|boyfriend|wife|husband|mom|mother|dad|father)\b"
    if re.search(product_keywords, q) or re.search(gift_keywords, q):
        if re.search(r"\b(need|want|looking for|show|find|buy|suggest|recommend|have)\b", q) or re.search(gift_keywords, q):
            return ("product_search", 0.90)
    
    # Price-related with context
    if re.search(r"\b(price|cost|under|below|less than|within|cheap|affordable)\b", q) and (re.search(product_keywords, q) or re.search(gift_keywords, q)):
        return ("price_filter", 0.88)
    
    # Order/personal queries
    if re.search(r"\b(my order|order id|track|profile|account|purchase)\b", q):
        return ("personal_query", 0.85)
    
    # Count queries about products
    if re.search(r"\b(how many|count|number of)\b", q) and re.search(r"\b(products|items|available|catalog|inventory)\b", q):
        return ("meta_count", 0.92)
    
    # === EMBEDDING SIMILARITY (Lower Priority) ===
    q_vec = _embedder.encode([q])[0]
    sims = {k: float(np.dot(q_vec, v) / (np.linalg.norm(v) + 1e-9)) for k, v in _INTENT_VECS.items()}

    best_intent = max(sims, key=sims.get)
    best_score = sims[best_intent]

    # Much stricter threshold for embedding-only classification
    if best_score < 0.65:  # Raised from 0.45
        return ("general_knowledge", best_score)
    
    # Additional check: if similarity suggests product intent but query has no product context
    if best_intent in ["product_search", "price_filter", "meta_count"]:
        if not re.search(product_keywords, q) and not re.search(r"\b(item|product|buy|sell|purchase|store|gift|present)\b", q):
            return ("general_knowledge", best_score * 0.5)  # Downgrade confidence
    
    return (best_intent, best_score)

