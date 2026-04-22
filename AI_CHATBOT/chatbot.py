## Upgraded Chatbot with Smart Intent Routing
# Routes queries intelligently instead of blindly applying LLM tone
# - Greeting → QuickResponder
# - Product/DB Query → Direct DB fetch + light LLM humanize (constrained)
# - Personalized Query → Full history + DB context + personalized response
# - General Knowledge → Model answer + DB product suggestions

import asyncio
import time
import re
from contextlib import contextmanager
from typing import Tuple

from model_session_manager import ModelSessionManager
from fast_embedder import FastEmbedder
from retrievers.faiss_retriever import FaissRetriever
from retrievers.local_retriever import LocalRetriever
from memory.session_manager import SessionManager
from quick_responder.quick_responder import QuickResponder
from controller.intent_detector import detect_intent
from nlp_core import extract_fact

# ============ CONFIG ============
MODEL_PATH = "models/Phi-3-mini-4k-instruct-q4.gguf"
FAISS_INDEX_PATH = "./faiss_index.index"
FAISS_META_PATH = "./faiss_meta.json"
DATA_CSV = "./data/products2.csv"

N_CTX = 1024
TOP_K = 3
DEBUG = True

# System preambles for different intents
SYSTEM_PREAMBLE_PRODUCT = """You are Provis Technologies' customer support AI.
You provide concise, factual, datasource-based answers about products.
Use only the provided product context. Never invent products or prices.
If the requested product is not available in the database, say so clearly and suggest related items from the database only.
Keep answers short and professional."""

SYSTEM_PREAMBLE_PERSONALIZED = """You are Provis Technologies' customer support AI.
You help customers with personalized assistance based on their history and preferences.
Be friendly, remember context from their previous interactions.
Reference their past questions when relevant."""

SYSTEM_PREAMBLE_GENERAL = """You are Provis Technologies' customer support AI.
Answer general questions normally.
If you provide product suggestions, use ONLY the provided suggestions list. Never invent product names or prices.
If no suggestions are provided, do not mention any products.
Be helpful and concise."""

# ============ TIMING ============
@contextmanager
def timed(label: str, timings: dict):
    start = time.time()
    yield
    timings[label] = round(time.time() - start, 3)

def clean_text(t: str):
    if not t:
        return ""
    return t.replace("Assistant:", "").replace("User:", "").strip()

# ============ MODULE LOADING ============
print("🔧 Loading model manager and modules...")
model_manager = ModelSessionManager(model_path=MODEL_PATH, n_ctx=N_CTX, verbose=False)
embedder = FastEmbedder(persist_cache=True)
faiss = FaissRetriever(index_path=FAISS_INDEX_PATH, meta_path=FAISS_META_PATH, debug=False)
local = LocalRetriever(DATA_CSV, embedder=embedder, debug=False)
sessions = SessionManager(short_turns=3, long_memory_path="memory/long_memory.json", debug=DEBUG)
quick = QuickResponder(beautify_with_model=False)
print("✅ Modules loaded. Chatbot ready.")

# ============ RETRIEVAL ============
async def retrieve(query: str, top_k: int = TOP_K):
    try:
        emb = list(embedder.encode_cached(query))
        items = await faiss.search_with_embedding(emb, top_k=top_k)
        if DEBUG:
            print(f"[DEBUG] Faiss returned {len(items)} items")
        return items
    except Exception as e:
        if DEBUG:
            print(f"[DEBUG] Faiss failed: {e} — fallback to local")
        return await local.search(query, top_k=top_k)

# ============ HISTORY HELPERS ============
def get_recent_history(user_id: str, max_turns: int = 3) -> str:
    """Return last N conversation turns."""
    conv = sessions.get_session(user_id) or []
    recent = conv[-(max_turns * 2):]
    lines = []
    for msg in recent:
        role = "User" if msg["role"] == "user" else "Assistant"
        lines.append(f"{role}: {msg['text']}")
    return "\n".join(lines)

def get_full_history(user_id: str) -> str:
    """Return entire conversation history."""
    conv = sessions.get_recent_short(user_id) or []
    lines = []
    for msg in conv:
        role = "User" if msg["role"] == "user" else "Assistant"
        lines.append(f"{role}: {msg['text']}")
    return "\n".join(lines)

def build_product_context(results, max_items: int = 4) -> str:
    """Format product retrieval results into context (dominant category to avoid mixing)."""
    if not results:
        return ""
    # Determine dominant category
    cats = []
    for r in results:
        m = r.get("metadata", {}) or {}
        cats.append(m.get("category", "") or "uncategorized")
    dominant = None
    if cats:
        dominant = max(set(cats), key=cats.count)
    # Build lines, prefer dominant category
    lines = []
    for r in results:
        m = r.get("metadata", {}) or {}
        cat = m.get("category", "") or "uncategorized"
        if dominant and cat != dominant:
            continue
        title = m.get("title") or m.get("name") or ""
        price = m.get("price", "")
        desc = m.get("description", "") or (r.get("content") or "")[:160]
        lines.append(f"- {title} {('₹'+str(price)) if price else ''} [{cat}] - {desc}".strip())
        if len(lines) >= max_items:
            break
    if not lines:
        # fallback without filtering
        for r in results[:max_items]:
            m = r.get("metadata", {}) or {}
            title = m.get("title") or m.get("name") or ""
            price = m.get("price", "")
            desc = m.get("description", "") or (r.get("content") or "")[:160]
            lines.append(f"- {title} {('₹'+str(price)) if price else ''} [{m.get('category','')}] - {desc}".strip())
    return "\n".join(lines) if lines else "No product info found."

# ============ INTENT ROUTING LOGIC ============
async def answer_query_async(user_id: str, query: str, voice_mode: bool = False) -> str:
    """
    Smart routing based on query intent:
    1. Greeting → QuickResponder (fast)
    2. Product/DB → Direct retrieval + constrained LLM humanize
    3. Personalized → Full history + retrieval + personalized response
    4. General → LLM answer + product suggestions
    """
    timings = {}
    start = time.time()
    retrieval_k = 2 if voice_mode else TOP_K
    product_max_tokens = 48 if voice_mode else 80
    personalized_max_tokens = 72 if voice_mode else 120
    general_max_tokens = 64 if voice_mode else 120
    general_history_turns = 1 if voice_mode else 2

    # ===== PATH 1: QUICK RESPONSES (GREETINGS) =====
    with timed("quick", timings):
        q = quick.get_response(query)
    if q:
        sessions.add_bot_msg(user_id, q)
        timings["total"] = round(time.time() - start, 3)
        if DEBUG:
            print(f"[ROUTE] GREETING → Quick Responder")
            print(f"[TIMING] {timings}")
        return q

    sessions.add_user_msg(user_id, query)

    # ===== DETECT INTENT =====
    with timed("intent_detect", timings):
        intent, confidence = detect_intent(query)

    if DEBUG:
        print(f"[INTENT] {intent} (confidence: {confidence:.2f})")

    # ===== PATH 2: PRODUCT/DATABASE QUERIES =====
    if intent in ["product_search", "price_filter", "meta_count"]:
        if DEBUG:
            print(f"[ROUTE] PRODUCT QUERY → Direct DB retrieval + Light LLM humanize")

        with timed("retrieval", timings):
            results = await retrieve(query, top_k=retrieval_k)

        # Try deterministic extraction first
        with timed("extract", timings):
            fact = extract_fact(query, results)

        if fact:
            # Deterministic fact found — return directly (no LLM to avoid drift)
            reply = fact
        else:
            # No deterministic fact, use context with LLM (still constrained)
            context = build_product_context(results)
            with timed("llm_product", timings):
                reply = await model_manager.generate_reply(
                    SYSTEM_PREAMBLE_PRODUCT,
                    query,
                    context_text=context,
                    history_text="",
                    temperature=0.25,
                    max_tokens=product_max_tokens
                )
            if not reply.strip():
                reply = "I couldn't find that product. Could you be more specific?"

        sessions.add_bot_msg(user_id, reply)
        if not voice_mode:
            with timed("save", timings):
                await model_manager.save_context_async()
        timings["total"] = round(time.time() - start, 3)
        if DEBUG:
            print(f"[TIMING] {timings}")
        return reply

    # ===== PATH 3: PERSONALIZED QUERIES =====
    if intent in ["personal_query"]:
        if DEBUG:
            print(f"[ROUTE] PERSONALIZED QUERY → Full history + DB context + personalized response")

        # Early handling for order-related queries (avoid hallucination)
        ql = query.lower()
        if re.search(r"\border(\b|\s)|track|order\s*status", ql):
            reply = "To check your order, please share your order ID (e.g., ORD1234). I'll track the status for you."
            sessions.add_bot_msg(user_id, reply)
            if not voice_mode:
                with timed("save", timings):
                    await model_manager.save_context_async()
            timings["total"] = round(time.time() - start, 3)
            if DEBUG:
                print(f"[ROUTE] ORDER STATUS → Template response")
                print(f"[TIMING] {timings}")
            return reply

        with timed("retrieval", timings):
            results = await retrieve(query, top_k=retrieval_k)

        context = build_product_context(results, max_items=3)
        history = get_full_history(user_id)

        with timed("llm_personalized", timings):
            reply = await model_manager.generate_reply(
                SYSTEM_PREAMBLE_PERSONALIZED,
                query,
                context_text=context,
                history_text=history,  # full history for personalization
                temperature=0.4,
                max_tokens=personalized_max_tokens
            )
        if not reply.strip():
            reply = "I'm here to help! Could you tell me more about what you need?"

        sessions.add_bot_msg(user_id, reply)
        if not voice_mode:
            with timed("save", timings):
                await model_manager.save_context_async()
        timings["total"] = round(time.time() - start, 3)
        if DEBUG:
            print(f"[TIMING] {timings}")
        return reply

    # ===== PATH 4: GENERAL KNOWLEDGE QUERIES =====
    if DEBUG:
        print(f"[ROUTE] GENERAL QUERY → Model answer + product suggestions")

    with timed("retrieval", timings):
        results = await retrieve(query, top_k=retrieval_k)

    # Build light product suggestions (strictly from DB — never invent)
    suggestions = []
    for r in (results or [])[:2]:
        m = r.get('metadata', {}) or {}
        t = m.get('title') or 'Product'
        p = m.get('price')
        suggestions.append(f"- {t} (₹{p})")
    product_suggestions = "\n".join(suggestions) if suggestions else ""

    context = (
        f"Only suggest from this list:\n{product_suggestions}"
        if product_suggestions
        else ""
    )

    history = get_recent_history(user_id, max_turns=general_history_turns)

    with timed("llm_general", timings):
        reply = await model_manager.generate_reply(
            SYSTEM_PREAMBLE_GENERAL,
            query,
            context_text=context,
            history_text=history,
            temperature=0.5,
            max_tokens=general_max_tokens
        )
    if not reply.strip():
        reply = "That's an interesting question! I'm not sure, but feel free to ask about our products."

    sessions.add_bot_msg(user_id, reply)
    if not voice_mode:
        with timed("save", timings):
            await model_manager.save_context_async()
    timings["total"] = round(time.time() - start, 3)
    if DEBUG:
        print(f"[TIMING] {timings}")
    return reply

# ============ CLI ============
async def main():
    print("🤖 Customer Support Assistant (Smart Intent Routing V2)")
    user_id = "demo_user"
    while True:
        try:
            query = input("\n🧑 You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            break
        if not query:
            continue
        if query.lower() in {"exit", "quit"}:
            print("👋 Session ended.")
            break
        print(await answer_query_async(user_id, query))

if __name__ == "__main__":
    asyncio.run(main())
