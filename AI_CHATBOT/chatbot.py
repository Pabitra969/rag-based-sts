## Upgraded Chatbot with Smart Intent Routing
# Routes queries intelligently instead of blindly applying LLM tone
# - Greeting → QuickResponder
# - Product/DB Query → Direct DB fetch + light LLM humanize (constrained)
# - Personalized Query → Full history + DB context + personalized response
# - General Knowledge → Fast LLM answer without retrieval

import asyncio
import time
import re
import json
from contextlib import contextmanager
from typing import Tuple
from datetime import datetime

from model_session_manager import ModelSessionManager
from fast_embedder import FastEmbedder
from retrievers.faiss_retriever import FaissRetriever
from retrievers.local_retriever import LocalRetriever
from memory.session_manager import SessionManager
from quick_responder.quick_responder import QuickResponder
from controller.intent_detector import detect_intent
from nlp_core import extract_fact
from web_search import search_web, compress_web_results, web_search_enabled

# ============ CONFIG ============
# Use the smaller local model by default for lower latency.
MODEL_PATH = "models/Llama-3.2-1B-Instruct-Q4_K_M.gguf"
FAISS_INDEX_PATH = "./faiss_index.index"
FAISS_META_PATH = "./faiss_meta.json"
DATA_CSV = "./data/products2.csv"

N_CTX = 768
TOP_K = 3
DEBUG = True

# System preambles for different intents
SYSTEM_PREAMBLE_PRODUCT = """You are Aria, a specialized product support chatbot for a shopping catalog.
Your job is to answer product questions clearly, consistently, and professionally using only the provided product context.
Rules:
- Never invent product names, prices, categories, stock, features, delivery times, discounts, or policies.
- If the requested detail is missing, say: "I don't have that product detail in the current catalog context."
- Answer the user's exact product question first.
- When listing matching products, use one line per product in this exact format:
  Product Name | ₹Price | Category. Short description
- Show at most 3 products unless the user explicitly asks for more.
- When the user asks to know more about a product, explain the available details in short plain sentences from the context only.
- Keep the tone calm, helpful, and specialized for product support.
- Do not repeat abusive or vulgar wording from the user. Refer to it neutrally as "that language" if needed."""

SYSTEM_PREAMBLE_PERSONALIZED = """You are Aria, a specialized customer support chatbot.
Use the recent conversation and provided product context to help the customer with consistent, practical answers.
Rules:
- Be warm, clear, and factual.
- Never invent account details, order details, shipment updates, refund outcomes, preferences, or product facts.
- If the customer asks for order status, tracking, account status, refund status, or similar information that is not verifiable from the current data, say clearly that you cannot verify it from the current information and ask for the exact missing detail such as order ID.
- If context is missing, say so directly instead of guessing.
- Answer in 2 to 4 concise sentences when possible.
- Do not repeat abusive or vulgar wording from the user. Refer to it neutrally as "that language" if needed."""

SYSTEM_PREAMBLE_GENERAL_FAST = """You are Aria, a specialized customer support chatbot.
Give consistent, direct, well-formed answers in 2 to 4 short sentences.
Rules:
- Answer the question directly first.
- Be clear, calm, and accurate.
- If you do not have enough verified information, say: "I don't have enough verified information for that."
- Do not pretend to know live facts like weather, news, stock prices, traffic, or private customer data unless verified context is provided.
- Do not drift into unrelated topics, roleplay, hype, or filler.
- Do not repeat abusive or vulgar wording from the user. Refer to it neutrally as "that language" if needed.
- Mention products only when the user asks about products."""

SYSTEM_PREAMBLE_WEB_SEARCH = """You are Aria, a specialized customer support chatbot.
Answer using only the provided web search results.
Rules:
- Write a direct 2 to 4 sentence summary in plain language.
- Prefer the most concrete and recent-looking result from the provided snippets.
- If the snippets are weak, missing, or conflicting, say that briefly instead of guessing.
- Do not mention any fact that is not supported by the provided web search results.
- Keep the answer concise, neutral, and useful.
- Do not repeat abusive or vulgar wording from the user. Refer to it neutrally as "that language" if needed."""

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


def normalize_support_reply(text: str) -> str:
    cleaned = clean_text(text)
    if not cleaned:
        return ""
    cleaned = re.sub(r"\b(i\s+am|i'm)\s+sorry\b[:,]?\s*", "", cleaned, count=1, flags=re.I)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned

PRODUCT_HINT_RE = re.compile(
    r"\b("
    r"jeans|t[- ]?shirts?|tshirts?|shirt|pants|trouser|shorts|shoes|sneakers|"
    r"electronics|phone|laptop|pc|computer|desktop|watch|bag|wallet|belt|saree|"
    r"dress|kurti|kurta|sweater|jacket|hoodie|leggings|clothes|clothing|"
    r"perfume|cream|skincare|skin|face|hair|beauty|cosmetics?|toothbrush|trimmer|"
    r"chair|table|furniture|mouse|speaker|headphones?|headset|earphones?|earbuds|"
    r"backpack|keyboard|webcam|ssd|tablet|drawing|graphic|plug|charger|router|screen protector|"
    r"cooler|purifier|heater|fan|geyser|refrigerator|microwave|dishwasher|"
    r"product|item|catalog|inventory|price|cost"
    r")\b",
    re.I,
)

HELP_HINT_RE = re.compile(r"\b(help|what can you do|how can you help)\b", re.I)
LIVE_INFO_HINT_RE = re.compile(
    r"\b("
    r"temperature|weather|forecast|rain|humidity|wind|climate|"
    r"news|headline|stock|share price|traffic|score|match score"
    r")\b",
    re.I,
)
DATE_QUERY_RE = re.compile(
    r"\b("
    r"what('?s| is)\s+(the\s+)?date|"
    r"what\s+day\s+is\s+it|"
    r"which\s+day\s+is\s+it|"
    r"today('?s| is)\s+date|"
    r"current\s+date"
    r")\b",
    re.I,
)
TIME_QUERY_RE = re.compile(
    r"\b("
    r"what('?s| is)\s+(the\s+)?time|"
    r"current\s+time|"
    r"time\s+now"
    r")\b",
    re.I,
)
WEB_FALLBACK_HINT_RE = re.compile(
    r"\b("
    r"latest|current|currently|today|recent|news|headline|weather|forecast|temperature|"
    r"stock|score|traffic|time in|date in|search web|"
    r"who is the current|what is the current|"
    r"who is the (prime minister|president|ceo|governor|mayor)"
    r")\b",
    re.I,
)

def _quick_intent_reply(query: str, intent: str) -> str:
    canned = quick.get_response(query)
    if canned:
        return canned
    if intent == "thanks":
        return "You're welcome."
    if intent == "farewell":
        return "Goodbye."
    return "Hello! How can I help you?"

def _extract_user_name_from_history(user_id: str) -> str:
    conv = sessions.get_session(user_id) or []
    for msg in reversed(conv):
        if msg.get("role") != "user":
            continue
        match = re.search(r"\bmy name is\s+([A-Za-z][A-Za-z\s'-]{0,30})", msg.get("text", ""), re.I)
        if match:
            return match.group(1).strip()
    return ""

def _fast_general_reply(user_id: str, query: str, voice_mode: bool) -> str:
    q = query.lower().strip()
    math_match = re.fullmatch(r"\s*(\d+)\s*([+\-*/])\s*(\d+)\s*", q)

    if math_match:
        a, op, b = math_match.groups()
        a = int(a)
        b = int(b)
        if op == "+":
            return f"{a} plus {b} equals {a + b}."
        if op == "-":
            return f"{a} minus {b} equals {a - b}."
        if op == "*":
            return f"{a} times {b} equals {a * b}."
        if b != 0:
            return f"{a} divided by {b} equals {round(a / b, 2)}."
        return "Division by zero is not allowed."

    if re.search(r"\b(who are you|what('?s| is) your name)\b", q):
        return "I'm Aria, your AI support assistant."

    if re.search(r"\bwhat('?s| is) my name\b", q):
        known_name = _extract_user_name_from_history(user_id)
        if known_name:
            return f"Your name is {known_name}."
        return "You haven't told me your name yet."

    if re.search(r"\b(my name is)\b", q):
        match = re.search(r"\bmy name is\s+([A-Za-z][A-Za-z\s'-]{0,30})", query, re.I)
        if match:
            return f"Nice to meet you, {match.group(1).strip()}."

    if LIVE_INFO_HINT_RE.search(q):
        if not web_search_enabled():
            return "I can't check live weather or other real-time updates in this offline setup yet."
        return ""

    if DATE_QUERY_RE.search(q):
        return datetime.now().strftime("Today is %A, %d %B %Y.")

    if TIME_QUERY_RE.search(q):
        return datetime.now().strftime("The time is %I:%M %p.")

    if re.search(r"\b(how are you|are you there|can you hear me)\b", q):
        return "I'm here and listening."

    if HELP_HINT_RE.search(q):
        return "I can help with product questions, catalog details, and general support answers."

    if re.search(r"\b(who made you|who created you)\b", q):
        return "I'm Aria, a local AI support assistant built for this project."

    return ""


def should_use_web_search(query: str) -> bool:
    q = query.lower().strip()
    if not web_search_enabled():
        return False
    return bool(WEB_FALLBACK_HINT_RE.search(q))


async def get_web_context(query: str, voice_mode: bool = False) -> str:
    max_results = 2 if voice_mode else 3
    max_chars = 360 if voice_mode else 560
    results = await search_web(query, max_results=max_results)
    if not results:
        return ""
    return compress_web_results(results, max_chars=max_chars)

async def _persist_conversation_state(user_id: str, save_model_context: bool = False):
    await asyncio.to_thread(sessions.flush_pending, user_id)
    if save_model_context:
        await model_manager.save_context_async()

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
    local_items = await local.search(query, top_k=top_k)
    if not local_items:
        if DEBUG:
            print("[DEBUG] Local catalog returned 0 items")
        return []
    try:
        emb = list(embedder.encode_cached(query))
        items = await faiss.search_with_embedding(emb, top_k=top_k)
        if local_items:
            seen = set()
            merged = []
            for item in local_items + items:
                m = item.get("metadata", {}) or {}
                key = str(m.get("id") or m.get("sku") or m.get("title") or item.get("content", ""))
                if key in seen:
                    continue
                seen.add(key)
                merged.append(item)
                if len(merged) >= top_k:
                    break
            items = merged
        if DEBUG:
            print(f"[DEBUG] Faiss returned {len(items)} items")
        return items
    except Exception as e:
        if DEBUG:
            print(f"[DEBUG] Faiss failed: {e} — fallback to local")
        return local_items

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
    conv = sessions.get_session(user_id) or []
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
    4. General → Fast LLM answer
    """
    timings = {}
    start = time.time()
    retrieval_k = TOP_K
    product_max_tokens = 110 if voice_mode else 180
    personalized_max_tokens = 110 if voice_mode else 220
    general_max_tokens = 72 if voice_mode else 64
    web_max_tokens = 88 if voice_mode else 120

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

    fast_reply = _fast_general_reply(user_id, query, voice_mode)
    if fast_reply:
        sessions.add_user_msg(user_id, query, persist_long=False)
        sessions.add_bot_msg(user_id, fast_reply, persist_long=False)
        timings["total"] = round(time.time() - start, 3)
        if DEBUG:
            print(f"[ROUTE] FAST GENERAL → Early rule-based reply")
            print(f"[TIMING] {timings}")
        return fast_reply

    # ===== DETECT INTENT =====
    with timed("intent_detect", timings):
        intent, confidence = detect_intent(query)

    if DEBUG:
        print(f"[INTENT] {intent} (confidence: {confidence:.2f})")

    if intent in ["greeting", "thanks", "farewell"]:
        reply = _quick_intent_reply(query, intent)
        sessions.add_bot_msg(user_id, reply)
        timings["total"] = round(time.time() - start, 3)
        if DEBUG:
            print(f"[ROUTE] QUICK INTENT → Direct reply")
            print(f"[TIMING] {timings}")
        return reply

    # ===== PATH 2: PRODUCT/DATABASE QUERIES =====
    if intent in ["product_search", "price_filter", "meta_count"]:
        if DEBUG:
            print(f"[ROUTE] PRODUCT QUERY → Direct DB retrieval + Light LLM humanize")

        sessions.add_user_msg(user_id, query, persist_long=True)

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
                    temperature=0.15,
                    max_tokens=product_max_tokens
                )
            reply = normalize_support_reply(reply)
            if not reply:
                reply = "I couldn't find a matching product in the current catalog. Please be a little more specific."

        sessions.add_bot_msg(user_id, reply, persist_long=True)
        if not voice_mode:
            with timed("persist", timings):
                await _persist_conversation_state(user_id, save_model_context=True)
        else:
            with timed("persist", timings):
                await _persist_conversation_state(user_id, save_model_context=False)
        timings["total"] = round(time.time() - start, 3)
        if DEBUG:
            print(f"[TIMING] {timings}")
        return reply

    # ===== PATH 3: PERSONALIZED QUERIES =====
    if intent in ["personal_query"]:
        if DEBUG:
            print(f"[ROUTE] PERSONALIZED QUERY → Full history + DB context + personalized response")

        sessions.add_user_msg(user_id, query, persist_long=True)

        # Early handling for order-related queries (avoid hallucination)
        ql = query.lower()
        if re.search(r"\border(\b|\s)|track|order\s*status", ql):
            reply = (
                "I can help with order status, but I cannot verify it from the current information. "
                "Please share your order ID, such as ORD1234."
            )
            sessions.add_bot_msg(user_id, reply, persist_long=True)
            if not voice_mode:
                with timed("persist", timings):
                    await _persist_conversation_state(user_id, save_model_context=True)
            else:
                with timed("persist", timings):
                    await _persist_conversation_state(user_id, save_model_context=False)
            timings["total"] = round(time.time() - start, 3)
            if DEBUG:
                print(f"[ROUTE] ORDER STATUS → Template response")
                print(f"[TIMING] {timings}")
            return reply

        with timed("retrieval", timings):
            results = await retrieve(query, top_k=retrieval_k)

        context = build_product_context(results, max_items=3)
        history = get_recent_history(user_id, max_turns=2) if voice_mode else get_full_history(user_id)

        with timed("llm_personalized", timings):
            reply = await model_manager.generate_reply(
                SYSTEM_PREAMBLE_PERSONALIZED,
                query,
                context_text=context,
                history_text=history,  # full history for personalization
                temperature=0.25,
                max_tokens=personalized_max_tokens
            )
        reply = normalize_support_reply(reply)
        if not reply:
            reply = "I don't have enough verified information for that yet. Please share a bit more detail."

        sessions.add_bot_msg(user_id, reply, persist_long=True)
        if not voice_mode:
            with timed("persist", timings):
                await _persist_conversation_state(user_id, save_model_context=True)
        else:
            with timed("persist", timings):
                await _persist_conversation_state(user_id, save_model_context=False)
        timings["total"] = round(time.time() - start, 3)
        if DEBUG:
            print(f"[TIMING] {timings}")
        return reply

    # ===== PATH 4: GENERAL KNOWLEDGE QUERIES =====
    if DEBUG:
        print(f"[ROUTE] GENERAL QUERY → Fast LLM path")

    sessions.add_user_msg(user_id, query, persist_long=False)
    timings["retrieval"] = 0.0

    if should_use_web_search(query):
        if DEBUG:
            print("[ROUTE] GENERAL QUERY → Web fallback")

        with timed("web_search", timings):
            web_context = await get_web_context(query, voice_mode=voice_mode)

        if web_context:
            with timed("llm_general_web", timings):
                reply = await model_manager.generate_reply(
                    SYSTEM_PREAMBLE_WEB_SEARCH,
                    query,
                    web_results_text=web_context,
                    temperature=0.1,
                    max_tokens=web_max_tokens,
                )
            reply = normalize_support_reply(reply)
            if reply:
                sessions.add_bot_msg(user_id, reply, persist_long=False)
                timings["persist"] = 0.0
                timings["total"] = round(time.time() - start, 3)
                if DEBUG:
                    print(f"[TIMING] {timings}")
                return reply

        reply = "I don't have a reliable verified answer for that right now."
        sessions.add_bot_msg(user_id, reply, persist_long=False)
        timings["persist"] = 0.0
        timings["total"] = round(time.time() - start, 3)
        if DEBUG:
            print(f"[TIMING] {timings}")
        return reply

    with timed("llm_general_fast", timings):
        reply = await model_manager.generate_fast_reply(
            SYSTEM_PREAMBLE_GENERAL_FAST,
            query,
            temperature=0.10 if voice_mode else 0.15,
            max_tokens=general_max_tokens
        )
    reply = normalize_support_reply(reply)
    if not reply:
        reply = "I don't have enough verified information for that."

    sessions.add_bot_msg(user_id, reply, persist_long=False)
    timings["persist"] = 0.0
    timings["total"] = round(time.time() - start, 3)
    if DEBUG:
        print(f"[TIMING] {timings}")
    return reply

async def answer_query_stream_async(user_id: str, query: str, voice_mode: bool = False):
    timings = {}
    start = time.time()
    retrieval_k = TOP_K
    product_max_tokens = 110 if voice_mode else 180
    personalized_max_tokens = 110 if voice_mode else 220
    general_max_tokens = 96 if voice_mode else 140
    web_max_tokens = 100 if voice_mode else 120

    with timed("quick", timings):
        quick_reply = quick.get_response(query)
    if quick_reply:
        sessions.add_bot_msg(user_id, quick_reply)
        timings["total"] = round(time.time() - start, 3)
        if DEBUG:
            print(f"[ROUTE] GREETING → Quick Responder")
            print(f"[TIMING] {timings}")
        yield quick_reply
        return

    fast_reply = _fast_general_reply(user_id, query, voice_mode)
    if fast_reply:
        sessions.add_user_msg(user_id, query, persist_long=False)
        sessions.add_bot_msg(user_id, fast_reply, persist_long=False)
        timings["total"] = round(time.time() - start, 3)
        if DEBUG:
            print(f"[ROUTE] FAST GENERAL → Early rule-based reply")
            print(f"[TIMING] {timings}")
        yield fast_reply
        return

    with timed("intent_detect", timings):
        intent, confidence = detect_intent(query)

    if DEBUG:
        print(f"[INTENT] {intent} (confidence: {confidence:.2f})")

    if intent in ["greeting", "thanks", "farewell"]:
        reply = _quick_intent_reply(query, intent)
        sessions.add_bot_msg(user_id, reply)
        timings["total"] = round(time.time() - start, 3)
        if DEBUG:
            print(f"[ROUTE] QUICK INTENT → Direct reply")
            print(f"[TIMING] {timings}")
        yield reply
        return

    if intent in ["product_search", "price_filter", "meta_count"]:
        if DEBUG:
            print(f"[ROUTE] PRODUCT QUERY → Direct DB retrieval + Light LLM humanize")

        sessions.add_user_msg(user_id, query, persist_long=True)

        with timed("retrieval", timings):
            results = await retrieve(query, top_k=retrieval_k)

        with timed("extract", timings):
            fact = extract_fact(query, results)

        if fact:
            reply = fact
            sessions.add_bot_msg(user_id, reply, persist_long=True)
            if not voice_mode:
                with timed("persist", timings):
                    await _persist_conversation_state(user_id, save_model_context=True)
            else:
                with timed("persist", timings):
                    await _persist_conversation_state(user_id, save_model_context=False)
            timings["total"] = round(time.time() - start, 3)
            if DEBUG:
                print(f"[TIMING] {timings}")
            yield reply
            return

        context = build_product_context(results)
        collected = []
        with timed("llm_product", timings):
            async for chunk in model_manager.stream_reply(
                SYSTEM_PREAMBLE_PRODUCT,
                query,
                context_text=context,
                history_text="",
                temperature=0.15,
                max_tokens=product_max_tokens,
            ):
                collected.append(chunk)
                yield chunk

        reply = normalize_support_reply("".join(collected)) or (
            "I couldn't find a matching product in the current catalog. Please be a little more specific."
        )
        sessions.add_bot_msg(user_id, reply, persist_long=True)
        if not voice_mode:
            with timed("persist", timings):
                await _persist_conversation_state(user_id, save_model_context=True)
        else:
            with timed("persist", timings):
                await _persist_conversation_state(user_id, save_model_context=False)
        timings["total"] = round(time.time() - start, 3)
        if DEBUG:
            print(f"[TIMING] {timings}")
        return

    if intent in ["personal_query"]:
        if DEBUG:
            print(f"[ROUTE] PERSONALIZED QUERY → Full history + DB context + personalized response")

        sessions.add_user_msg(user_id, query, persist_long=True)
        ql = query.lower()
        if re.search(r"\border(\b|\s)|track|order\s*status", ql):
            reply = (
                "I can help with order status, but I cannot verify it from the current information. "
                "Please share your order ID, such as ORD1234."
            )
            sessions.add_bot_msg(user_id, reply, persist_long=True)
            if not voice_mode:
                with timed("persist", timings):
                    await _persist_conversation_state(user_id, save_model_context=True)
            else:
                with timed("persist", timings):
                    await _persist_conversation_state(user_id, save_model_context=False)
            timings["total"] = round(time.time() - start, 3)
            if DEBUG:
                print(f"[ROUTE] ORDER STATUS → Template response")
                print(f"[TIMING] {timings}")
            yield reply
            return

        with timed("retrieval", timings):
            results = await retrieve(query, top_k=retrieval_k)
        context = build_product_context(results, max_items=3)
        history = get_recent_history(user_id, max_turns=2) if voice_mode else get_full_history(user_id)
        collected = []
        with timed("llm_personalized", timings):
            async for chunk in model_manager.stream_reply(
                SYSTEM_PREAMBLE_PERSONALIZED,
                query,
                context_text=context,
                history_text=history,
                temperature=0.25,
                max_tokens=personalized_max_tokens,
            ):
                collected.append(chunk)
                yield chunk

        reply = normalize_support_reply("".join(collected)) or (
            "I don't have enough verified information for that yet. Please share a bit more detail."
        )
        sessions.add_bot_msg(user_id, reply, persist_long=True)
        if not voice_mode:
            with timed("persist", timings):
                await _persist_conversation_state(user_id, save_model_context=True)
        else:
            with timed("persist", timings):
                await _persist_conversation_state(user_id, save_model_context=False)
        timings["total"] = round(time.time() - start, 3)
        if DEBUG:
            print(f"[TIMING] {timings}")
        return

    if DEBUG:
        print(f"[ROUTE] GENERAL QUERY → Fast LLM path")

    sessions.add_user_msg(user_id, query, persist_long=False)
    timings["retrieval"] = 0.0
    collected = []

    if should_use_web_search(query):
        if DEBUG:
            print("[ROUTE] GENERAL QUERY → Web fallback")

        with timed("web_search", timings):
            web_context = await get_web_context(query, voice_mode=voice_mode)

        if web_context:
            with timed("llm_general_web", timings):
                async for chunk in model_manager.stream_reply(
                    SYSTEM_PREAMBLE_WEB_SEARCH,
                    query,
                    web_results_text=web_context,
                    temperature=0.1,
                    max_tokens=web_max_tokens,
                    top_p=0.85,
                    stop_tokens=[
                        "\nCustomer:",
                        "\nSystem:",
                        "\nSupport Agent:",
                    ],
                ):
                    collected.append(chunk)
                    yield chunk

            reply = normalize_support_reply("".join(collected))
            if reply:
                sessions.add_bot_msg(user_id, reply, persist_long=False)
                timings["persist"] = 0.0
                timings["total"] = round(time.time() - start, 3)
                if DEBUG:
                    print(f"[TIMING] {timings}")
                return

        reply = "I don't have a reliable verified answer for that right now."
        sessions.add_bot_msg(user_id, reply, persist_long=False)
        timings["persist"] = 0.0
        timings["total"] = round(time.time() - start, 3)
        if DEBUG:
            print(f"[TIMING] {timings}")
        yield reply
        return

    with timed("llm_general_fast", timings):
        async for chunk in model_manager.stream_reply(
            SYSTEM_PREAMBLE_GENERAL_FAST,
            query,
            temperature=0.10 if voice_mode else 0.15,
            max_tokens=general_max_tokens,
            top_p=0.9,
            stop_tokens=[
                "\nCustomer:",
                "\nSystem:",
                "\nSupport Agent:",
            ],
        ):
            collected.append(chunk)
            yield chunk

    reply = normalize_support_reply("".join(collected)) or "I don't have enough verified information for that."
    sessions.add_bot_msg(user_id, reply, persist_long=False)
    timings["persist"] = 0.0
    timings["total"] = round(time.time() - start, 3)
    if DEBUG:
        print(f"[TIMING] {timings}")

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
