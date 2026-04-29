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

N_CTX = 2048
TOP_K = 5
DEBUG = True

# System preambles for different intents
SYSTEM_PREAMBLE_PRODUCT = """You are Aria, a helpful product assistant for an online shopping catalog.
Answer product questions using ONLY the provided Product Information below. Never invent products, prices, or details.

Format rules:
- List products as: Product Name | ₹Price | Category. Description
- Show at most 3 products unless asked for more.
- For product details, show the product card first, then a short paragraph about features and benefits.
- If the product is not found, say: "I don't have that product in our catalog."

Behavior rules:
- Answer directly. Do not say "I am Aria" or refer to yourself.
- Do not discuss topics outside the catalog.
- Keep answers concise: 2-5 sentences max.

Example:
User: show me track pants
Assistant: Here are some options:
Track Pants | ₹799 | clothing. Moisture-wicking polyester track pants with drawstring waist
Yoga Pants | ₹1099 | clothing. 4-way stretch yoga pants with hidden waistband pocket

Example:
User: tell me about the track pants
Assistant: Track Pants | ₹799 | clothing. Moisture-wicking polyester track pants with drawstring waist

These track pants are ideal for workouts and casual wear. Made from moisture-wicking polyester, they keep you dry during exercise. The drawstring waist offers an adjustable fit, available in sizes S to XXL."""

SYSTEM_PREAMBLE_PERSONALIZED = """You are Aria, a helpful shopping assistant.
Use the Recent Conversation and Product Information provided below to give consistent, personalized answers.

Rules:
- Be warm, clear, and factual. Use the conversation history to stay consistent.
- NEVER invent order details, tracking info, refund status, or account data.
- If the user asks about orders or account info you cannot verify, say: "I can't verify that right now. Could you share your order ID?"
- If context is missing, say so honestly instead of guessing.
- Keep answers to 2-4 sentences.
- Answer directly. Do not say "I am Aria" or refer to yourself."""

SYSTEM_PREAMBLE_GENERAL_FAST = """You are Aria, a knowledgeable AI assistant.
Answer general knowledge questions accurately and concisely in 2-4 sentences.

Rules:
- Answer the question directly first. No preamble.
- Be accurate. If unsure, say: "I'm not confident about that."
- Do NOT guess about live/real-time data (weather, news, scores, stocks) unless provided in context.
- Do NOT mention products or shopping unless the user asks.
- Keep the tone natural and conversational.
- Answer directly. Do not say "I am Aria" or refer to yourself."""

SYSTEM_PREAMBLE_WEB_SEARCH = """You are Aria, a knowledgeable AI assistant.
Answer the user's question using ONLY the Web Search Results provided below.

STRICT rules:
- Write your answer in 2-4 sentences MAX. Be concise.
- Summarize in your OWN words. Do NOT copy-paste snippets.
- Do NOT include URLs, links, source names, or "Wikipedia" in your answer.
- If the search results don't contain the answer, say: "I couldn't find a clear answer for that."
- ONLY state facts that appear in the search results. Do not add your own knowledge.
- Answer directly. Do not say "I am Aria" or refer to yourself."""

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
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned).strip()
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
    r"blender|mixer|kettle|cooker|pressure cooker|frying pan|knife|"
    r"chopping board|tiffin|casserole|idli|grater|rolling pin|roti maker|"
    r"spice rack|mixing bowl|oil dispenser|food storage|container|"
    r"stool|bar stool|recliner|bookshelf|sofa|mattress|pillow|bedsheet|curtain|"
    r"iron|washing machine|vacuum|air conditioner|ac|tv|television|"
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
    r"who is the (prime minister|president|ceo|governor|mayor|cm|chief minister)"
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
    return web_search_enabled()


async def get_web_context(query: str, voice_mode: bool = False) -> str:
    import urllib.request
    import json
    import re
    import asyncio

    if re.search(r"\b(weather|temperature|forecast|rain|humidity|wind)\b", query, re.I):
        try:
            def fetch_loc():
                with urllib.request.urlopen("http://ip-api.com/json/", timeout=2) as r:
                    return json.loads(r.read().decode())
            loc = await asyncio.to_thread(fetch_loc)
            city = loc.get("city")
            region = loc.get("regionName")
            if city:
                query = f"{query} in {city}, {region}"
        except Exception as e:
            if DEBUG:
                print(f"[GEO IP] Failed to get location: {e}")

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
sessions = SessionManager(short_turns=8, long_memory_path="memory/long_memory.json", debug=DEBUG)
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

def build_product_context(results, max_items: int = 5) -> str:
    """Format product retrieval results into context (dominant category to avoid mixing)."""
    if not results:
        return ""
    
    def get_price(r):
        try:
            return float(r.get("metadata", {}).get("price", 0))
        except:
            return 0.0
            
    results = sorted(results, key=get_price, reverse=True)
    
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
        
        price_str = f"₹{price}" if price else ""
        head = " | ".join([x for x in [title, price_str, cat] if x])
        lines.append(f"{head}. {desc}".strip())
        
        if len(lines) >= max_items:
            break
    if not lines:
        # fallback without filtering
        for r in results[:max_items]:
            m = r.get("metadata", {}) or {}
            title = m.get("title") or m.get("name") or ""
            price = m.get("price", "")
            cat = m.get("category", "")
            desc = m.get("description", "") or (r.get("content") or "")[:160]
            
            price_str = f"₹{price}" if price else ""
            head = " | ".join([x for x in [title, price_str, cat] if x])
            lines.append(f"{head}. {desc}".strip())
            
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
    product_max_tokens = 200 if voice_mode else 350
    personalized_max_tokens = 200 if voice_mode else 350
    general_max_tokens = 150 if voice_mode else 300
    web_max_tokens = 150 if voice_mode else 300

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

    # ===== PRODUCT CONTEXT CHECK (follow-up detail queries) =====
    stored_products = model_manager.get_product_context(user_id)
    if stored_products and intent in ["product_search", "price_filter"]:
        query_lower = query.lower()
        for prod in stored_products:
            m = prod.get("metadata", {}) or {}
            name = (m.get("title") or m.get("name") or "").lower()
            if name and len(name) > 3 and name in query_lower:
                title = m.get("title") or m.get("name") or "Product"
                price = m.get("price", "")
                desc = m.get("description", "")
                cat = m.get("category", "")
                price_str = f"₹{price}" if price else ""
                head = " | ".join([x for x in [title, price_str, cat] if x])
                reply = f"{head}. {desc}".strip()
                sessions.add_user_msg(user_id, query, persist_long=True)
                sessions.add_bot_msg(user_id, reply, persist_long=True)
                timings["total"] = round(time.time() - start, 3)
                if DEBUG:
                    print(f"[ROUTE] PRODUCT DETAIL FROM CONTEXT → Direct reply")
                    print(f"[TIMING] {timings}")
                return reply

    # ===== PATH 2: PRODUCT/DATABASE QUERIES =====
    if intent in ["product_search", "price_filter", "meta_count"]:
        if DEBUG:
            print(f"[ROUTE] PRODUCT QUERY → Direct DB retrieval + Light LLM humanize")

        sessions.add_user_msg(user_id, query, persist_long=True)

        with timed("retrieval", timings):
            results = await retrieve(query, top_k=retrieval_k)

        # Store product context for follow-up queries
        if results:
            model_manager.set_product_context(user_id, results)

        # Try deterministic extraction first
        with timed("extract", timings):
            fact = extract_fact(query, results, voice_mode=voice_mode)

        if fact:
            # Deterministic fact found — return directly (no LLM to avoid drift)
            reply = fact
        else:
            # No deterministic fact, use context with LLM (still constrained)
            context = build_product_context(results)
            history = get_recent_history(user_id, max_turns=3)
            with timed("llm_product", timings):
                reply = await model_manager.generate_reply(
                    SYSTEM_PREAMBLE_PRODUCT,
                    query,
                    context_text=context,
                    history_text=history,
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
    general_history = get_recent_history(user_id, max_turns=3)
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
        reply = await model_manager.generate_reply(
            SYSTEM_PREAMBLE_GENERAL_FAST,
            query,
            history_text=general_history,
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
    product_max_tokens = 200 if voice_mode else 350
    personalized_max_tokens = 200 if voice_mode else 350
    general_max_tokens = 150 if voice_mode else 300
    web_max_tokens = 150 if voice_mode else 300

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

    # ===== PRODUCT CONTEXT CHECK (follow-up detail queries) =====
    stored_products = model_manager.get_product_context(user_id)
    if stored_products and intent in ["product_search", "price_filter"]:
        query_lower = query.lower()
        for prod in stored_products:
            m = prod.get("metadata", {}) or {}
            name = (m.get("title") or m.get("name") or "").lower()
            if name and len(name) > 3 and name in query_lower:
                title = m.get("title") or m.get("name") or "Product"
                price = m.get("price", "")
                desc = m.get("description", "")
                cat = m.get("category", "")
                price_str = f"₹{price}" if price else ""
                head = " | ".join([x for x in [title, price_str, cat] if x])
                reply = f"{head}. {desc}".strip()
                sessions.add_user_msg(user_id, query, persist_long=True)
                sessions.add_bot_msg(user_id, reply, persist_long=True)
                timings["total"] = round(time.time() - start, 3)
                if DEBUG:
                    print(f"[ROUTE] PRODUCT DETAIL FROM CONTEXT → Direct reply")
                    print(f"[TIMING] {timings}")
                yield reply
                return

    if intent in ["product_search", "price_filter", "meta_count"]:
        if DEBUG:
            print(f"[ROUTE] PRODUCT QUERY → Direct DB retrieval + Light LLM humanize")

        sessions.add_user_msg(user_id, query, persist_long=True)

        with timed("retrieval", timings):
            results = await retrieve(query, top_k=retrieval_k)

        # Store product context for follow-up queries
        if results:
            model_manager.set_product_context(user_id, results)

        with timed("extract", timings):
            fact = extract_fact(query, results, voice_mode=voice_mode)

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
        history = get_recent_history(user_id, max_turns=3)
        collected = []
        with timed("llm_product", timings):
            async for chunk in model_manager.stream_reply(
                SYSTEM_PREAMBLE_PRODUCT,
                query,
                context_text=context,
                history_text=history,
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
    general_history = get_recent_history(user_id, max_turns=3)
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
                    top_p=0.85
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
            history_text=general_history,
            temperature=0.10 if voice_mode else 0.15,
            max_tokens=general_max_tokens,
            top_p=0.9
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
