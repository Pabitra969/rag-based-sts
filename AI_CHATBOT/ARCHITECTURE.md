# Chatbot Architecture: Smart Intent Routing

## Overview
The chatbot now uses intelligent query routing instead of blindly applying LLM to all responses. Each query type is handled differently to ensure accuracy and relevance.

---

## Architecture Layers

### Layer 1: Entry Point (`app_server.py`)
- FastAPI server on port 5001
- Endpoint: `POST /api/chat`
- Calls: `answer_query_async(user_id, query)` from `chatbot.py`

### Layer 2: Chatbot Wrapper (`chatbot.py`)
- Clean interface between server and logic
- Initializes all modules (model, embedder, retrievers, sessions)
- Creates `ChatbotController` instance
- Exposes `answer_query_async()` function

### Layer 3: Controller (`controller/chatbot_controller.py`)
**Main orchestrator** - handles all routing logic
- **Intent Detection**: Classifies query type
- **Retrieval**: Fetches relevant products from DB
- **Response Generation**: Routes to appropriate handler
- **History Management**: Manages conversation context

---

## Query Routing Flow

```
User Query
    ↓
[1] Check Quick Responder (greetings)
    ├─ If match → Return immediately
    └─ Else → Continue
    ↓
[2] Detect Intent (using controller.intent_detector)
    ├─ "greeting" → Path 1 (already handled)
    ├─ "product_search" / "price_filter" / "meta_count" → Path 2
    ├─ "personal_query" → Path 3
    └─ "unknown" / "general" → Path 4
    ↓
[3] Retrieve Products (embedding-based search)
    └─ Query → Embed → FAISS search (top 3) → Results
    ↓
[4] Generate Response (based on detected intent)
```

---

## The 4 Response Paths

### PATH 1: GREETINGS (Quickest)
**Intent**: greeting  
**Handler**: `QuickResponder`  
**Logic**: 
- Pattern match against predefined greetings
- Return immediately without model inference
- No history, no retrieval

**Example**:
```
User: "Hi"
Bot: "Hello! How can I help you?" (from quick_bank.json)
```

---

### PATH 2: PRODUCT QUERIES (Fast + Accurate)
**Intents**: `product_search`, `price_filter`, `meta_count`  
**Handler**: `generate_product_response()`  
**Logic**:
1. Retrieve top 3 products from database
2. Try deterministic extraction (price, count, title)
3. If fact found: Polish with LLM (low temp=0.25, constrained)
4. Else: Generate with product context (temp=0.3)
5. NO history passed (stay factual, prevent hallucination)

**Key Feature**: Uses `extract_fact()` to pull structured data FIRST, then optionally refines tone

**Temperature**: 0.25 (highly constrained)  
**Context**: Product info only  
**History**: NO (stays grounded in facts)  
**Max Tokens**: 70-90

**Example**:
```
User: "What's the price of T-shirt?"
[Deterministic] → "T-shirt costs ₹599."
[LLM Polish] → "Our T-shirt is available for ₹599."
```

---

### PATH 3: PERSONALIZED QUERIES (Thoughtful)
**Intent**: `personal_query`  
**Handler**: `generate_personalized_response()`  
**Logic**:
1. Retrieve relevant products/context
2. Load FULL conversation history
3. Generate response using context + full history
4. Model can reference past interactions

**Temperature**: 0.4 (moderate creativity)  
**Context**: Product info (max 3 items)  
**History**: FULL conversation  
**Max Tokens**: 120

**Example**:
```
User (earlier): "I'm looking for formal clothes"
User: "Do you have something in white?"
[Reads history] → Knows user wants formal clothes
[Responds] → "Based on your interest in formal clothing, 
we have white formal shirts for ₹799."
```

---

### PATH 4: GENERAL KNOWLEDGE (Free + Helpful)
**Intent**: `unknown` or anything not matching above  
**Handler**: `generate_general_response()`  
**Logic**:
1. Retrieve products (for suggestions)
2. Answer general question freely
3. Suggest related products from database
4. Use recent history (last 2 turns) for context

**Temperature**: 0.5 (creative, natural)  
**Context**: Product suggestions only (2 items)  
**History**: Recent only (2 turns)  
**Max Tokens**: 120

**Example**:
```
User: "How to care for a T-shirt?"
[Model answers] → "T-shirts should be washed in cold water..."
[Suggests] → "By the way, we have premium T-shirts for ₹699."
```

---

## Key Components

### Intent Detector (`controller/intent_detector.py`)
- Hybrid detection: regex + embedding similarity
- Predefined intents: `greeting`, `thanks`, `product_search`, `price_filter`, `personal_query`, `meta_count`
- Returns: `(intent, confidence_score)`

### Retrievers
- **FaissRetriever**: Vector search on pre-built index
- **LocalRetriever**: Fallback keyword search on CSV
- Both async, thread-pooled for performance

### NLP Core (`nlp_core.py`)
- `extract_fact()`: Deterministically pulls price, count, or product details
- Safe metadata extraction with fallbacks

### Session Manager (`memory/session_manager.py`)
- Short-term: Last N turns (in-memory)
- Long-term: Full history (JSON file)
- Separate user sessions by `user_id`

### Model Manager (`model_session_manager.py`)
- Wraps TinyLlama-1.1B GGUF
- Async inference with context injection
- Safe session persistence

---

## Why This Works Better

| Issue | Old Approach | New Approach |
|-------|-------------|--------------|
| **Product hallucination** | LLM freely interprets facts | Extracts facts first, LLM only polishes tone |
| **Wrong greetings** | Full model inference | Quick lookup, instant response |
| **Personalization** | Light history only | Full history for personalized queries |
| **General questions** | Always grounded in products | Can answer freely, then suggest |
| **Response consistency** | Varied quality | Intent-specific templates + LLM |
| **Temperature tuning** | One size fits all | Per-path optimization (0.25 → 0.5) |

---

## Configuration

Edit in `chatbot.py`:
```python
N_CTX = 1024          # Model context size
TOP_K = 3             # Retrieval results to fetch
DEBUG = True           # Print routing info
```

Per-path temps and tokens are in `controller/chatbot_controller.py`:
- Product: temp=0.25, max_tokens=70
- Personalized: temp=0.4, max_tokens=120
- General: temp=0.5, max_tokens=120

---

## Testing the Routes

### Test Greeting
```
User: "Hi"
Expected: Quick response from bank
```

### Test Product Query
```
User: "What's the price of T-shirt?"
Expected: Direct price from DB, LLM-polished
```

### Test Personalized
```
User: "Do you remember I wanted formal clothes? Any updates?"
Expected: References history + product suggestions
```

### Test General
```
User: "What's the weather tomorrow?"
Expected: Model answers + product suggestions
```

---

## Debugging

Check logs:
```python
DEBUG = True  # In chatbot.py
```

Output shows:
```
[INTENT] product_search (confidence: 0.87)
[ROUTE] PRODUCT QUERY → Direct DB retrieval + Light LLM humanize
[RETRIEVAL] Faiss returned 3 items
[TIMING] {'quick': 0.001, 'intent_detect': 0.05, 'retrieval': 0.12, 'extract': 0.02, 'llm_humanize': 0.3, 'save': 0.01, 'total': 0.51}
```

---

## Future Improvements

1. **Dynamic Temperature**: Adjust based on confidence score
2. **Multi-turn Context**: Better context window management
3. **Fallback Escalation**: Route to human if confidence < threshold
4. **Product Category Filters**: Smart filtering to prevent cross-category confusion
5. **Feedback Loop**: Learn which paths user prefers
