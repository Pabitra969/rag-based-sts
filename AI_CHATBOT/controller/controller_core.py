# controller/controller_core.py
from typing import Dict, Any, List
from .intent_detector import detect_intent
from .context_manager import ContextManager
from nlp_core import detect_branch_and_handle
from quick_responder.quick_responder import QuickResponder

# --- Module Initialization ---
quick_responder = QuickResponder(beautify_with_model=False)


class Controller:
    def __init__(self, embedder, faiss_retriever, local_retriever, sessions, top_k: int = 4, debug: bool = False):
        self.context_manager = ContextManager(embedder, faiss_retriever, local_retriever, sessions, top_k=top_k, debug=debug)
        self.sessions = sessions
        self.debug = debug

        # Tunable base weights for scoring
        self.weights = {
            "retrieval": 0.9,
            "history": 0.8,
            "model": 1.0
        }

    async def prepare_context(self, user_id: str, query: str) -> Dict[str, Any]:
        intent, conf = detect_intent(query)
        decision = {
            "intent": intent,
            "confidence": conf,
            "use_model": True,
            "sources_text": "",
            "history_text": "",
            "results": []
        }

        # === STEP 1: Branch-specific quick handling ===
        branch_intents = {"price_filter", "meta_count", "product_search", "ambiguous_short"}
        if intent in branch_intents:
            results = await self.context_manager.fetch_retrieval(query)
            decision["results"] = results
            branch_answer = detect_branch_and_handle(query, results)
            if branch_answer:
                decision["use_model"] = False
                decision["branch_answer"] = branch_answer
                return decision

        # === STEP 2: Quick short-circuit for trivial greetings ===
        if intent in ("greeting", "thanks", "farewell"):
            decision["use_model"] = False
            decision["branch_answer"] = quick_responder.get_response(query)
            return decision

        # === STEP 3: Compute dynamic priority scores ===
        retrieval_score = conf * self.weights["retrieval"]
        history_score = conf * self.weights["history"]
        model_score = conf * self.weights["model"]

        # Boost based on type
        if intent in ("personal_query", "complex_product"):
            history_score *= 1.3
        if intent in ("product_search", "price_filter"):
            retrieval_score *= 1.2

        # Fetch long memory context if relevant

        # Normalize scores
        max_score = max(retrieval_score, history_score, model_score, 1e-5)
        retrieval_score /= max_score
        history_score /= max_score
        model_score /= max_score

        if self.debug:
            print(f"[Controller] Scores → retrieval:{retrieval_score:.2f}, history:{history_score:.2f}, model:{model_score:.2f}")

        # === STEP 4: Determine context depth dynamically ===
        if intent in ("greeting", "thanks", "farewell"):
            history_depth = 0
            use_long = False
        elif intent in ("product_search", "price_filter"):
            history_depth = 1
            use_long = False
        else:
            history_depth = 2
            use_long = True

        decision["history_depth"] = history_depth
        decision["use_long_memory"] = use_long

        # === STEP 5: Context routing logic ===
        results, sources_text, history_text = [], "", ""
        retrieval_needed = retrieval_score >= 0.5
        history_needed = history_score >= 0.5

        if retrieval_needed:
            results = await self.context_manager.fetch_retrieval(query)
            sources_text = self.context_manager.build_sources_text(results)

        if history_needed:
            history_text = self.context_manager.get_short_history(user_id)

        # Default: model always runs unless disabled above
        decision["use_model"] = True

        # Assign prepared context
        decision.update({
            "results": results,
            "sources_text": sources_text,
            "history_text": history_text,
            "scores": {
                "retrieval": retrieval_score,
                "history": history_score,
                "model": model_score
            }
        })

        # === STEP 6: Merge mode ===
        if abs(retrieval_score - history_score) <= 0.2 and retrieval_score > 0.4:
            decision["merge_mode"] = "hybrid"
        elif retrieval_score > history_score:
            decision["merge_mode"] = "retrieval_focus"
        else:
            decision["merge_mode"] = "history_focus"

        if self.debug:
            print(f"[Controller] Merge mode: {decision['merge_mode']} (intent={intent})")
            
        if intent in ("greeting", "thanks", "farewell"):
             decision["use_model"] = False
             decision["branch_answer"] = quick_responder.get_response(query)
             return decision

        return decision

