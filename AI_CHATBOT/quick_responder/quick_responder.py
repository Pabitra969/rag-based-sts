import json
import os
import re
from typing import Optional

class QuickResponder:
    def __init__(self, bank_path: str = "quick_responder/quick_bank.json",
                 beautify_with_model=False, model=None):
        self.beautify = beautify_with_model
        self.model = model
        self.responses = {}

        if os.path.exists(bank_path):
            with open(bank_path, "r", encoding="utf-8") as f:
                self.responses = json.load(f)
        else:
            print(f"⚠️ QuickResponder: no bank found at {bank_path}")

    def get_response(self, query: str) -> Optional[str]:
        q = query.lower().strip()

        for key, resp in self.responses.items():
            # exact or very short variant match only (ignore long sentences)
            pattern = rf"^\s*{re.escape(key.lower())}\s*[!.?,]*$"
            if re.match(pattern, q):  # full-line match only
                if self.beautify and self.model:
                    prompt = f"Rewrite politely but naturally: {resp}"
                    try:
                        out = self.model.create_completion(
                            prompt=prompt, max_tokens=60, temperature=0.2
                        )
                        return out["choices"][0]["text"].strip()
                    except Exception:
                        return resp
                return resp
        return None
