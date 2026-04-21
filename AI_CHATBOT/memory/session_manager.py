# memory/session_manager.py
import json, os, time
from collections import defaultdict
from typing import List, Dict

class SessionManager:
    def __init__(self, short_turns: int = 2, long_memory_path: str = "memory/long_memory.json", debug=False):
        self.short_turns = short_turns
        self.sessions = defaultdict(list)
        self.long_memory_path = long_memory_path
        self.debug = debug
        os.makedirs(os.path.dirname(long_memory_path), exist_ok=True)

    def add_user(self, user_id: str):
        if user_id not in self.sessions:
            self.sessions[user_id] = []

    def add_user_msg(self, user_id: str, text: str):
        self.add_user(user_id)
        self.sessions[user_id].append({"role": "user", "text": text, "ts": time.time()})
        self._save_to_long(user_id, "user", text)
        self._trim(user_id)
        
    def get_user_only_history(self, user_id: str, turns: int = 2) -> str:
        msgs = self.sessions.get(user_id, [])[-turns*2:]
        history_lines = []
        for m in msgs:
            if m["role"] == "user":
                history_lines.append(f"User: {m['text']}")
        # skip bot lines entirely
        return "\n".join(history_lines)


    def add_bot_msg(self, user_id: str, text: str):
        self.add_user(user_id)
        self.sessions[user_id].append({"role": "bot", "text": text, "ts": time.time()})
        self._save_to_long(user_id, "bot", text)
        self._trim(user_id)
        
    def get_session(self, user_id: str) -> list:
        """Return full in-memory session for a user."""
        return self.sessions.get(user_id, [])


    def get_recent_short(self, user_id: str) -> str:
        msgs = self.sessions.get(user_id, [])[-self.short_turns*2:]
        return "\n".join([f"{m['role']}: {m['text']}" for m in msgs])

    def _trim(self, user_id: str):
        if len(self.sessions[user_id]) > self.short_turns*2:
            self.sessions[user_id] = self.sessions[user_id][-self.short_turns*2:]

    def _save_to_long(self, user_id: str, role: str, text: str):
        entry = {"user_id": user_id, "role": role, "text": text, "ts": time.time()}
        try:
            if os.path.exists(self.long_memory_path):
                data = json.load(open(self.long_memory_path, "r", encoding="utf-8"))
            else:
                data = {}
            if user_id not in data:
                data[user_id] = []
            data[user_id].append(entry)
            with open(self.long_memory_path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            if self.debug:
                print("⚠️ Long-memory write error:", e)

    def clear_user(self, user_id: str):
        self.sessions[user_id] = []
