# memory/session_manager.py
import json, os, time
from collections import defaultdict
from typing import List, Dict

class SessionManager:
    def __init__(self, short_turns: int = 2, long_memory_path: str = "memory/long_memory.json", debug=False):
        self.short_turns = short_turns
        self.sessions = defaultdict(list)
        self.pending_long = defaultdict(list)
        self.long_memory_path = long_memory_path
        self.debug = debug
        os.makedirs(os.path.dirname(long_memory_path), exist_ok=True)

    def add_user(self, user_id: str):
        if user_id not in self.sessions:
            self.sessions[user_id] = []

    def add_user_msg(self, user_id: str, text: str, persist_long: bool = True):
        self.add_user(user_id)
        entry = {"role": "user", "text": text, "ts": time.time()}
        self.sessions[user_id].append(entry)
        if persist_long:
            self._queue_long_entry(user_id, "user", text, entry["ts"])
        self._trim(user_id)
        
    def get_user_only_history(self, user_id: str, turns: int = 2) -> str:
        msgs = self.sessions.get(user_id, [])[-turns*2:]
        history_lines = []
        for m in msgs:
            if m["role"] == "user":
                history_lines.append(f"User: {m['text']}")
        # skip bot lines entirely
        return "\n".join(history_lines)


    def add_bot_msg(self, user_id: str, text: str, persist_long: bool = True):
        self.add_user(user_id)
        entry = {"role": "bot", "text": text, "ts": time.time()}
        self.sessions[user_id].append(entry)
        if persist_long:
            self._queue_long_entry(user_id, "bot", text, entry["ts"])
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

    def _queue_long_entry(self, user_id: str, role: str, text: str, ts: float):
        entry = {"user_id": user_id, "role": role, "text": text, "ts": ts}
        self.pending_long[user_id].append(entry)

    def flush_pending(self, user_id: str = None):
        pending_users = [user_id] if user_id is not None else list(self.pending_long.keys())
        if not pending_users:
            return

        try:
            if os.path.exists(self.long_memory_path):
                with open(self.long_memory_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
            else:
                data = {}

            changed = False
            for current_user_id in pending_users:
                entries = self.pending_long.get(current_user_id) or []
                if not entries:
                    continue
                if current_user_id not in data:
                    data[current_user_id] = []
                data[current_user_id].extend(entries)
                self.pending_long[current_user_id] = []
                changed = True

            if not changed:
                return

            with open(self.long_memory_path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            if self.debug:
                print("⚠️ Long-memory write error:", e)

    def clear_user(self, user_id: str):
        self.sessions[user_id] = []
