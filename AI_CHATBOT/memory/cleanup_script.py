# memory/cleanup_script.py
import json, os, time

def cleanup_memory(path="memory/long_memory.json", max_users=50, max_age_days=30):
    if not os.path.exists(path):
        return
    data = json.load(open(path, "r", encoding="utf-8"))
    now = time.time()
    for uid, logs in list(data.items()):
        # remove users exceeding age limit
        if logs and (now - logs[-1]["ts"]) > (max_age_days * 86400):
            del data[uid]
    # trim total users
    if len(data) > max_users:
        # delete oldest users
        sorted_u = sorted(data.items(), key=lambda x: x[1][-1]["ts"])
        data = dict(sorted_u[-max_users:])
    json.dump(data, open(path, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
