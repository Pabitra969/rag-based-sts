# app_server.py
import os
import asyncio
from fastapi import FastAPI, Request, Response
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
import uvicorn

# Ensure all relative paths resolve from this file's directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(BASE_DIR)

# Import chatbot logic
from chatbot import answer_query_async  # Make sure chatbot.py exposes this function

app = FastAPI(title="Offline AI Chatbot", description="Local AI customer support chatbot")

# Serve static UI files from /ui directory
app.mount("/ui", StaticFiles(directory="ui"), name="ui")

@app.get("/", response_class=HTMLResponse)
async def root():
    """Serve the main chat UI."""
    return FileResponse("ui/index.html")

@app.post("/api/chat")
async def chat_endpoint(request: Request):
    """Chat endpoint to handle user queries."""
    data = await request.json()
    user_id = data.get("user_id", "default")
    query = data.get("query", "").strip()
    voice_mode = bool(data.get("voice_mode"))
    if not query:
        return JSONResponse({"error": "Empty query"}, status_code=400)

    try:
        # Call your chatbot's async answer function
        answer = await answer_query_async(user_id, query, voice_mode=voice_mode)
        return JSONResponse({"answer": answer})
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

if __name__ == "__main__":
    # Host on all interfaces for Android WebView access, or localhost for laptop testing
    host = "0.0.0.0" if os.environ.get("TERMUX") else "127.0.0.1"
    uvicorn.run(app, host=host, port=5010)
