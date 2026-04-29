# app_server.py
import os
import asyncio
from dotenv import load_dotenv
from fastapi import FastAPI, Request, Response
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
import uvicorn
import json

# Load environment variables from .env file
from dotenv import load_dotenv
load_dotenv()

# Ensure all relative paths resolve from this file's directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(BASE_DIR)

from chatbot import answer_query_async, answer_query_stream_async  # Make sure chatbot.py exposes this function

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

@app.post("/api/chat/stream")
async def chat_stream_endpoint(request: Request):
    data = await request.json()
    user_id = data.get("user_id", "default")
    query = data.get("query", "").strip()
    voice_mode = bool(data.get("voice_mode"))
    if not query:
        return JSONResponse({"error": "Empty query"}, status_code=400)

    async def event_stream():
        try:
            async for chunk in answer_query_stream_async(user_id, query, voice_mode=voice_mode):
                if chunk:
                    payload = json.dumps({"type": "delta", "text": chunk})
                    yield f"data: {payload}\n\n"
            yield f"data: {json.dumps({'type': 'done'})}\n\n"
        except Exception as exc:
            yield f"data: {json.dumps({'type': 'error', 'error': str(exc)})}\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        },
    )

if __name__ == "__main__":
    # Host on all interfaces for Android WebView access, or localhost for laptop testing
    host = "0.0.0.0" if os.environ.get("TERMUX") else "127.0.0.1"
    uvicorn.run(app, host=host, port=5010)
