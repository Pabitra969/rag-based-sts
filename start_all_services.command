#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"

AI_DIR="$ROOT_DIR/AI_CHATBOT"
VOICE_DIR="$ROOT_DIR/Voice_STS"
BACKEND_DIR="$VOICE_DIR/backend"
FRONTEND_DIR="$VOICE_DIR/frontend"

if [[ ! -d "$AI_DIR" || ! -d "$VOICE_DIR" ]]; then
  echo "Could not find AI_CHATBOT or Voice_STS directories from: $ROOT_DIR"
  exit 1
fi

AI_PY_CMD="python3 app_server.py"

# Opens a new Terminal tab and runs one command.
run_in_new_tab() {
  local title="$1"
  local cmd="$2"

  /usr/bin/osascript <<APPLESCRIPT
 tell application "Terminal"
   activate
   do script "echo '[${title}]'; ${cmd}"
 end tell
APPLESCRIPT
}

echo "Starting services in separate Terminal tabs..."

run_in_new_tab "AI_CHATBOT" "cd '$AI_DIR' && $AI_PY_CMD"
sleep 0.8

run_in_new_tab "VOICE_BACKEND" "cd '$BACKEND_DIR' && npm run dev"
sleep 0.8

run_in_new_tab "VOICE_FRONTEND" "cd '$FRONTEND_DIR' && python3 -m http.server 8080"
sleep 0.8

run_in_new_tab "LOCAL_STS" "cd '$VOICE_DIR' && python3 local_speech_server.py"

echo "All launch commands sent to Terminal."
