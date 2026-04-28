#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"

AI_DIR="$ROOT_DIR/AI_CHATBOT"
VOICE_DIR="$ROOT_DIR/Voice_STS"
BACKEND_DIR="$VOICE_DIR/backend"
FRONTEND_DIR="$VOICE_DIR/frontend"
MAC_LLM_ENV="export LLM_N_GPU_LAYERS=-1; export LLM_MAIN_GPU=0; export LLM_N_BATCH=256; export LLM_OFFLOAD_KQV=true; export LLM_FLASH_ATTN=true;"
PIPER_ENV="export PIPER_MODEL_PATH='$VOICE_DIR/models/piper/en_US-lessac-medium.onnx'; export PIPER_CONFIG_PATH='$VOICE_DIR/models/piper/en_US-lessac-medium.onnx.json'; export PIPER_ACCELERATION='cpu';"

if [[ ! -d "$AI_DIR" || ! -d "$VOICE_DIR" ]]; then
  echo "Could not find AI_CHATBOT or Voice_STS directories from: $ROOT_DIR"
  exit 1
fi

if [[ "${TERM_PROGRAM:-}" == "vscode" ]]; then
  echo "You are in VS Code integrated terminal."
  echo "Use Cmd+Shift+B to run the default task: Start All Services."
  exit 0
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

run_in_new_tab "AI_CHATBOT" "cd '$AI_DIR' && $MAC_LLM_ENV $AI_PY_CMD"
sleep 0.8

run_in_new_tab "VOICE_BACKEND" "cd '$BACKEND_DIR' && npm run dev"
sleep 0.8

run_in_new_tab "VOICE_FRONTEND" "cd '$FRONTEND_DIR' && python3 -m http.server 8080"
sleep 0.8

run_in_new_tab "LOCAL_STS" "cd '$VOICE_DIR' && $PIPER_ENV python3 local_speech_server.py"

echo "All launch commands sent to Terminal."
