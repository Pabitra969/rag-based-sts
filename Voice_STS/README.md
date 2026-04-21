# Voice Universal Chat

A web-based chat application with a modular speech-to-text (STT) system that intelligently switches between browser-native APIs and cloud-based services.

## Features

*   **Dual Input Modes**: Supports both traditional text input and voice input via microphone.
*   **Voice Chat (full duplex)**: A headset button opens a live voice session; speech is transcribed, answered by the (pluggable) LLM, and streamed back as TTS audio.
*   **Dynamic UI**: The input controls (mic/send button) adapt based on user interaction.
*   **Smart STT Provider Switching**: Automatically uses the browser's free Web Speech API on supported browsers (like Chrome/Edge) and falls back to Microsoft Azure Speech for others, ensuring wide compatibility.
*   **Secure Backend**: A Node.js/Express backend securely manages and provides temporary access tokens for the Azure Speech service, keeping sensitive API keys off the client-side.
*   **Modular Architecture**: Both frontend and backend STT providers are abstracted, making it easy to add new services.
*   **Real-time Transcription**: Provides live feedback as the user speaks.

## How It Works

The project is divided into two main parts: JavaScript **frontend** and a Node.js **backend**.

### Frontend

The frontend is responsible for the user interface and all client-side logic.

1.  **UI Interaction (`speech.js`)**: Manages the chat display and toggles between the "send" and "microphone" buttons.
2.  **STT Provider Selection (`stt/index.js`)**: This is the core of the frontend's voice logic. When the user clicks the mic button, it checks if the browser supports the native `Web Speech API`.
    *   If **yes**, it uses the `webspeech.stt.js` provider, which is free and built into the browser.
    *   If **no**, it falls back to the `azure.stt.js` provider.
3.  **Azure STT (`stt/azure.stt.js`)**: To use Azure, it first requests a temporary access token from its own backend (`/api/speech-token`). It then uses this token to connect directly to Microsoft Azure's Speech service from the browser for real-time transcription.

### Backend

The backend is a simple, secure service with one primary role.

1.  **Token Endpoint (`routes/speechToken.route.js`)**: It exposes a single endpoint, `GET /api/speech-token`.
2.  **Secure Token Generation (`services/speech/azure.provider.js`)**: When the frontend requests a token, the backend uses its securely stored Azure Speech API Key (`AZURE_SPEECH_KEY`) to make an authenticated request to Microsoft's servers.
3.  **Response**: It receives a short-lived token from Azure and forwards it to the frontend.

This architecture ensures that the secret Azure API key is never exposed to the user's browser.

## Project Structure

```
voice_universal/
├── backend/
│   ├── services/
│   │   ├── speech/
│   │   │   ├── azure.provider.js     # Logic to get token from Azure
│   │   │   ├── ollama.provider.js    # next long term idea to implement
│   │   │   ├── speech.interface.js   # Defines the provider interface(token)
│   │   │   └── index.js              # Selects the backend provider
│   ├── routes/
│   │   └── speechToken.route.js  # The /api/speech-token endpoint
│   ├── .env.example              # Example environment variables
│   ├── package.json
│   └── server.js                 # Main Express server file
│
└── frontend/
    ├── stt/
    │   ├── azure.stt.js          # Client-side Azure STT logic
    │   ├── local.stt.js          # (Unused) Example of local audio capture (websockets)
    │   ├── webspeech.stt.js      # Client-side Web Speech API logic
    │   └── index.js              # Selects the frontend STT provider
    ├── index.html                # Main HTML file
    ├── speech.js                 # Main UI and event handling logic
    └── style.css                 # Styles for the chat UI
```

## Getting Started

### Prerequisites

*   Node.js and Go live
*   An Azure account with a Speech service resource to get an API Key and Region (if you want to test the Azure provider).

### 1. Backend Setup

```bash
# 1. Navigate to the backend directory
cd backend

# 2. Install dependencies
npm install
npm install express, cors, dotenv, axios, express-rate-limit


# 3. Create a .env file from the example
cp .env.example .env
```

Now, open the `.env` file and add your Azure credentials. Set the `SPEECH_PROVIDER` to `azure`.

```.env
PORT=5005
SPEECH_PROVIDER=azure/openai/ollama or if you set local
AZURE_SPEECH_KEY=YOUR_AZURE_SPEECH_API_KEY
AZURE_SPEECH_REGION=YOUR_AZURE_SPEECH_REGION
# Voice chat (LLM) – optional
# Point to your own local LLM HTTP endpoint (expects {text} in response)
LOCAL_LLM_URL=http://localhost:11434/api/chat
```

```bash
# 4. Start the backend server
npm start
```

The server will be running at `http://localhost:5005`.

### 2. Frontend Setup

The frontend is a static site. You can serve it using any simple web server. A popular choice is (Go Live)`live-server`.

```bash
# 1. If you don't have live-server, install it globally
npm install -g live-server

# 2. Navigate to the frontend directory
cd frontend

# 3. Start the server
live-server
```

Your browser will open to the chat application, and it will be able to communicate with the backend.

*   On **Chrome/Edge**, it will use the Web Speech API.
*   On **Firefox/Safari**, it will automatically fall back to using Azure by fetching a token from your backend.
*   Voice mode connects via WebSocket at `ws://localhost:5005/ws/voice` by default. If you host the backend elsewhere, set `window.VOICE_WS_URL` in `index.html` before loading `speech.js`.
*   LLM replies default to a canned response unless you set `LOCAL_LLM_URL` to your own model endpoint (e.g., Ollama / LM Studio / custom HTTP).






                         ┌──────────────────┐
                         │  Frontend Mic    │
                         └────────┬─────────┘
                                  │
                                  ▼
                        ┌────────────────────┐
                        │ Default Frontend   │
                        │ index.html         │
                        │ style.css          │
                        │ speech.js          │
                        └────────┬───────────┘
                                 │
                                 ▼
                              index.js
                                 │
        ┌────────────────────────┼────────────────────────┐
        │                        │                        │
        ▼                        ▼                        ▼
┌──────────────────┐   ┌────────────────────┐   ┌──────────────────┐
│ SpeechRecognition│   │ Azure Speech SDK   │   │ Local STT        │
│ (Web Speech API) │   │ (4 Functions)      │   │ (WebSocket)      │
│ webspeech.stt.js │   │ azure.stt.js       │   │ local.stt.js     │
│                  │   │                    │   │                  │
│ - No backend     │   │ - Uses backend     │   │ - Uses backend   │
│ - Chromium only  │   │                    │   │                  │
│ - Simple & Free  │   │                    │   │                  │
└──────────────────┘   └──────────┬─────────┘   └──────────┬───────┘
                                   │                       │
                                   └──────────┬────────────┘
                                              ▼
                                       ┌──────────────┐
                                       │  server.js   │
                                       └──────┬───────┘
                                              ▼
                                 ┌────────────────────────┐
                                 │ routes/speechToken.js  │
                                 └──────────┬─────────────┘
                                            ▼
                                       ┌──────────────┐
                                       │  index.js    │
                                       └──────┬───────┘
                                              ▼
                                   ┌──────────────────┐
                                   | service provider |
                                   │   model / API    │
                                   │ (Azure / Ollama) │
                                   │ (Local / Cloud)  │
                                   └──────────┬───────┘
                                              ▼
                              ┌──────────────────────────┐
                              │ Azure Speech Key (ENV)   │
                              └──────────────────────────┘
