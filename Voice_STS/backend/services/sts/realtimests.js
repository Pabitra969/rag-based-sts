//realtimests.js
const sdk = require("microsoft-cognitiveservices-speech-sdk");
const AzureSTS = require("./azure.sts");
const LLMProvider = require("./llm.provider");
module.exports = function handleVoiceSession(ws) {
    const azure = new AzureSTS();
    const llm = new LLMProvider();

    const format = sdk.AudioStreamFormat.getWaveFormatPCM(16000, 16, 1);
    const pushStream = sdk.AudioInputStream.createPushStream(format);
    const recognizer = azure.createRecognizer(pushStream);
    const synthesizer = azure.createSynthesizer();

    const history = [];
    let isSpeaking = false;
    let speakController = null;

    function stopSpeaking() {
        if (!isSpeaking || !speakController) return;
        speakController.stopSpeakingAsync(
            () => {
                isSpeaking = false;
            },
            (err) => {
                console.error("stopSpeakingAsync error:", err);
                isSpeaking = false;
            }
        );
    }

    recognizer.recognizing = (_, e) => {
        if (!e.result.text) return;
        ws.send(
            JSON.stringify({
                type: "partial",
                text: e.result.text,
            })
        );
    };

    recognizer.recognized = async (_, e) => {
        if (!e.result.text) return;

        const userText = e.result.text.trim();
        history.push({ role: "user", content: userText });

        ws.send(
            JSON.stringify({
                type: "final",
                text: userText,
            })
        );

        try {
            const replyText = await llm.generateReply(userText, history);
            history.push({ role: "assistant", content: replyText });

            ws.send(
                JSON.stringify({
                    type: "bot_response",
                    text: replyText,
                })
            );

            // Avoid overlapping synth requests
            if (isSpeaking) return;
            isSpeaking = true;

            speakController = synthesizer;
            synthesizer.speakTextAsync(
                replyText,
                (result) => {
                    ws.send(result.audioData);
                    isSpeaking = false;
                    speakController = null;
                },
                (err) => {
                    isSpeaking = false;
                    speakController = null;
                    console.error("Error synthesizing speech:", err);
                }
            );
        } catch (err) {
            console.error("LLM/TTS error:", err);
            ws.send(
                JSON.stringify({
                    type: "bot_response",
                    text: "Sorry, I ran into a problem thinking about that.",
                })
            );
        }
    };

    recognizer.sessionStarted = () => {
        console.log("🎤 STT session started");
    };

    recognizer.canceled = (_, e) => {
        console.error("❌ STT canceled:", e.errorDetails);
    };

    recognizer.startContinuousRecognitionAsync();

    ws.on("message", (data, isBinary) => {
        // Control messages are sent as text
        if (!isBinary && typeof data === "string") {
            try {
                const msg = JSON.parse(data.toString());
                if (msg.type === "interrupt") {
                    stopSpeaking();
                }
            } catch (err) {
                console.error("Invalid control message:", err);
            }
            return;
        }

        // Audio stream
        pushStream.write(data);
    });

    ws.on("close", () => {
        pushStream.close();
        recognizer.stopContinuousRecognitionAsync();
        stopSpeaking();
    });
};
