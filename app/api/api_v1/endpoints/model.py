# app/api/api_v1/endpoints/model.py
# 코드 설명: 모델 관련 API 엔드포인트를 정의합니다.
from fastapi import (
    APIRouter,
    File,
    HTTPException,
    UploadFile,
    WebSocket,
    WebSocketDisconnect,
)
from fastapi.responses import HTMLResponse
from google import genai
from google.genai import types
from model.topic import extract_topics_from_text
from pydantic import BaseModel
from app.services.conversation_service import ConversationService
from app.services.stt_service import WhisperSTTService

router = APIRouter()
client = genai.Client()
conversation_service = ConversationService()
stt_service = WhisperSTTService()

class LLMRequest(BaseModel):
    content: str

@router.post("/topic")
def topic_endpoint(request: LLMRequest):
    # Use the topic extraction logic from `model.topic`
    try:
        topic = extract_topics_from_text(request.content, use_local=False)
        return {
            "topic": {
                "count": topic.count,
                "tag1": topic.tag1,
                "tag2": topic.tag2,
            }
        }
    except Exception as e:
        return {"error": str(e)}

@router.post("/emotion")
def emotion_endpoint(request: LLMRequest):
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        config=types.GenerateContentConfig(
            system_instruction="You're an emotion analysis bot. Analyze the emotion of the given content and respond with one word representing the emotion (happy, joy, neutral, sad, annoyed, angry).",
        ),
        contents=request.content
    )
    return {"response": response.text}

@router.post("/summary")
def summary_endpoint(request: LLMRequest):
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        config=types.GenerateContentConfig(
            system_instruction="You're a summary bot. Create a one-line summary of the given content in korean.",
        ),
        contents=request.content
    )
    return {"summary": response.text}

@router.websocket("/ws/conversations")
async def conversations_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            chunk = await websocket.receive_bytes()
            response_chunk = await conversation_service.process_audio_chunk(chunk)
            await websocket.send_bytes(response_chunk)
    except WebSocketDisconnect:
        pass
    except Exception:
        await websocket.close(code=1011, reason="Internal server error")


@router.websocket("/ws/stt")
async def stt_websocket(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            chunk = await websocket.receive_bytes()
            text = await stt_service.transcribe_wav_bytes(chunk)
            await websocket.send_json({"text": text})
    except WebSocketDisconnect:
        pass
    except Exception as e:
        print(f"Error in STT WebSocket: {e}")
        await websocket.close(code=1011, reason="Internal server error")
        
@router.post("/stt/transcribe-file")
async def stt_transcribe_file(file: UploadFile = File(...)):
    if not file:
        raise HTTPException(status_code=400, detail="File is required.")
    audio_bytes = await file.read()
    text = await stt_service.transcribe_wav_bytes(audio_bytes)
    return {"text": text}


STT_TEST_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0" />
    <title>STT Test</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            background: #f7f7f9;
            padding: 20px;
            color: #222;
        }
        .container {
            max-width: 720px;
            margin: 0 auto;
            background: #fff;
            border-radius: 12px;
            padding: 24px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.08);
        }
        h1 {
            margin-top: 0;
            text-align: center;
        }
        button {
            background: #2563eb;
            color: #fff;
            border: none;
            padding: 10px 16px;
            border-radius: 6px;
            cursor: pointer;
            font-size: 15px;
        }
        button:disabled {
            opacity: 0.4;
            cursor: not-allowed;
        }
        .section {
            margin-bottom: 24px;
        }
        .section label {
            display: block;
            margin-bottom: 8px;
            font-weight: 600;
        }
        .log {
            background: #0f172a;
            color: #e2e8f0;
            padding: 16px;
            border-radius: 8px;
            height: 220px;
            overflow-y: auto;
        }
        .log ul {
            list-style: none;
            padding: 0;
            margin: 0;
        }
        .log li {
            margin-bottom: 10px;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>실시간 STT 테스트</h1>
        <div class="section">
            <label>마이크 녹음</label>
            <button id="startRecord">녹음 시작</button>
            <button id="stopRecord" disabled>녹음 종료</button>
        </div>
        <div class="section">
            <label for="audioUpload">음성 파일 업로드 (wav)</label>
            <input type="file" id="audioUpload" accept="audio/wav,audio/wave,audio/x-wav" />
        </div>
        <div class="section log">
            <strong>인식 로그</strong>
            <ul id="logList"></ul>
        </div>
    </div>
    <script>
        (() => {
            const TARGET_RATE = 16000;
            const CHUNK_SECONDS = 2;
            const logList = document.getElementById("logList");
            const startBtn = document.getElementById("startRecord");
            const stopBtn = document.getElementById("stopRecord");
            const audioUpload = document.getElementById("audioUpload");

            let ws = null;
            let audioContext = null;
            let mediaStream = null;
            let processor = null;
            let sourceNode = null;
            let pendingBuffers = [];
            let pendingLength = 0;

            function appendLog(text) {
                const li = document.createElement("li");
                li.textContent = `[${new Date().toLocaleTimeString()}] ${text}`;
                logList.prepend(li);
            }

            function mergeBuffers(buffers, totalLength) {
                const result = new Float32Array(totalLength);
                let offset = 0;
                buffers.forEach((buffer) => {
                    result.set(buffer, offset);
                    offset += buffer.length;
                });
                return result;
            }

            function downsampleBuffer(buffer, sampleRate, outSampleRate) {
                if (outSampleRate >= sampleRate) {
                    return buffer;
                }
                const ratio = sampleRate / outSampleRate;
                const newLength = Math.round(buffer.length / ratio);
                const result = new Float32Array(newLength);
                let offsetResult = 0;
                let offsetBuffer = 0;
                while (offsetResult < newLength) {
                    const nextOffset = Math.round((offsetResult + 1) * ratio);
                    let accum = 0;
                    let count = 0;
                    for (let i = offsetBuffer; i < nextOffset && i < buffer.length; i++) {
                        accum += buffer[i];
                        count++;
                    }
                    result[offsetResult] = count ? accum / count : 0;
                    offsetResult++;
                    offsetBuffer = nextOffset;
                }
                return result;
            }

            function encodeWAV(samples, sampleRate) {
                const buffer = new ArrayBuffer(44 + samples.length * 2);
                const view = new DataView(buffer);

                function writeString(view, offset, string) {
                    for (let i = 0; i < string.length; i++) {
                        view.setUint8(offset + i, string.charCodeAt(i));
                    }
                }

                writeString(view, 0, "RIFF");
                view.setUint32(4, 36 + samples.length * 2, true);
                writeString(view, 8, "WAVE");
                writeString(view, 12, "fmt ");
                view.setUint32(16, 16, true);
                view.setUint16(20, 1, true);
                view.setUint16(22, 1, true);
                view.setUint32(24, sampleRate, true);
                view.setUint32(28, sampleRate * 2, true);
                view.setUint16(32, 2, true);
                view.setUint16(34, 16, true);
                writeString(view, 36, "data");
                view.setUint32(40, samples.length * 2, true);

                let offset = 44;
                for (let i = 0; i < samples.length; i++, offset += 2) {
                    const s = Math.max(-1, Math.min(1, samples[i]));
                    view.setInt16(offset, s < 0 ? s * 0x8000 : s * 0x7fff, true);
                }
                return buffer;
            }

            async function startRecording() {
                if (!navigator.mediaDevices) {
                    appendLog("이 브라우저는 녹음을 지원하지 않습니다.");
                    return;
                }
                startBtn.disabled = true;
                stopBtn.disabled = false;
                pendingBuffers = [];
                pendingLength = 0;

                const protocol = window.location.protocol === "https:" ? "wss://" : "ws://";
                const wsUrl = protocol + window.location.host + "/api/v1/model/ws/stt";
                ws = new WebSocket(wsUrl);

                ws.addEventListener("message", (event) => {
                    try {
                        const payload = JSON.parse(event.data);
                        if (payload.text) {
                            appendLog(`Streaming: ${payload.text}`);
                        }
                    } catch {
                        appendLog("인식 결과 수신");
                    }
                });
                ws.addEventListener("error", () => appendLog("웹소켓 오류가 발생했습니다."));

                await new Promise((resolve, reject) => {
                    ws.addEventListener("open", resolve, { once: true });
                    ws.addEventListener(
                        "error",
                        (event) => reject(event),
                        { once: true }
                    );
                });

                mediaStream = await navigator.mediaDevices.getUserMedia({ audio: true });
                audioContext = new (window.AudioContext || window.webkitAudioContext)();
                sourceNode = audioContext.createMediaStreamSource(mediaStream);
                processor = audioContext.createScriptProcessor(4096, 1, 1);
                processor.onaudioprocess = handleAudioProcess;
                sourceNode.connect(processor);
                processor.connect(audioContext.destination);
                appendLog("녹음을 시작했습니다.");
            }

            function stopRecording() {
                startBtn.disabled = false;
                stopBtn.disabled = true;
                pendingBuffers = [];
                pendingLength = 0;

                if (processor) {
                    processor.disconnect();
                    processor.onaudioprocess = null;
                    processor = null;
                }
                if (sourceNode) {
                    sourceNode.disconnect();
                    sourceNode = null;
                }
                if (mediaStream) {
                    mediaStream.getTracks().forEach((t) => t.stop());
                    mediaStream = null;
                }
                if (audioContext) {
                    audioContext.close();
                    audioContext = null;
                }
                if (ws) {
                    ws.close();
                    ws = null;
                }
                appendLog("녹음을 종료했습니다.");
            }

            function handleAudioProcess(event) {
                if (!audioContext || !ws || ws.readyState !== WebSocket.OPEN) {
                    return;
                }
                const buffer = new Float32Array(event.inputBuffer.getChannelData(0));
                pendingBuffers.push(buffer);
                pendingLength += buffer.length;

                const threshold = (audioContext.sampleRate || 44100) * CHUNK_SECONDS;
                if (pendingLength >= threshold) {
                    const merged = mergeBuffers(pendingBuffers, pendingLength);
                    pendingBuffers = [];
                    pendingLength = 0;
                    const downsampled = downsampleBuffer(merged, audioContext.sampleRate, TARGET_RATE);
                    const wavBuffer = encodeWAV(downsampled, TARGET_RATE);
                    ws.send(wavBuffer);
                }
            }

            audioUpload.addEventListener("change", async (event) => {
                const file = event.target.files[0];
                if (!file) return;
                appendLog("파일 전송 중...");
                const formData = new FormData();
                formData.append("file", file);
                try {
                    const response = await fetch("/api/v1/model/stt/transcribe-file", {
                        method: "POST",
                        body: formData,
                    });
                    const data = await response.json();
                    appendLog(`File: ${data.text || "인식 실패"}`);
                } catch (error) {
                    appendLog("파일 업로드 중 오류가 발생했습니다.");
                } finally {
                    audioUpload.value = "";
                }
            });

            startBtn.addEventListener("click", () => {
                startRecording().catch((err) => {
                    console.error(err);
                    appendLog("녹음을 시작하지 못했습니다.");
                    startBtn.disabled = false;
                    stopBtn.disabled = true;
                });
            });
            stopBtn.addEventListener("click", stopRecording);
        })();
    </script>
</body>
</html>
"""


@router.get("/stt-test", response_class=HTMLResponse)
async def stt_test_page():
    return STT_TEST_HTML
