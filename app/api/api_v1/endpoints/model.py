# app/api/api_v1/endpoints/model.py
# 코드 설명: 모델 관련 API 엔드포인트를 정의합니다.
from pathlib import Path

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

class ConversationTurn(BaseModel):
    role: str
    text: str

class DiaryRequest(BaseModel):
    turns: list[ConversationTurn]

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

@router.post("/diary")
def diary_endpoint(request: DiaryRequest):
    try:
        diary = conversation_service.generate_diary(request.turns)
        return {"diary": diary}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.websocket("/ws/conversations")
async def conversations_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        await conversation_service.handle_conversation(websocket)
    except WebSocketDisconnect:
        pass
    except Exception as e:
        print(f"Error in conversations WebSocket: {e}")
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


STT_TEST_HTML_PATH = Path(__file__).resolve().parents[3] / "static" / "stt_test.html"


@router.get("/stt-test", response_class=HTMLResponse)
async def stt_test_page():
    if not STT_TEST_HTML_PATH.exists():
        raise HTTPException(status_code=500, detail="STT test page not found.")
    return HTMLResponse(content=STT_TEST_HTML_PATH.read_text(encoding="utf-8"))
