# app/api/api_v1/endpoints/test_model.py
# 코드 설명: 테스트용 모델 API 엔드포인트를 정의합니다.
# 간단한 테스트 및 LLM 관련 기능을 제공합니다.
from fastapi import APIRouter
from google import genai
from google.genai import types
from model.topic import extract_topics_from_text
from pydantic import BaseModel

router = APIRouter()
client = genai.Client()

EMOTION_ALIASES = {
    "happy": "행복",
    "happiness": "행복",
    "joy": "기쁨",
    "joyful": "기쁨",
    "sad": "슬픔",
    "sadness": "슬픔",
    "angry": "분노",
    "anger": "분노",
    "annoyed": "짜증",
    "annoying": "짜증",
    "irritated": "짜증",
    "irritating": "짜증",
    "neutral": "무난함",
    "calm": "무난함",
    "plain": "무난함",
}


def _normalize_emotion_label(value: str) -> str:
    cleaned = (value or "").strip().lower()
    cleaned = cleaned.replace(".", "").replace(",", "")
    if cleaned in EMOTION_ALIASES:
        return EMOTION_ALIASES[cleaned]
    for key, normalized in EMOTION_ALIASES.items():
        if key in cleaned:
            return normalized
    return "무난함"

class LLMRequest(BaseModel):
    content: str
    
@router.get("/test")
def test_endpoint():
    return {"message": "This is a test endpoint."}

@router.post("/llm")
def llm_endpoint(request: LLMRequest):
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
    return {"response": _normalize_emotion_label(response.text)}
