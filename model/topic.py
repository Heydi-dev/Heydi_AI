# model/topic.py
# 코드 설명: 일기 내용으로부터 주제 태그를 추출하는 기능을 제공하는 모듈입니다.
# 로컬 KoBART 모델과 Google Gemini API를 모두 지원합니다.
import os
import torch
from google import genai
from google.genai import types
from dataclasses import dataclass
from transformers import PreTrainedTokenizerFast, BartForConditionalGeneration

# ==========================================
# 1. 데이터 구조 정의 (Topic)
# ==========================================
@dataclass
class Topic:
    count: int
    tag1: str
    tag2: str = "" # 기본값 설정

# ==========================================
# 2. Gemini (기존 코드) 설정
# ==========================================
def extract_topics_gemini(text: str) -> Topic:
    """기존 Gemini API를 사용한 태그 추출"""
    try:
        client = genai.Client()
        config = types.GenerateContentConfig(
                system_instruction="You're a summary bot. Create a summary tag for the given content in one or two words in korean. the number of tags should be 1 or 2. structure: (number of tags),(tag1),(tag2),... if there is no tag2, leave it blank. example: 2,기술,인공지능",
            )

        response = client.models.generate_content(
            model="gemini-2.5-flash",
            config=config,
            contents=text
        )

        topic = Topic(count=0, tag1="", tag2="")
        s = response.text.strip().split(",")
        
        # 파싱 로직 유지
        if len(s) >= 2:
            topic.count = int(s[0])
            topic.tag1 = s[1]
            topic.tag2 = s[2] if topic.count == 2 and len(s) > 2 else ""
        
        return topic
    
    except Exception as e:
        print(f"Gemini Error: {e}")
        return Topic(0, "Error", "")

# ==========================================
# 3. KoBART (로컬 모델) 설정
# ==========================================
# 전역 변수 (Lazy Loading)
_kobart_tokenizer = None
_kobart_model = None
_device = None

# 경로 설정: 현재 파일 위치 기준으로 모델 경로 탐색
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# 1순위: 같은 폴더 내 model/final_diary_tag_model (루트에서 실행 시)
# 2순위: 현재 폴더 내 final_diary_tag_model (model 폴더 내에서 실행 시)
MODEL_PATHS = [
    os.path.join(BASE_DIR, "model", "final_diary_tag_model"),
    os.path.join(BASE_DIR, "final_diary_tag_model")
]

def _load_kobart_model():
    """모델이 로드되어 있지 않을 때만 로드하는 내부 함수"""
    global _kobart_tokenizer, _kobart_model, _device
    
    if _kobart_model is not None:
        return

    # 유효한 모델 경로 찾기
    model_path = next((path for path in MODEL_PATHS if os.path.exists(path)), None)
    
    if model_path is None:
        raise FileNotFoundError(f"❌ 모델을 찾을 수 없습니다. 다음 경로들을 확인해주세요: {MODEL_PATHS}")

    try:
        # print(f">>> [System] Local Model loading... ({model_path})")
        _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        _kobart_tokenizer = PreTrainedTokenizerFast.from_pretrained(model_path)
        _kobart_model = BartForConditionalGeneration.from_pretrained(model_path)
        _kobart_model.to(_device)
        _kobart_model.eval()
        # print(f">>> [System] Loaded on {_device}")
        
    except Exception as e:
        print(f"❌ 모델 로드 중 오류 발생: {e}")
        raise e

def extract_topics_kobart(text: str) -> Topic:
    """학습된 KoBART 모델을 사용한 태그 추출"""
    _load_kobart_model()
    
    # 1. 전처리 및 토큰화
    inputs = _kobart_tokenizer(
        text, 
        return_tensors="pt", 
        max_length=128, 
        truncation=True, 
        padding="max_length"
    )
    
    input_ids = inputs["input_ids"].to(_device)
    attention_mask = inputs["attention_mask"].to(_device)

    # 2. 모델 추론
    with torch.no_grad():
        summary_text_ids = _kobart_model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=32,
            num_beams=5,
            repetition_penalty=1.2,
            no_repeat_ngram_size=0,
            length_penalty=1.0,
            early_stopping=True
        )

    # 3. 디코딩 및 후처리
    output_str = _kobart_tokenizer.decode(summary_text_ids[0], skip_special_tokens=True)
    raw_tags = [t.strip() for t in output_str.split(',')]
    
    unique_tags = []
    seen = set()
    for t in raw_tags:
        if t and t not in seen:
            seen.add(t)
            unique_tags.append(t)
            
    # 4. Topic 객체 반환
    count = len(unique_tags)
    
    if count >= 2:
        return Topic(count=2, tag1=unique_tags[0], tag2=unique_tags[1])
    elif count == 1:
        return Topic(count=1, tag1=unique_tags[0], tag2="")
    else:
        return Topic(count=0, tag1="", tag2="")

# ==========================================
# 4. 통합 인터페이스 (선택 가능)
# ==========================================
def extract_topics_from_text(text: str, use_local: bool = True) -> Topic:
    """
    use_local=True 이면 KoBART(로컬) 사용
    use_local=False 이면 Gemini(API) 사용
    """
    if use_local:
        return extract_topics_kobart(text)
    else:
        return extract_topics_gemini(text)

# ==========================================
# 5. 실행 테스트
# ==========================================
if __name__ == "__main__":
    print("=== 태그 생성기 (Mode: local / gemini) ===")
    
    while True:
        text = input("\n일기 내용을 입력하세요 ('exit' 종료): ")
        if text.lower() == 'exit':
            break
            
        # 로컬 모델(KoBART) 우선 테스트
        # 1. 로컬 모델 결과
        try:
            local_topic = extract_topics_from_text(text, use_local=True)
            print(f"✅ [KoBART] : {local_topic.tag1}" + (f", {local_topic.tag2}" if local_topic.tag2 else ""))
        except Exception as e:
            print(f"❌ [KoBART] Error: {e}")

        # 2. Gemini 결과 (비교용, 필요 없으면 주석 처리)
        try:
            gemini_topic = extract_topics_from_text(text, use_local=False)
            print(f"✨ [Gemini] : {gemini_topic.tag1}, {gemini_topic.tag2}")
        except Exception as e:
            print(f"❌ [Gemini] Error: {e}")