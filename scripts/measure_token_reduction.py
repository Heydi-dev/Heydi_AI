from __future__ import annotations

import math
import re
from dataclasses import dataclass
from datetime import datetime

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


MODEL_NAME = "LimYeri/HowRU-KoELECTRA-Emotion-Classifier"
MIN_CONFIDENCE = 0.6
MAX_ANCHORS_PER_DAY = 3
MAX_ANCHORS_PER_WEEK = 8
MAX_CHUNK_CHARS = 360
MAX_POS_CHUNKS = 6
MAX_NEG_CHUNKS = 6
MAX_PER_DATE = 2
MAX_PER_WEEK = 4
SIMILARITY_THRESHOLD = 0.82

SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")
TOKEN_RE = re.compile(r"[A-Za-z0-9가-힣]+")

POS_LABEL_HINTS = [
    "기쁨",
    "행복",
    "즐거",
    "만족",
    "사랑",
    "감사",
    "고마",
    "설렘",
    "기대",
    "편안",
    "좋",
    "joy",
    "happy",
    "love",
]
NEG_LABEL_HINTS = [
    "분노",
    "화",
    "짜증",
    "불만",
    "슬픔",
    "우울",
    "불안",
    "걱정",
    "공포",
    "혐오",
    "지침",
    "피곤",
    "스트레스",
    "긴장",
    "후회",
    "외로",
    "괴로",
    "답답",
    "sad",
    "anger",
    "fear",
    "disgust",
]


SAMPLE_ENTRIES = [
    {
        "date": "2026-05-27",
        "text": (
            "아침에 일어나자마자 비가 와서 조금 느리게 움직였다. 출근길 버스가 평소보다 막혀서 회의 시간에 늦을까 봐 계속 시계를 봤다. "
            "그래도 회사에 도착해서 따뜻한 커피를 마시니 마음이 조금 가라앉았다. 오전에는 발표 자료의 흐름을 다시 정리했고, 팀원이 내 설명이 전보다 훨씬 자연스러워졌다고 말해줘서 안심이 됐다. "
            "점심에는 근처 식당에서 김치찌개를 먹었는데 생각보다 맛있어서 기분이 좋아졌다. 오후에는 버그 하나가 잘 잡히지 않아 꽤 답답했지만, 퇴근 전에 원인을 찾아서 마음이 후련했다. "
            "집에 돌아와서는 씻고 나서 책을 조금 읽었다. 하루가 아주 특별하진 않았지만, 막혔던 일을 해결하고 나니 나름 괜찮은 하루였다는 생각이 들었다."
        ),
    },
    {
        "date": "2026-05-28",
        "text": (
            "오늘은 오전부터 프로젝트 회의가 있어서 조금 긴장한 채로 하루를 시작했다. 어제 정리한 내용을 바탕으로 기능 흐름을 설명했는데, 다행히 팀원들이 대부분 이해해줬다. "
            "특히 음성 대화 이후 일기가 만들어지는 과정을 시연했을 때 반응이 좋아서 뿌듯했다. 회의가 끝난 뒤에는 해야 할 일이 많다는 사실이 다시 보였고, 잠깐 부담감이 올라왔다. "
            "그래도 일을 하나씩 나누어 적어보니 생각보다 감당할 수 있을 것 같았다. 저녁에는 친구와 짧게 통화했는데 별 이야기는 아니었지만 목소리를 들으니 마음이 편해졌다. "
            "오늘은 큰 사건보다 준비한 것을 설명하고 인정받은 느낌이 오래 남았다."
        ),
    },
    {
        "date": "2026-05-29",
        "text": (
            "아침부터 몸이 무거워서 집중이 잘 되지 않았다. 어제 늦게 잔 탓인지 커피를 마셔도 머리가 맑아지는 느낌은 별로 없었다. "
            "오전 업무는 단순한 정리 작업이 많아서 천천히 처리했다. 점심 이후에는 감정 분석 결과를 확인했는데 예상과 다른 문장들이 몇 개 있어서 원인을 살펴봤다. "
            "처음에는 짜증이 났지만, 모델이 어떤 문장을 헷갈려 하는지 보이기 시작하니 오히려 개선 방향이 조금 분명해졌다. 퇴근길에는 일부러 한 정거장 먼저 내려서 걸었다. "
            "바람이 시원했고, 이어폰으로 잔잔한 음악을 들으니 지친 마음이 조금 풀렸다. 오늘은 생산성이 높지는 않았지만, 무리하지 않고 버틴 하루였다."
        ),
    },
    {
        "date": "2026-05-30",
        "text": (
            "오전에는 집에서 발표 대본을 다시 읽어봤다. 처음에는 문장이 너무 기술적으로만 느껴져서 듣는 사람이 지루하지 않을까 걱정됐다. "
            "그래서 실제 사용자가 앱을 쓰는 장면을 먼저 설명하고, 그 뒤에 구현 방식을 붙이는 식으로 순서를 바꿔봤다. 바꾸고 나니 훨씬 말하기 편해졌다. "
            "점심에는 가족과 같이 밥을 먹었고, 오랜만에 여유 있게 이야기를 나눴다. 오후에는 카페에 가서 조용한 자리에 앉아 작업했는데, 적당한 소음이 오히려 집중에 도움이 됐다. "
            "저녁에는 조금 피곤했지만 오늘 정리한 발표 흐름이 마음에 들어서 뿌듯했다."
        ),
    },
    {
        "date": "2026-05-31",
        "text": (
            "오늘은 주말이라 늦잠을 잤다. 아침 겸 점심을 먹고 나서 밀린 빨래와 청소를 했다. 특별한 일은 없었지만 방이 정리되니 머릿속도 조금 정리되는 느낌이었다. "
            "오후에는 산책을 나갔는데 공원이 생각보다 사람이 많아서 오래 있지는 못했다. 사람 많은 곳에 있으니 조금 답답해서 조용한 골목길로 돌아왔다. "
            "집에 와서는 다음 주 일정표를 확인하고 해야 할 일을 적었다. 해야 할 일이 많아 살짝 부담스러웠지만, 적어두니 막연한 불안은 줄었다. "
            "오늘은 조용하고 느린 하루였고, 그런 속도가 나쁘지 않았다."
        ),
    },
    {
        "date": "2026-06-01",
        "text": (
            "월요일이라 그런지 아침부터 메시지가 많이 쌓여 있었다. 우선순위를 정해서 하나씩 처리했지만, 중간중간 새로운 요청이 들어와 흐름이 자주 끊겼다. "
            "오전에는 조금 정신없었고, 점심 전에는 괜히 예민해진 느낌도 있었다. 오후에는 팀원과 같이 오류 로그를 보면서 문제를 좁혀갔다. 혼자 볼 때는 막막했는데 같이 이야기하니 생각보다 빨리 실마리를 찾았다. "
            "작은 해결책을 찾고 나니 긴장이 풀렸고, 협업이 확실히 도움이 된다는 생각이 들었다. 저녁에는 집에서 간단히 밥을 먹고 일찍 쉬기로 했다."
        ),
    },
    {
        "date": "2026-06-02",
        "text": (
            "오늘은 발표 자료에 넣을 정량 평가 항목을 고민했다. 처음에는 어떤 지표를 잡아야 할지 애매했지만, 토큰 감소율과 근거 일치율처럼 설명하기 쉬운 지표부터 정리하기로 했다. "
            "자료를 만들다 보니 우리가 구현한 기능의 장점이 생각보다 분명하게 보였다. 특히 감정 필터링으로 LLM 입력을 줄이는 구조는 발표에서 잘 설명하면 좋을 것 같다. "
            "오후에는 테스트 데이터를 정리했고, 너무 인위적으로 보이지 않도록 실제 사용자가 쓸 법한 문장을 골랐다. 반복 작업이 많아 조금 지루했지만, 결과물이 쌓이는 느낌은 좋았다. "
            "오늘은 차분하게 진전이 있었던 하루였다."
        ),
    },
    {
        "date": "2026-06-03",
        "text": (
            "아침에 발표 연습을 해봤는데 시간이 예상보다 길었다. 설명하고 싶은 내용이 많아서 자꾸 사족이 붙는 것 같았다. 핵심만 남기려고 문장을 줄였지만, 줄이다 보니 중요한 기술적 설명까지 빠지는 느낌이 들어 균형을 잡기가 어려웠다. "
            "점심 이후에는 팀원에게 자료 초안을 보여줬고, 전체 흐름은 괜찮지만 구현 방식이 조금 더 보여야 한다는 피드백을 받았다. 처음에는 다시 고쳐야 한다는 생각에 살짝 지쳤지만, 피드백을 반영하니 확실히 자료가 좋아졌다. "
            "저녁에는 좋아하는 라멘집에 가서 밥을 먹었다. 따뜻한 국물을 먹으니 하루 종일 긴장했던 몸이 조금 풀렸다."
        ),
    },
    {
        "date": "2026-06-04",
        "text": (
            "오늘은 서버 배포 상태를 확인했다. 오전에는 별문제 없어 보였는데, 중간에 환경 변수 설정이 맞지 않아 잠깐 당황했다. 로그를 확인하면서 원인을 찾았고, 설정 파일을 다시 맞추니 정상적으로 동작했다. "
            "문제를 해결하고 나니 마음이 놓였지만, 배포 관련 설정은 작은 실수도 바로 티가 나서 늘 긴장된다. 오후에는 API 응답을 몇 번 더 확인했고, 실시간 대화 연결도 테스트했다. "
            "음성이 자연스럽게 오가고 전사 결과가 표시되는 것을 보니 꽤 만족스러웠다. 오늘은 피곤했지만 실제 서비스처럼 돌아가는 모습을 확인해서 보람이 있었다."
        ),
    },
    {
        "date": "2026-06-05",
        "text": (
            "오늘은 비교적 평온한 하루였다. 오전에는 어제 남겨둔 문서를 정리했고, 오후에는 팀 회의에서 발표 순서를 맞춰봤다. 각자 맡은 부분이 조금씩 달라서 처음에는 연결이 어색했지만, 앞뒤 문장을 조정하니 자연스러워졌다. "
            "회의가 끝난 뒤에는 근처 카페에서 잠깐 쉬었다. 창가 자리에 앉아 노트북을 닫고 멍하니 밖을 보는데, 며칠 동안 계속 긴장했던 마음이 조금 느슨해졌다. "
            "저녁에는 가족과 산책을 했다. 특별히 많이 웃은 날은 아니지만, 조용히 안정감을 느낀 하루였다."
        ),
    },
    {
        "date": "2026-06-06",
        "text": (
            "아침에는 늦게 일어나서 천천히 식사를 했다. 쉬는 날이라 마음이 가벼울 줄 알았는데, 발표가 얼마 남지 않았다는 생각이 자꾸 떠올라 완전히 편하지는 않았다. "
            "오후에는 집 근처 도서관에 가서 자료를 다시 읽었다. 조용한 공간에 앉아 있으니 머릿속이 조금 정리됐고, 복잡하게 느껴졌던 설명도 몇 문장으로 줄일 수 있었다. "
            "중간에 집중이 흐트러져 잠깐 산책을 했는데, 햇빛이 좋아서 기분이 조금 나아졌다. 저녁에는 더 손대지 않고 쉬기로 했다. 오늘은 불안이 조금 있었지만, 차분히 정리하면서 버틴 하루였다."
        ),
    },
    {
        "date": "2026-06-07",
        "text": (
            "오늘은 오전부터 팀원들과 온라인으로 발표 리허설을 했다. 처음에는 서로 말하는 속도가 달라서 흐름이 끊겼고, 화면 전환 타이밍도 몇 번 어긋났다. "
            "그래도 한 번씩 다시 맞춰보니 점점 자연스러워졌다. 내가 맡은 AI 구현 설명 부분은 생각보다 시간이 짧게 나와서, 토큰 감소율 측정 결과를 조금 더 명확히 넣기로 했다. "
            "회의가 끝난 뒤에는 피곤했지만 큰 방향이 잡힌 느낌이라 안도감이 컸다. 저녁에는 매운 떡볶이를 먹었는데 스트레스가 풀리는 것 같았다. 오늘은 힘들었지만 팀으로 맞춰가는 과정이 의미 있었다."
        ),
    },
    {
        "date": "2026-06-08",
        "text": (
            "월요일이라 다시 업무 분위기로 돌아왔다. 오전에는 지난주에 미뤄둔 메일을 정리했고, 생각보다 답해야 할 내용이 많아서 조금 부담스러웠다. "
            "점심 이후에는 기억 검색 기능을 다시 테스트했다. 사용자가 말한 취향이나 일정이 제대로 조회되는 것을 확인했을 때는 꽤 만족스러웠다. "
            "다만 일부 질문에서는 원하는 기억이 바로 나오지 않아 검색어 처리 방식을 더 고민해야겠다고 느꼈다. 퇴근길에는 지하철이 붐벼서 몸이 많이 지쳤다. 집에 와서 조용히 음악을 틀어두니 겨우 긴장이 풀렸다."
        ),
    },
    {
        "date": "2026-06-09",
        "text": (
            "오늘은 발표 자료 디자인을 다듬었다. 기능 설명이 너무 많아 보이지 않도록 핵심 문장만 남기고, 흐름도와 표를 중심으로 정리했다. "
            "처음에는 줄이는 과정이 아까웠지만, 막상 정리하고 보니 훨씬 보기 좋아졌다. 점심에는 팀원과 같이 샌드위치를 먹으면서 발표 순서를 다시 이야기했다. "
            "서로 놓친 부분을 자연스럽게 보완해줘서 고마웠다. 오후에는 피드백 문장을 몇 개 수정했고, 마지막 슬라이드의 결론 문구도 정했다. 오늘은 크게 흔들리지 않고 필요한 일을 끝낸 느낌이라 마음이 안정적이었다."
        ),
    },
    {
        "date": "2026-06-10",
        "text": (
            "아침부터 컨디션이 조금 좋지 않았다. 목이 살짝 잠겨서 발표 연습을 길게 하기가 어려웠고, 괜히 내일 더 나빠지면 어쩌나 걱정됐다. "
            "그래도 중요한 부분만 짧게 읽어보며 발음이 꼬이는 문장을 체크했다. 오후에는 정량 평가 표를 다시 확인했고, 수치를 과장하지 않도록 표현을 조심스럽게 바꿨다. "
            "실제 운영 비용 절감이라고 말하기보다는 샘플 기준 입력량 감소라고 설명하는 편이 맞다고 느꼈다. 저녁에는 따뜻한 차를 마시고 일찍 쉬었다. 오늘은 몸이 따라주지 않아 답답했지만, 무리하지 않으려고 했다."
        ),
    },
    {
        "date": "2026-06-11",
        "text": (
            "오늘은 최종 리허설을 했다. 발표 시간이 거의 맞았고, 팀원들의 설명도 전보다 훨씬 부드럽게 이어졌다. 내 차례에서는 장기 기억과 Function Calling을 설명할 때 조금 빨라지는 습관이 있어서 의식적으로 속도를 늦췄다. "
            "교수님이 들었을 때 구현 방식이 보이도록 데이터 흐름을 먼저 말하고, 그다음 차별점을 설명하는 순서로 정리했다. 리허설이 끝나고 나니 긴장이 풀리면서 갑자기 피곤해졌다. "
            "그래도 마지막까지 방향을 맞춘 느낌이 들어서 뿌듯했다. 저녁에는 가볍게 산책하고 일찍 들어왔다."
        ),
    },
    {
        "date": "2026-06-12",
        "text": (
            "발표 당일이라 아침부터 꽤 긴장됐다. 자료를 여러 번 확인했는데도 빠뜨린 것이 있을까 봐 계속 슬라이드를 넘겨봤다. "
            "막상 발표가 시작되니 생각보다 말이 잘 나왔고, 음성 대화에서 일기가 만들어지는 흐름을 설명할 때 청중이 고개를 끄덕이는 것이 보여서 조금 안심됐다. "
            "질문 시간에는 기억 기능의 신뢰성에 대한 질문이 나왔고, Function Calling과 DB 기반 저장 구조를 설명했다. 완벽하게 답했는지는 모르겠지만 준비한 만큼은 말한 것 같다. "
            "끝나고 나니 긴장이 한꺼번에 풀렸고, 팀원들과 같이 밥을 먹으며 서로 고생했다고 말했다."
        ),
    },
    {
        "date": "2026-06-13",
        "text": (
            "발표가 끝난 다음 날이라 그런지 몸이 무겁게 느껴졌다. 오전에는 별다른 일을 하지 않고 늦게까지 쉬었다. 그동안 계속 신경 쓰던 일이 끝나서 마음은 편했지만, 갑자기 목표가 사라진 것처럼 조금 허전하기도 했다. "
            "오후에는 발표 피드백을 다시 읽어봤다. 개인화 구조와 감정 분석 설명은 좋았지만, 실제 사용자 데이터 평가가 더 있으면 좋겠다는 의견이 있었다. "
            "아쉬움도 있었지만 다음에 개선할 방향이 분명해졌다고 생각했다. 저녁에는 오랜만에 영화를 봤고, 아무 생각 없이 웃을 수 있어서 좋았다."
        ),
    },
    {
        "date": "2026-06-14",
        "text": (
            "오늘은 프로젝트를 정리하는 마음으로 남은 파일들을 확인했다. 필요 없는 로그와 임시 메모가 많아서 정리하는 데 시간이 꽤 걸렸다. "
            "처음에는 귀찮았지만, 폴더가 깔끔해질수록 마음도 가벼워졌다. 오후에는 다음에 추가하면 좋을 기능을 적어봤다. 실제 사용자 일기 기반 평가, 기억 검색 정확도 개선, 취향 분석 결과 시각화 같은 아이디어가 떠올랐다. "
            "한동안 바쁘게 달려와서 그런지 아직은 새 기능을 바로 만들고 싶지는 않았지만, 기록해두니 나중에 다시 시작할 수 있을 것 같았다. 오늘은 조용히 마무리하는 하루였다."
        ),
    },
    {
        "date": "2026-06-15",
        "text": (
            "아침에는 평소보다 일찍 일어나서 산책을 했다. 공기가 맑고 길이 한산해서 기분이 좋았다. 며칠 동안 발표 생각만 하다가 오랜만에 주변을 천천히 본 느낌이었다. "
            "오전에는 회고 문서를 작성했다. 잘한 점보다 아쉬운 점이 먼저 떠올라 조금 씁쓸했지만, 그래도 구현 과정에서 배운 것이 많다는 생각이 들었다. "
            "특히 기능을 만드는 것만큼 그 기능의 효과를 어떻게 보여줄지 고민하는 일이 중요하다는 걸 느꼈다. 저녁에는 좋아하는 카페에 들러 아이스 아메리카노를 마셨다. 편안하게 하루를 마무리할 수 있어서 좋았다."
        ),
    },
]


@dataclass
class SentenceRecord:
    date: str
    index: int
    text: str
    emotion: str = "neu"
    confidence: float = 0.0
    label: str = ""


@dataclass
class Chunk:
    date: str
    text: str
    emotion: str
    confidence: float
    anchor_index: int


def split_sentences(text: str) -> list[str]:
    sentences: list[str] = []
    for line in re.split(r"\n+", text.replace("\r", "\n").strip()):
        line = line.strip()
        if not line:
            continue
        for part in SENTENCE_SPLIT_RE.split(line):
            part = part.strip()
            if part:
                sentences.append(part)
    return sentences


def week_key(date_str: str) -> str:
    parsed = datetime.strptime(date_str, "%Y-%m-%d")
    year, week, _ = parsed.isocalendar()
    return f"{year}-W{week:02d}"


def tokenize(text: str) -> set[str]:
    return {token.lower() for token in TOKEN_RE.findall(text)}


def jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def estimate_llm_tokens(text: str) -> int:
    # Gemini tokenizer is not available locally, so use a stable token-equivalent
    # estimate for before/after comparison. The same estimator is applied to both.
    if not text:
        return 0
    return math.ceil(len(text.encode("utf-8")) / 4)


class EmotionScorer:
    def __init__(self) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        self.model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()
        self.labels = [
            self.model.config.id2label[i] for i in range(self.model.config.num_labels)
        ]
        self.label_groups = self._initialize_label_groups()

    def _initialize_label_groups(self) -> dict[str, str]:
        label_groups = {label: self._map_label(label) for label in self.labels}
        if any(group == "pos" for group in label_groups.values()) and any(
            group == "neg" for group in label_groups.values()
        ):
            return label_groups
        return self._auto_calibrate_label_groups()

    def _map_label(self, label: str) -> str:
        lowered = label.lower()
        if any(hint.lower() in lowered for hint in POS_LABEL_HINTS):
            return "pos"
        if any(hint.lower() in lowered for hint in NEG_LABEL_HINTS):
            return "neg"
        return "neu"

    def _predict_probs(self, texts: list[str]) -> list[list[float]]:
        encoded = self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=128,
        )
        encoded = {key: value.to(self.device) for key, value in encoded.items()}
        with torch.no_grad():
            logits = self.model(**encoded).logits
            return torch.softmax(logits, dim=-1).cpu().tolist()

    def _auto_calibrate_label_groups(self) -> dict[str, str]:
        seeds = {
            "pos": ["정말 행복했다.", "마음이 편안했다.", "너무 즐거웠다."],
            "neg": ["너무 화가 났다.", "스트레스를 많이 받았다.", "정말 슬펐다."],
            "neu": ["오늘은 평범한 하루였다.", "별다른 일은 없었다."],
        }
        counts = {label: {"pos": 0, "neg": 0, "neu": 0} for label in self.labels}
        for group, texts in seeds.items():
            for prob in self._predict_probs(texts):
                label = self.labels[max(range(len(prob)), key=lambda idx: prob[idx])]
                counts[label][group] += 1
        return {
            label: max(group_counts, key=group_counts.get)
            if sum(group_counts.values())
            else "neu"
            for label, group_counts in counts.items()
        }

    def score(self, texts: list[str], batch_size: int = 16) -> list[dict]:
        results: list[dict] = []
        for start in range(0, len(texts), batch_size):
            batch = texts[start : start + batch_size]
            for prob in self._predict_probs(batch):
                label_scores = {
                    self.labels[idx]: float(prob[idx]) for idx in range(len(self.labels))
                }
                group_scores = {"pos": 0.0, "neg": 0.0, "neu": 0.0}
                for label, score in label_scores.items():
                    group_scores[self.label_groups.get(label, "neu")] += score
                emotion = max(group_scores, key=group_scores.get)
                top_label = max(label_scores, key=label_scores.get)
                results.append(
                    {
                        "emotion": emotion,
                        "confidence": group_scores[emotion],
                        "label": top_label,
                    }
                )
        return results


def select_anchors(sentences: list[SentenceRecord]) -> list[SentenceRecord]:
    candidates = [
        sentence
        for sentence in sentences
        if sentence.emotion in ("pos", "neg")
        and sentence.confidence >= MIN_CONFIDENCE
    ]
    candidates.sort(key=lambda sentence: sentence.confidence, reverse=True)

    per_day: dict[str, int] = {}
    per_week: dict[str, int] = {}
    selected: list[SentenceRecord] = []
    for sentence in candidates:
        date_key = sentence.date
        week = week_key(sentence.date)
        if per_day.get(date_key, 0) >= MAX_ANCHORS_PER_DAY:
            continue
        if per_week.get(week, 0) >= MAX_ANCHORS_PER_WEEK:
            continue
        selected.append(sentence)
        per_day[date_key] = per_day.get(date_key, 0) + 1
        per_week[week] = per_week.get(week, 0) + 1
    return selected


def build_chunks(
    anchors: list[SentenceRecord],
    sentences_by_date: dict[str, list[SentenceRecord]],
) -> list[Chunk]:
    chunks: list[Chunk] = []
    for anchor in anchors:
        date_sentences = sentences_by_date.get(anchor.date, [])
        start = max(0, anchor.index - 2)
        include_next = len(anchor.text) < 40
        end = min(len(date_sentences), anchor.index + 1 + (1 if include_next else 0))
        selected = date_sentences[start:end]
        text = " ".join(sentence.text for sentence in selected).strip()
        while len(text) > MAX_CHUNK_CHARS and len(selected) > 1:
            left_distance = anchor.index - start
            right_distance = (start + len(selected) - 1) - anchor.index
            if left_distance >= right_distance:
                selected.pop(0)
                start += 1
            else:
                selected.pop()
            text = " ".join(sentence.text for sentence in selected).strip()
        if len(text) > MAX_CHUNK_CHARS:
            text = text[:MAX_CHUNK_CHARS].rstrip()
        if text:
            chunks.append(
                Chunk(
                    date=anchor.date,
                    text=text,
                    emotion=anchor.emotion,
                    confidence=anchor.confidence,
                    anchor_index=anchor.index,
                )
            )

    seen: set[str] = set()
    unique_chunks: list[Chunk] = []
    for chunk in chunks:
        key = re.sub(r"\s+", " ", chunk.text)
        if key in seen:
            continue
        seen.add(key)
        unique_chunks.append(chunk)
    return unique_chunks


def dedupe_similar(chunks: list[Chunk]) -> list[Chunk]:
    selected: list[Chunk] = []
    token_cache: list[set[str]] = []
    for chunk in sorted(chunks, key=lambda item: item.confidence, reverse=True):
        tokens = tokenize(chunk.text)
        if any(jaccard(tokens, existing) >= SIMILARITY_THRESHOLD for existing in token_cache):
            continue
        token_cache.append(tokens)
        selected.append(chunk)
    return selected


def sample_chunks(chunks: list[Chunk], max_chunks: int) -> list[Chunk]:
    by_date: dict[str, list[Chunk]] = {}
    for chunk in sorted(chunks, key=lambda item: item.confidence, reverse=True):
        by_date.setdefault(chunk.date, []).append(chunk)

    selected: list[Chunk] = []
    per_date: dict[str, int] = {}
    per_week: dict[str, int] = {}
    token_cache: list[set[str]] = []

    while len(selected) < max_chunks:
        candidates = [
            date
            for date, items in by_date.items()
            if items
            and per_date.get(date, 0) < MAX_PER_DATE
            and per_week.get(week_key(date), 0) < MAX_PER_WEEK
        ]
        if not candidates:
            break
        candidates.sort(
            key=lambda date: (per_date.get(date, 0), -by_date[date][0].confidence)
        )

        made_selection = False
        for date in candidates:
            chunk = by_date[date].pop(0)
            tokens = tokenize(chunk.text)
            if any(jaccard(tokens, existing) >= SIMILARITY_THRESHOLD for existing in token_cache):
                continue
            selected.append(chunk)
            token_cache.append(tokens)
            per_date[date] = per_date.get(date, 0) + 1
            week = week_key(date)
            per_week[week] = per_week.get(week, 0) + 1
            made_selection = True
            break
        if not made_selection:
            break
    return selected


def main() -> None:
    sentences_by_date: dict[str, list[SentenceRecord]] = {}
    for entry in SAMPLE_ENTRIES:
        date = entry["date"]
        sentence_texts = split_sentences(entry["text"])
        sentences_by_date[date] = [
            SentenceRecord(date=date, index=index, text=text)
            for index, text in enumerate(sentence_texts)
        ]

    all_sentences = [
        sentence for sentences in sentences_by_date.values() for sentence in sentences
    ]
    scorer = EmotionScorer()
    scores = scorer.score([sentence.text for sentence in all_sentences])
    for sentence, score in zip(all_sentences, scores):
        sentence.emotion = score["emotion"]
        sentence.confidence = score["confidence"]
        sentence.label = score["label"]

    anchors = select_anchors(all_sentences)
    chunks = build_chunks(anchors, sentences_by_date)
    pos_chunks = dedupe_similar([chunk for chunk in chunks if chunk.emotion == "pos"])
    neg_chunks = dedupe_similar([chunk for chunk in chunks if chunk.emotion == "neg"])
    selected_chunks = sample_chunks(pos_chunks, MAX_POS_CHUNKS) + sample_chunks(
        neg_chunks, MAX_NEG_CHUNKS
    )

    full_text = "\n".join(f"[{entry['date']}] {entry['text']}" for entry in SAMPLE_ENTRIES)
    filtered_text = "\n".join(
        f"[{chunk.date}] {chunk.text}" for chunk in selected_chunks
    )

    full_tokens = estimate_llm_tokens(full_text)
    filtered_tokens = estimate_llm_tokens(filtered_text)
    reduction_rate = (full_tokens - filtered_tokens) / full_tokens * 100

    print("=== Token Reduction Measurement ===")
    print(f"sample_entries: {len(SAMPLE_ENTRIES)}")
    print(f"total_sentences: {len(all_sentences)}")
    print(f"selected_emotion_anchors: {len(anchors)}")
    print(f"selected_llm_chunks: {len(selected_chunks)}")
    print(f"full_text_chars: {len(full_text)}")
    print(f"filtered_text_chars: {len(filtered_text)}")
    print(f"estimated_full_tokens: {full_tokens}")
    print(f"estimated_filtered_tokens: {filtered_tokens}")
    print(f"estimated_token_reduction_rate: {reduction_rate:.2f}%")
    print()
    print("=== Selected LLM Chunks ===")
    for index, chunk in enumerate(selected_chunks, start=1):
        print(
            f"{index}. [{chunk.date}] {chunk.emotion} "
            f"confidence={chunk.confidence:.4f} text={chunk.text}"
        )


if __name__ == "__main__":
    main()
