import json
import re
from dataclasses import dataclass
from datetime import datetime
from typing import Iterable, Optional

import torch
from google import genai
from google.genai import types
from pydantic import BaseModel, Field
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# Preferences extraction pipeline:
# - score sentence emotion with KoELECTRA
# - anchor strong sentiment and build context chunks
# - sample evenly by date/week, de-duplicate similar chunks
# - constrain LLM to literal keywords with multi-date evidence
EMOTION_MODEL_NAME = "LimYeri/HowRU-KoELECTRA-Emotion-Classifier"

POS_LABEL_HINTS = [
    "기쁨",
    "행복",
    "즐거",
    "만족",
    "사랑",
    "감사",
    "고마",
    "희망",
    "기대",
    "설렘",
    "평온",
    "편안",
    "좋아",
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
    "두려",
    "공포",
    "혐오",
    "싫",
    "지침",
    "피곤",
    "스트레스",
    "긴장",
    "후회",
    "실망",
    "괴로",
    "답답",
    "좌절",
    "질투",
    "불쾌",
    "sad",
    "anger",
    "fear",
    "disgust",
]
NEU_LABEL_HINTS = ["중립", "무감", "보통", "평범", "neutral", "calm"]

BANNED_KEYWORDS = {
    "기쁨",
    "행복",
    "즐거움",
    "즐거움",
    "만족",
    "사랑",
    "감사",
    "고마움",
    "희망",
    "기대",
    "설렘",
    "평온",
    "편안",
    "좋음",
    "좋아",
    "불만",
    "짜증",
    "분노",
    "화",
    "슬픔",
    "우울",
    "불안",
    "두려움",
    "공포",
    "혐오",
    "싫음",
    "스트레스",
    "피곤",
    "지침",
    "무기력",
    "긴장",
    "후회",
    "실망",
    "괴로움",
    "답답함",
    "좌절",
    "질투",
    "불쾌",
    "중립",
}

SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")
TOKEN_RE = re.compile(r"[A-Za-z0-9가-힣]+")


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


class EvidenceItem(BaseModel):
    date: str = Field(description="Date of the supporting quote.")
    quote: str = Field(description="Exact quote from the chunk.")


class PreferenceItem(BaseModel):
    keyword: str = Field(description="Literal keyword found in chunks.")
    evidence: list[EvidenceItem]


class PreferencesResponse(BaseModel):
    like: PreferenceItem
    dislike: PreferenceItem


class EmotionScorer:
    def __init__(self) -> None:
        self._tokenizer = None
        self._model = None
        self._device = None
        self._labels = None
        self._label_groups = None

    def _load(self) -> None:
        if self._model is not None:
            return
        # Lazy-load to avoid model init on import.
        self._tokenizer = AutoTokenizer.from_pretrained(EMOTION_MODEL_NAME)
        self._model = AutoModelForSequenceClassification.from_pretrained(
            EMOTION_MODEL_NAME
        )
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._model.to(self._device)
        self._model.eval()
        self._labels = [
            self._model.config.id2label[i]
            for i in range(self._model.config.num_labels)
        ]
        self._label_groups = self._initialize_label_groups()

    def _initialize_label_groups(self) -> dict[str, str]:
        # Map model labels to pos/neg/neu; auto-calibrate if hints are insufficient.
        label_groups = {label: self._map_label(label) for label in self._labels}
        has_pos = any(group == "pos" for group in label_groups.values())
        has_neg = any(group == "neg" for group in label_groups.values())
        if not has_pos or not has_neg:
            label_groups = self._auto_calibrate_label_groups()
        return label_groups

    def _map_label(self, label: str) -> str:
        lowered = label.lower()
        if any(hint.lower() in lowered for hint in POS_LABEL_HINTS):
            return "pos"
        if any(hint.lower() in lowered for hint in NEG_LABEL_HINTS):
            return "neg"
        if any(hint.lower() in lowered for hint in NEU_LABEL_HINTS):
            return "neu"
        return "neu"

    def _predict_probs(self, texts: list[str]) -> list[list[float]]:
        # Shared inference path for batch scoring and calibration.
        encoded = self._tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=128,
        )
        encoded = {k: v.to(self._device) for k, v in encoded.items()}
        with torch.no_grad():
            logits = self._model(**encoded).logits
            probs = torch.softmax(logits, dim=-1).cpu().tolist()
        return probs

    def _auto_calibrate_label_groups(self) -> dict[str, str]:
        # Use seed sentences to learn which labels behave as pos/neg/neu.
        seeds = {
            "pos": [
                "\uC815\uB9D0 \uD589\uBCF5\uD574.",
                "\uC624\uB298\uC740 \uB9C8\uC74C\uC774 \uD3B8\uC548\uD574.",
                "\uB108\uBB34 \uC990\uAC70\uC6E0\uC5B4.",
            ],
            "neg": [
                "\uB108\uBB34 \uD654\uAC00 \uB098.",
                "\uC644\uC804\uD788 \uC2A4\uD2B8\uB808\uC2A4\uBC1B\uC558\uC5B4.",
                "\uC815\uB9D0 \uC2AC\uD504\uB2E4.",
            ],
            "neu": [
                "\uC624\uB298\uC740 \uD3C9\uBC94\uD55C \uD558\uB8E8\uC600\uC5B4.",
                "\uBCC4\uB2E4\uB978 \uC77C\uC740 \uC5C6\uC5C8\uC5B4.",
            ],
        }
        counts = {
            label: {"pos": 0, "neg": 0, "neu": 0} for label in self._labels
        }
        for group, texts in seeds.items():
            probs = self._predict_probs(texts)
            for prob in probs:
                top_idx = max(range(len(prob)), key=lambda idx: prob[idx])
                label = self._labels[top_idx]
                counts[label][group] += 1

        label_groups = {}
        for label, group_counts in counts.items():
            if sum(group_counts.values()) == 0:
                label_groups[label] = "neu"
            else:
                label_groups[label] = max(group_counts, key=group_counts.get)
        return label_groups

    def score(self, sentences: Iterable[str], batch_size: int = 16) -> list[dict]:
        self._load()
        results = []
        texts = list(sentences)
        for start in range(0, len(texts), batch_size):
            batch = texts[start : start + batch_size]
            probs = self._predict_probs(batch)
            for prob in probs:
                label_scores = {
                    self._labels[idx]: float(prob[idx])
                    for idx in range(len(self._labels))
                }
                group_scores = {"pos": 0.0, "neg": 0.0, "neu": 0.0}
                for label, score in label_scores.items():
                    group = self._label_groups.get(label, "neu")
                    group_scores[group] += score
                emotion = max(group_scores, key=group_scores.get)
                confidence = group_scores[emotion]
                top_label = max(label_scores, key=label_scores.get)
                results.append(
                    {
                        "emotion": emotion,
                        "confidence": confidence,
                        "label": top_label,
                    }
                )
        return results


class PreferencesService:
    def __init__(self) -> None:
        self._emotion_scorer = EmotionScorer()
        self._client = genai.Client()

    def extract_preferences(
        self,
        entries: list[dict],
        *,
        min_confidence: float = 0.6,
        max_anchors_per_day: int = 3,
        max_anchors_per_week: int = 8,
        max_chunk_chars: int = 360,
        max_pos_chunks: int = 6,
        max_neg_chunks: int = 6,
        max_per_date: int = 2,
        max_per_week: int = 4,
        similarity_threshold: float = 0.82,
    ) -> dict:
        if not entries:
            return {"error": "No diary entries provided."}

        sentences_by_date = {}
        for entry in entries:
            date = entry.get("date", "").strip()
            text = entry.get("text", "")
            if not date or not text:
                continue
            # Split raw diary text into sentence units for scoring.
            sentences = self._split_sentences(text)
            if not sentences:
                continue
            base_index = len(sentences_by_date.get(date, []))
            sentence_records = [
                SentenceRecord(date=date, index=base_index + idx, text=sentence)
                for idx, sentence in enumerate(sentences)
            ]
            sentences_by_date.setdefault(date, []).extend(sentence_records)

        all_sentences = [
            record for records in sentences_by_date.values() for record in records
        ]
        if not all_sentences:
            return {"error": "No valid sentences found."}

        scores = self._emotion_scorer.score([s.text for s in all_sentences])
        for record, score in zip(all_sentences, scores):
            record.emotion = score["emotion"]
            record.confidence = score["confidence"]
            record.label = score["label"]

        # Anchor high-confidence sentiment sentences.
        anchors = self._select_anchors(
            all_sentences,
            min_confidence=min_confidence,
            max_per_day=max_anchors_per_day,
            max_per_week=max_anchors_per_week,
        )
        if not anchors:
            return {"error": "No emotion anchors found after filtering."}

        # Build context chunks around anchors.
        chunks = self._build_chunks(
            anchors,
            sentences_by_date,
            max_chunk_chars=max_chunk_chars,
        )
        if not chunks:
            return {"error": "No context chunks generated."}

        pos_chunks = [chunk for chunk in chunks if chunk.emotion == "pos"]
        neg_chunks = [chunk for chunk in chunks if chunk.emotion == "neg"]

        # De-duplicate within polarity before balanced sampling.
        pos_chunks = self._dedupe_similar(pos_chunks, similarity_threshold)
        neg_chunks = self._dedupe_similar(neg_chunks, similarity_threshold)

        # Sample evenly by date/week to prevent single-event dominance.
        pos_samples = self._sample_chunks(
            pos_chunks,
            max_pos_chunks,
            max_per_date=max_per_date,
            max_per_week=max_per_week,
            similarity_threshold=similarity_threshold,
        )
        neg_samples = self._sample_chunks(
            neg_chunks,
            max_neg_chunks,
            max_per_date=max_per_date,
            max_per_week=max_per_week,
            similarity_threshold=similarity_threshold,
        )

        if not pos_samples or not neg_samples:
            return {
                "error": "Insufficient positive/negative chunks for extraction.",
                "positive_chunks": len(pos_samples),
                "negative_chunks": len(neg_samples),
            }

        # Constrained extraction based on provided chunks.
        response = self._call_llm(pos_samples, neg_samples)
        if "error" in response:
            return response

        # Final validation: literal keyword, evidence quotes, 2+ dates.
        validation = self._validate_output(response, pos_samples, neg_samples)
        if validation:
            return {"error": "Validation failed.", "details": validation}

        return response

    def _split_sentences(self, text: str) -> list[str]:
        cleaned = text.replace("\r", "\n").strip()
        if not cleaned:
            return []
        sentences = []
        for line in re.split(r"\n+", cleaned):
            line = line.strip()
            if not line:
                continue
            parts = SENTENCE_SPLIT_RE.split(line)
            for part in parts:
                part = part.strip()
                if part:
                    sentences.append(part)
        return sentences

    def _select_anchors(
        self,
        sentences: list[SentenceRecord],
        *,
        min_confidence: float,
        max_per_day: int,
        max_per_week: int,
    ) -> list[SentenceRecord]:
        candidates = [
            s
            for s in sentences
            if s.emotion in ("pos", "neg") and s.confidence >= min_confidence
        ]
        candidates.sort(key=lambda s: s.confidence, reverse=True)
        per_day = {}
        per_week = {}
        selected = []
        for sentence in candidates:
            day_key = sentence.date
            week_key = self._week_key(sentence.date)
            if per_day.get(day_key, 0) >= max_per_day:
                continue
            if per_week.get(week_key, 0) >= max_per_week:
                continue
            selected.append(sentence)
            per_day[day_key] = per_day.get(day_key, 0) + 1
            per_week[week_key] = per_week.get(week_key, 0) + 1
        return selected

    def _build_chunks(
        self,
        anchors: list[SentenceRecord],
        sentences_by_date: dict[str, list[SentenceRecord]],
        *,
        max_chunk_chars: int,
    ) -> list[Chunk]:
        chunks = []
        for anchor in anchors:
            date_sentences = sentences_by_date.get(anchor.date, [])
            if not date_sentences:
                continue
            start = max(0, anchor.index - 2)
            include_next = len(anchor.text) < 40
            end = anchor.index + 1 + (1 if include_next else 0)
            if end > len(date_sentences):
                end = len(date_sentences)
            # Anchor + previous two, optionally the next sentence.
            selected = date_sentences[start:end]
            text = " ".join(sentence.text for sentence in selected).strip()
            while len(text) > max_chunk_chars and len(selected) > 1:
                left_distance = anchor.index - start
                right_distance = (start + len(selected) - 1) - anchor.index
                if left_distance >= right_distance:
                    selected.pop(0)
                    start += 1
                else:
                    selected.pop()
                text = " ".join(sentence.text for sentence in selected).strip()
            if len(text) > max_chunk_chars:
                text = text[:max_chunk_chars].rstrip()
            if not text:
                continue
            chunks.append(
                Chunk(
                    date=anchor.date,
                    text=text,
                    emotion=anchor.emotion,
                    confidence=anchor.confidence,
                    anchor_index=anchor.index,
                )
            )
        seen = set()
        unique_chunks = []
        for chunk in chunks:
            key = re.sub(r"\s+", " ", chunk.text)
            if key in seen:
                continue
            seen.add(key)
            unique_chunks.append(chunk)
        return unique_chunks

    def _sample_chunks(
        self,
        chunks: list[Chunk],
        max_chunks: int,
        *,
        max_per_date: int,
        max_per_week: int,
        similarity_threshold: float,
    ) -> list[Chunk]:
        # Balance by date/week while preserving high-confidence chunks.
        by_date = {}
        for chunk in sorted(chunks, key=lambda c: c.confidence, reverse=True):
            by_date.setdefault(chunk.date, []).append(chunk)

        selected = []
        per_date = {}
        per_week = {}
        tokens_cache = []

        while len(selected) < max_chunks:
            candidates = []
            for date, items in by_date.items():
                if not items:
                    continue
                if per_date.get(date, 0) >= max_per_date:
                    continue
                week_key = self._week_key(date)
                if per_week.get(week_key, 0) >= max_per_week:
                    continue
                candidates.append(date)
            if not candidates:
                break
            candidates.sort(
                key=lambda date: (per_date.get(date, 0), -by_date[date][0].confidence)
            )
            made_selection = False
            for date in candidates:
                if not by_date[date]:
                    continue
                chunk = by_date[date].pop(0)
                tokens = self._tokenize(chunk.text)
                if any(
                    self._jaccard(tokens, existing) >= similarity_threshold
                    for existing in tokens_cache
                ):
                    continue
                selected.append(chunk)
                tokens_cache.append(tokens)
                per_date[date] = per_date.get(date, 0) + 1
                week_key = self._week_key(date)
                per_week[week_key] = per_week.get(week_key, 0) + 1
                made_selection = True
                break
            if not made_selection:
                break
        return selected

    def _dedupe_similar(self, chunks: list[Chunk], threshold: float) -> list[Chunk]:
        # Remove near-duplicate chunks using token Jaccard similarity.
        selected = []
        tokens_cache = []
        for chunk in sorted(chunks, key=lambda c: c.confidence, reverse=True):
            tokens = self._tokenize(chunk.text)
            if any(
                self._jaccard(tokens, existing) >= threshold for existing in tokens_cache
            ):
                continue
            tokens_cache.append(tokens)
            selected.append(chunk)
        return selected

    def _call_llm(self, pos_chunks: list[Chunk], neg_chunks: list[Chunk]) -> dict:
        # Serialize chunks and rules for strict extraction.
        def to_payload(chunks: list[Chunk], prefix: str) -> list[dict]:
            payload = []
            for idx, chunk in enumerate(chunks, start=1):
                payload.append(
                    {
                        "id": f"{prefix}{idx}",
                        "date": chunk.date,
                        "text": chunk.text,
                    }
                )
            return payload

        prompt = {
            "positive_chunks": to_payload(pos_chunks, "P"),
            "negative_chunks": to_payload(neg_chunks, "N"),
            "rules": {
                "keyword_literal": "Keywords must appear exactly in chunk text.",
                "no_emotion_words": sorted(BANNED_KEYWORDS),
                "min_evidence_dates": 2,
                "format": {
                    "like": {
                        "keyword": "string",
                        "evidence": [{"date": "YYYY-MM-DD", "quote": "string"}],
                    },
                    "dislike": {
                        "keyword": "string",
                        "evidence": [{"date": "YYYY-MM-DD", "quote": "string"}],
                    },
                },
            },
        }

        response = self._client.models.generate_content(
            model="gemini-2.5-flash",
            config=types.GenerateContentConfig(
                system_instruction=(
                    "You are a strict extraction engine. Use only the provided chunks. "
                    "Select exactly one liked keyword from positive chunks and one disliked "
                    "keyword from negative chunks. The keyword must be a literal substring in "
                    "at least two chunks with different dates. Do not output emotion/state words "
                    "listed in no_emotion_words. Output valid JSON only."
                ),
                temperature=0.2,
                #max_output_tokens=2048,
                response_mime_type="application/json",
                response_json_schema=PreferencesResponse.model_json_schema(),
            ),
            contents=json.dumps(prompt, ensure_ascii=False),
        )
        text = (response.text or "").strip()
        parsed = self._parse_json(text)
        if parsed is None:
            return {"error": "LLM response is not valid JSON.", "raw": text}
        return parsed

    def _parse_json(self, text: str) -> Optional[dict]:
        cleaned = text.strip()
        if cleaned.startswith("```"):
            cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
            cleaned = re.sub(r"\s*```$", "", cleaned)
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            match = re.search(r"\{.*\}", cleaned, re.DOTALL)
            if not match:
                return None
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                return None

    def _validate_output(
        self,
        response: dict,
        pos_chunks: list[Chunk],
        neg_chunks: list[Chunk],
    ) -> list[str]:
        # Enforce literal keywords and multi-date evidence.
        errors = []
        like = response.get("like")
        dislike = response.get("dislike")
        if not like or not dislike:
            return ["Missing like/dislike fields."]

        for key, chunks in (("like", pos_chunks), ("dislike", neg_chunks)):
            payload = response.get(key, {})
            keyword = (payload.get("keyword") or "").strip()
            evidence = payload.get("evidence") or []
            if not keyword:
                errors.append(f"{key}: keyword missing.")
                continue
            if keyword in BANNED_KEYWORDS:
                errors.append(f"{key}: keyword is an emotion/state word.")
            matched_dates = {
                chunk.date for chunk in chunks if keyword in chunk.text
            }
            if len(matched_dates) < 2:
                errors.append(f"{key}: keyword does not appear on 2 dates.")
            evidence_dates = set()
            for item in evidence:
                date = (item.get("date") or "").strip()
                quote = (item.get("quote") or "").strip()
                if not date or not quote:
                    errors.append(f"{key}: evidence item missing fields.")
                    continue
                if keyword not in quote:
                    errors.append(f"{key}: evidence quote missing keyword.")
                if not any(
                    chunk.date == date and quote in chunk.text for chunk in chunks
                ):
                    errors.append(f"{key}: evidence quote not found in chunks.")
                evidence_dates.add(date)
            if len(evidence_dates) < 2:
                errors.append(f"{key}: evidence dates fewer than 2.")
        return errors

    def _tokenize(self, text: str) -> set[str]:
        return {token.lower() for token in TOKEN_RE.findall(text)}

    def _jaccard(self, a: set[str], b: set[str]) -> float:
        if not a or not b:
            return 0.0
        return len(a & b) / len(a | b)

    def _week_key(self, date_str: str) -> str:
        parsed = self._parse_date(date_str)
        if not parsed:
            return date_str
        year, week, _ = parsed.isocalendar()
        return f"{year}-W{week:02d}"

    def _parse_date(self, date_str: str) -> Optional[datetime]:
        for fmt in ("%Y-%m-%d", "%Y.%m.%d", "%Y/%m/%d", "%Y-%m-%d %H:%M:%S"):
            try:
                return datetime.strptime(date_str, fmt)
            except ValueError:
                continue
        return None
