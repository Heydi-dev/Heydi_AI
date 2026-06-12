# Gemini Live STT CER Measurement

## Purpose

Gemini Live의 `input transcription`이 한국어 자연 발화를 얼마나 정확하게 전사하는지 확인하기 위해 CER(Character Error Rate)을 측정했다.

측정 방식은 다음과 같다.

- 입력: `sample_long.wav`
- 정답: `sample_long.txt`
- STT 결과: Gemini Live WebSocket에서 반환된 `input transcription`
- 비교 기준: 공백과 문장부호를 제거한 문자 단위 비교

## Sample Setup

- 음성 샘플: `tests/sample_long.wav`
- 정답 텍스트: `tests/sample_long.txt`
- 발화 길이: 약 24.12초
- 발화 내용: 발표 준비, 팀원과의 자료 확인, 카페에서의 휴식, 저녁 회고를 포함한 한국어 자연 발화

## Measurement Method

1. WAV 파일을 16kHz mono PCM으로 변환
2. 변환된 오디오를 WebSocket으로 Gemini Live 세션에 실시간 전송
3. 서버가 반환하는 `input transcription` 이벤트를 모두 수집
4. 수집된 전사 조각을 하나의 문장으로 결합
5. 정답 문장과 전사 결과에서 공백과 문장부호를 제거
6. 문자 단위 Levenshtein distance를 계산해 CER 산출

CER 계산식은 다음과 같다.

```text
CER = edit_distance / reference_character_count * 100
```

## Result

| Item | Value |
|---|---:|
| Audio duration | 24,120.0 ms |
| Audio send duration | 29,027.3 ms |
| Total elapsed time | 53,639.0 ms |
| Input transcription events | 59 |
| Normalized reference characters | 115 |
| Edit distance | 22 |
| CER | 19.13% |

## Reference

```text
오늘 아침부터 발표 준비하느라 좀 바쁘더라고. 그래도 팀원들과 같이하면서 자료도 찾아보고 하니까 빠진 부분을 찾을 수 있었어. 그리고 점심에는 카페에서 잠깐 쉬었는데 분위기가 조용해서 되게 편안했어. 저녁에도 할 일이 남아 있었지만 전체 방향이 잡힌 것 같아서 다행이라고 생각해.
```

## Hypothesis

```text
오늘아침부터발표준비하느라좀 바쁘더라고.그래도팀원들과같이하면서자료도찾아보고하니까빠진부분을찾을수 있었어.그리고점심에는카페에서 잠깐 쉬었는데의가조용해서 되게 편안했어.저녁에도할일이 남아 있었지만
```

## Interpretation

전사된 대부분의 구간은 의미가 유지되었다. 주요 오류는 단어가 전반적으로 틀린 것보다는 다음 두 가지에서 발생했다.

- 마지막 구간 누락: `전체 방향이 잡힌 것 같아서 다행이라고 생각해`
- 일부 단어 누락 또는 오인식: `분위기가` → `의가`

따라서 CER 19.13%는 전체 문장 기준의 문자 오류율이며, 전사된 구간의 의미 보존 정도와는 구분해서 해석할 필요가 있다.

## Presentation Summary

발표에서는 다음과 같이 설명할 수 있다.

> 약 24초 길이의 한국어 자연 발화를 Gemini Live input transcription으로 수집해 정답과 비교한 결과, 공백과 문장부호를 제외한 CER은 19.13%로 측정되었다. 전사된 대부분의 구간은 의미가 보존되었으며, 주요 오류는 마지막 발화 구간 누락과 일부 단어 누락에서 발생했다.
