# KoELECTRA Filter Token Reduction Measurement

## Purpose

KoELECTRA 감정 필터를 적용했을 때 LLM에 전달되는 입력량이 얼마나 줄어드는지 측정했다.

비교 방식은 다음과 같다.

- Before: 전체 일기 원문을 LLM에 전달한다고 가정
- After: KoELECTRA로 긍정/부정 감정이 강한 문장을 선별하고, 해당 문장 주변 문맥만 LLM에 전달한다고 가정

## Sample Setup

- 직접 작성한 일기 샘플: 20개
- 전체 문장 수: 143개
- 감정 변화가 지나치게 잦지 않도록 실제 사용자가 작성할 법한 하루 단위 일기로 구성
- 각 일기는 업무, 발표 준비, 회의, 휴식, 배포 확인, 팀 협업 등 자연스러운 일상 맥락을 포함

## Measurement Method

1. 일기 원문을 문장 단위로 분리
2. KoELECTRA 감정 분류 모델로 각 문장의 감정 분석
3. 긍정 또는 부정 감정이 강한 문장을 앵커로 선택
4. 앵커 문장 주변 문맥을 묶어 LLM 입력 후보 청크 생성
5. 날짜/주 단위 샘플링과 유사 문장 중복 제거 적용
6. 전체 원문과 필터링 후 청크의 입력량 비교

Gemini 공식 tokenizer를 로컬에서 직접 사용한 것은 아니며, 동일한 추정식으로 before/after를 비교했다.

```text
estimated_tokens = ceil(UTF-8 byte length / 4)
```

따라서 수치는 절대 토큰 수라기보다, 동일 기준에서 본 입력량 감소율로 해석하는 것이 적절하다.

## Result

| Item | Value |
|---|---:|
| Sample entries | 20 |
| Total sentences | 143 |
| Selected emotion anchors | 18 |
| Selected LLM chunks | 11 |
| Full text characters | 6,062 |
| Filtered text characters | 1,502 |
| Estimated full tokens | 3,595 |
| Estimated filtered tokens | 871 |
| Estimated token reduction rate | 75.77% |

## Presentation Summary

KoELECTRA를 감정 필터로 사용하면 LLM이 전체 일기 원문을 모두 읽지 않고, 감정적으로 의미 있는 문장과 주변 문맥만 분석할 수 있다. 이번 샘플 기준으로 LLM 입력 후보는 전체 143문장에서 11개 청크로 줄었고, 동일한 토큰 추정 기준 입력량은 약 75.77% 감소했다.

발표에서는 다음과 같이 설명할 수 있다.

> KoELECTRA를 필터로 사용해 감정적으로 중요한 문장만 선별한 결과, 20개 샘플 기준 LLM에 전달되는 입력량을 약 76% 줄일 수 있었다. 이를 통해 감정 분석의 안정성을 확보하면서도 LLM API 토큰 비용을 절감할 수 있는 구조를 확인했다.
