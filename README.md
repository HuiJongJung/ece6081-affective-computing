# ECE6081 감성컴퓨팅 프로젝트

**생체신호 융합을 이용한 스트레스 조건 분류와 결과 분석**

WESAD의 손목 EDA(4 Hz)와 BVP(64 Hz)로 스트레스 조건(Stress=1 / Baseline·Amusement=0)을 분류한다.
기준 모델(Logistic Regression)과 Advanced Model을 EDA / BVP / 특징 융합 / 예측 융합 네 조건에서
LOSO로 비교하고, 연구 질문 RQ1~RQ4를 검증한다.

- 제출 마감: 2026-12-13(일) 23:59:59, iClass
- 사전 결정 기록: [docs/decisions.md](docs/decisions.md)

## 디렉터리 구조

```
configs/         실험 공통 설정 (구간, 라벨, 참가자, 기준 모델)
data/            WESAD 원본 및 전처리 산출물 (git 제외)
docs/            사전 결정 기록, 보고서 자료
notebooks/       탐색용 노트북 (최종 결과는 scripts/ 출력에서만 인용)
results/         제출용 CSV (구간 예측, 참가자별 점수, 융합 차이, 조건별 오류표)
scripts/         실행 진입점
src/
  data/          WESAD 로드, 60초 구간 생성, 공통 구간 인덱스
  features/      EDA·BVP 특징 추출
  models/        기준 LR, Advanced Model
  fusion/        특징 융합, 예측 융합, 가중치 선택(RQ3)
  eval/          LOSO, 내부 CV, 지표, RQ 분석
```

## 환경 설정

```bash
conda env create -f environment.yml
conda activate ece6081
```

## 데이터 준비

1. [WESAD 공식 페이지](https://ubi29.informatik.uni-siegen.de/usi/data_wesad.html)에서 `WESAD.zip`(약 2 GB)을 받는다.
2. `data/raw/WESAD/` 아래에 `S2/ ... S17/` 폴더가 오도록 압축을 푼다. (S1, S12는 원본에 없음)
3. 원본 데이터는 커밋하지 않는다.

## 실행

> 구현 후 채울 것. 표·그림마다 생성한 명령을 기록한다.

```bash
# python scripts/make_windows.py      # 공통 구간 인덱스 생성
# python scripts/run_baseline.py      # 기준 모델 4조건 LOSO
# python scripts/run_advanced.py      # Advanced Model 4조건 LOSO
# python scripts/run_rq.py --rq 1     # RQ1~RQ4
```

## 평가 규칙 요약

- 60초 길이, 60초 간격의 겹치지 않는 구간을 기록 시작점부터 만든다. 구간 안의 원래 라벨이 모두 같은 경우만 사용한다.
- 모든 모델과 조건은 같은 구간 인덱스를 평가한다. 특징 계산이 실패해도 구간을 빼지 않고 결측 대체로 처리한다.
- 외부 평가는 LOSO(15명). 하이퍼파라미터와 RQ3 가중치는 외부 학습 참가자 내부의 참가자 단위 K-fold로만 정한다.
- 정규화, 결측 대체, 특징 선택, 클래스 가중치, 임곗값에 평가 참가자 정보를 쓰지 않는다.
- 참가자 ID, 시간, 구간 ID, 정답 라벨은 입력 특징에 넣지 않는다.

## 팀원별 기여

| 이름 | 담당 |
|---|---|
|  |  |
|  |  |
|  |  |

## 외부 코드·도구 사용 범위

> 사용한 라이브러리, 인용 코드, AI 도구 사용 범위를 기록한다.
