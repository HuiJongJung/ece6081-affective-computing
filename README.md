# ECE6081 감성컴퓨팅 프로젝트

**생체신호 융합을 이용한 스트레스 조건 분류와 결과 분석**

WESAD의 손목 EDA(4 Hz)와 BVP(64 Hz)로 스트레스 조건(Stress = 1 / Baseline · Amusement = 0)을 분류한다.
세 모델을 EDA / BVP / 특징 융합 / 예측 융합 네 조건에서 LOSO 15-fold로 비교하고, RQ1~RQ4를 검증한다.

| 모델 | 구성 |
|---|---|
| M0 | 수작업 특징 + Logistic Regression (C = 1, balanced, max_iter = 2000 고정) |
| M1 | 1D-ResNet 인코더 + Linear, 라벨로 함께 학습 |
| M2 | 교차 모달 JEPA로 인코더 사전학습(라벨 미사용) → 인코더 동결 + Linear (제안 모델) |

- 제출 마감: 2026-12-13(일) 23:59:59, iClass
- 실험 계획: [docs/experiment_plan.pptx](docs/experiment_plan.pptx) · 결정 기록: [docs/decisions.md](docs/decisions.md)

## 디렉터리 구조

```
configs/common.yaml       데이터·구간·평가·M0 설정 (모든 실험 공통)
configs/experiments.yaml  M1·M2 학습 설정, RQ1~RQ4 설정 (TODO = 미정 값)
scripts/                  실행 진입점 (아래 "실행")
src/data/wesad.py         WESAD 로드, 60초 구간 생성
src/features/handcrafted.py  M0 EDA·BVP 특징
src/models/               M0(baseline.py), M1(deep.py), M2(jepa.py), ResNet 인코더(resnet.py)
src/eval/                 LOSO(loso.py), 지표·오류표(metrics.py), RQ2 잡음(noise.py)
src/fusion/weights.py     RQ3 가중치 선택
data/                     WESAD 원본·전처리 (git 제외)
outputs/models/           fold별 학습된 모델 (git 제외, RQ2·RQ3가 재사용)
results/                  제출용 CSV
```

## 환경 설정

```bash
conda env create -f environment.yml
conda activate ece6081
```

M2 사전학습은 CPU에서 epoch당 약 10초(fold당 60 epoch)라 GPU를 권장한다.

## 데이터 준비

1. [WESAD 공식 페이지](https://ubi29.informatik.uni-siegen.de/usi/data_wesad.html)에서 `WESAD.zip`(약 2 GB)을 받는다.
2. `data/raw/WESAD/S2/S2.pkl … S17/S17.pkl`이 되도록 압축을 푼다. (S1, S12는 원본에 없음)

## 실행

순서대로 실행한다. 모든 스크립트는 `--set key=value`로 설정을 덮어쓸 수 있다 (예: `--set m2.pretrain.epochs=30`).

```bash
python scripts/make_windows.py                    # 공통 60초 구간 (모든 모델·RQ가 같은 구간 사용)

python scripts/run_common.py --model m0           # 공통 실험: 모델별 4조건 × LOSO 15-fold
python scripts/run_common.py --model m1
python scripts/run_common.py --model m2
python scripts/make_tables.py                     # 결과표 ① 공통 실험, ② RQ1

python scripts/run_rq2.py                         # 결과표 ③ RQ2 (저장된 M2 fold 모델에 잡음 BVP, 재학습 없음)
python scripts/run_rq3.py                         # 결과표 ④ RQ3 (학습 참가자 내부 3-fold로 w* 선택)
python scripts/run_common.py --model m2 --seed 43 # RQ4 두 번째 구성원 (rq4.bootstrap: true면 --bootstrap 추가)
python scripts/run_rq4.py                         # 결과표 ⑤ RQ4
```

| 계획서 결과표 | 파일 |
|---|---|
| ① 공통 실험 | `results/tables/table1_common.csv` |
| ② RQ1 | `results/tables/table2_rq1.csv` |
| ③ RQ2 | `results/tables/table3_rq2.csv` |
| ④ RQ3 | `results/tables/table4_rq3.csv` |
| ⑤ RQ4 | `results/tables/table5_rq4.csv` |

안내서가 요구하는 CSV:

| 요구 사항 | 파일 |
|---|---|
| 모델별 구간 예측 | `results/predictions/<model>_seed<seed>.csv` (RQ별: `rq2_*`, `rq3_*`, `rq4_members_*`) |
| 참가자별 점수 | `results/scores/scores_per_subject.csv` |
| 융합 성능 차이 (EDA 대비, BVP 대비) | `results/scores/fusion_diff.csv` |
| 조건별 오류표 (n / N) | `results/scores/error_table.csv`, `error_table_per_subject.csv` |
| 참가자 분할 | `results/splits/<run>.csv`, `results/splits/rq3_inner_<run>.csv` |
| 제외 구간 수와 이유 | `results/windows/window_summary.csv`, `window_log.csv` |

### 합성 데이터로 실행 확인

WESAD 없이 전체 흐름이 돌아가는지만 확인한다. **이 결과는 보고서에 쓰지 않는다.**

```bash
python scripts/make_synthetic_wesad.py
SYN="--raw-dir data/synthetic/raw/WESAD --processed-dir data/synthetic/processed --results-dir results/synthetic --models-dir outputs/synthetic/models"
FAST="m1.epochs=2 m2.pretrain.epochs=1 m2.pretrain.crops_per_subject=16 m2.head.epochs=10"
python scripts/make_windows.py $SYN
python scripts/run_common.py --model m0 $SYN
python scripts/run_common.py --model m2 $SYN --set $FAST
python scripts/make_tables.py $SYN
```

## 평가 규칙 요약

- 60초 길이, 60초 간격의 겹치지 않는 구간을 기록 시작점부터 만든다. 구간 안의 원래 라벨이 모두 같은 경우만 사용한다.
- 모든 모델과 조건은 같은 구간을 평가한다. 특징 계산이 실패해도 구간을 빼지 않고 학습 자료 중앙값으로 대체한다.
- 외부 평가는 LOSO(15명). RQ3 가중치는 외부 학습 참가자 내부의 참가자 단위 3-fold로만 정한다.
- 정규화, 결측 대체, 클래스 가중치, 임곗값(0.5 고정)에 평가 참가자 정보를 쓰지 않는다. M2 사전학습도 학습 참가자의 기록만 쓴다.
- 참가자 ID, 시간, 구간 ID, 정답 라벨은 입력 특징에 넣지 않는다.

## 팀원별 기여

| 이름 | 담당 |
|---|---|
|  |  |
|  |  |
|  |  |

## 외부 코드·도구 사용 범위

> 사용한 라이브러리, 인용 코드, AI 도구 사용 범위를 기록한다.
