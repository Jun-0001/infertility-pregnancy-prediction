# 🏥 난임 환자 임신 성공 여부 예측 AI

> **데이콘 × 오즈코딩스쿨 해커톤 2026.04**  
> 8조 팔로피안 | 신정호 · 이로건 · 진태준

[![ROC-AUC](https://img.shields.io/badge/ROC--AUC-0.7423-brightgreen?style=flat-square)]()
[![Python](https://img.shields.io/badge/Python-3.10+-blue?style=flat-square&logo=python)]()
[![LightGBM](https://img.shields.io/badge/LightGBM-✓-orange?style=flat-square)]()
[![CatBoost](https://img.shields.io/badge/CatBoost-✓-yellow?style=flat-square)]()
[![XGBoost](https://img.shields.io/badge/XGBoost-✓-red?style=flat-square)]()

---

## 📌 프로젝트 개요

난임 환자의 IVF/DI 시술 데이터를 기반으로 **임신 성공 여부를 예측**하는 머신러닝 모델입니다.

- **데이터**: 256,351행 × 69컬럼 (시술 1회 = 1행)
- **타겟**: 임신 성공 여부 (0/1) — 성공 25.83% / 실패 74.17%
- **평가 지표**: ROC-AUC
- **최종 성능**: Public LB **0.7422861508** (상위권)

---

## 🩺 도메인 배경

### 난임이란?
1년 이상 정상적인 부부관계에도 임신이 되지 않는 상태 *(Zegers-Hochschild, 2017)*  
국내 환자 수: 22만 8,618명(2020) → **30만 401명(2024)** 으로 급증

### IVF 배아 7단계 파이프라인

```
1. 배란 유도 → 2. 난자 채취 → 3. 정자 준비 → 4. 수정
     → 5. 배양(Day 3~6) → 6. 이식/동결 → 7. 착상 확인 (타겟)
```

### 핵심 임상 시그널 (EDA 발견)

| 변수 | 임신 성공률 | 신호 강도 |
|------|------------|----------|
| 배아 이식 경과일 Day5 (배반포) | **40.4%** | ★★★★★ |
| 나이 만18-34세 | 32.3% | ★★★★★ |
| 단일 배아 이식 여부 | 36.7% | ★★★★ |
| 반복 주기 (`:` 포함) | 7.8% ↓ | ★★★★ (부정) |
| 이식 없음 (NaN) | 2.4% | ★★★★★ |

---

## 🗂️ 프로젝트 구조

```
📦 fertility-prediction
 ┣ 📄 README.md
 ┣ 📄 fertility_model.py       # 최종 추론 코드
 ┣ 📂 notebooks/
 ┃  ┣ 01_EDA.ipynb
 ┃  ┣ 02_preprocessing.ipynb
 ┃  ┣ 03_feature_engineering.ipynb
 ┃  └ 04_modeling.ipynb
 ┣ 📂 outputs/
 ┃  ┣ oof_predictions.csv
 ┃  └ sample_submission_0p7423.csv
 └ 📂 data/
    ┣ train.csv
    ┗ test.csv
```

---

## ⚙️ 전처리 파이프라인

전처리는 **데이터 누수 방지 원칙**에 따라 Fold 안/밖을 엄격히 분리합니다.

| 단계 | 작업 | Fold 위치 |
|------|------|----------|
| STEP 0 | 스키마 정의 (6그룹 분류) | Fold 밖 |
| STEP 1 | 컬럼 DROP + NaN→Flag 전환 | Fold 밖 |
| STEP 2 | Dtype 변환 (횟수 문자열→정수, is_censored) | Fold 밖 |
| STEP 3 | 도메인 규칙 R1~R6 결측 처리 | Fold 밖 |
| STEP 4 | 정합성 Flag 6종 + 복합 카테고리 분해 | Fold 밖 |
| STEP 5 | 범주형 정규화 + 희귀값 통합 | Fold 밖 |
| TE | Target Encoding (OOF 기반) | **Fold 안** ⚠️ |
| IMP | Imputer / Scaler | **Fold 안** ⚠️ |

### 도메인 규칙 R1~R6

```python
R1: 시술유형 == DI  →  배아/난자/미세주입 18개 컬럼 NaN → 0 (구조적 결측)
R2: 이식된 배아 수 == 0 or NaN  →  배아 이식 경과일 NaN → -1 (sentinel)
R3: 해동된 배아 수 == 0 or NaN  →  배아 해동 경과일 NaN → -1
R4: 해동 난자 수 == 0 or NaN    →  난자 해동 경과일 NaN → -1
R5: 혼합된 난자 수 == 0 or NaN  →  난자 혼합 경과일 NaN → -1
R6: 총 임신 횟수 == 0회          →  임신 시도 경과 연수 NaN → -1
```

> 💡 **핵심**: `배아 이식 경과일 NaN → 0` 처리 시 Day0 당일이식(성공률 25.1%)과 혼동 발생 → **`-1 sentinel`** 처리로 버그 수정

---

## 🔬 피처 엔지니어링

### 논문 기반 핵심 파생 피처

| 파생 피처 | 정의 | 근거 논문 |
|-----------|------|----------|
| `ever_delivered` ⭐ | 총 출산 횟수 > 0 | Templeton 1996 Lancet |
| `is_FET` ⭐ | 해동된 배아 수 > 0 | 임상 프로토콜 |
| `bast_transfer` | 이식 경과일 ≥ 5 (배반포) | EDA 직접 확인 |
| `ICSI_수정발달률` | ICSI 배아 수 / (미세주입 난자 수 + 1) | EDA Part 04 |
| `elderly_donor_egg` | (나이 ≥ 만40세) AND 기증 난자 사용 | Kawwass 2013 |
| `repeated_3plus` | 총 시술 횟수 ≥ 3 | Templeton 1996 |
| `embryo_group` | 배아 수 4범주 (Poor/Normal/High/Hyper) | Bologna·Sunkara 기준 |

### 배아 파이프라인 전환율 피처

```
성숙률        = 미세주입 난자 / (수집 난자 + 1)         → ~82%
ICSI 수정발달률 = ICSI 배아 / (미세주입 난자 + 1)       → ~68% (역U자 패턴)
이식가능률    = (이식 + 저장) / (총 배아 + 1)           → ~47%
배아_생성률   = 총 배아 / (수집 난자 + 1)               → 종합 질 지표
```

### 피처 전략 실험 결과

| 실험 | 피처 구성 | GBM AUC |
|------|----------|---------|
| A (기준) | 원본 3변수 | 0.586 |
| B | 파생만 (적합성 + 수정률) | 0.572 (-0.3%p) |
| C | 파생만 (적합성 1개) | 0.529 (-5.7%p) ⚠️ |
| **D ⭐** | **원본 + 파생 전체** | **0.586 (최고)** |

> 파생변수는 원본의 **대체재가 아닌 보완재** — 원본 유지 + 파생 추가가 Dominant Strategy

---

## 🤖 모델 학습

### 모델 진화 과정

```
베이스라인          5-seed 배깅        6-way 앙상블       AutoGluon
LGBM 5-fold   →   + real_age    →   LGBM+XGB+CB   →   8h 풀런
OOF: 0.73974      OOF: 0.74045      OOF: 0.74079      OOF: 0.74114
LB:  0.74184      LB:  0.74201      LB:  0.74208      (단일 최강)
                                           ↓
                              V14 NM 최적화 4-way 앙상블
                              OOF: 0.74119 / LB: 0.74228 ★
```

### 최종 모델 구성 (V14)

```python
# 4-way Nelder-Mead 가중치 최적화 앙상블
ensemble = {
    'AutoGluon_8h':  weight_ag,    # OOF 단일 최강
    'LightGBM':      weight_lgbm,  # 도메인 피처
    'CatBoost':      weight_cb,    # 유일한 진짜 다양성
    'XGBoost':       weight_xgb,
}
# Rank Normalization → OOF 기반 가중치 탐색
```

### 5-Fold Stratified CV 결과

| Fold | ROC-AUC |
|------|---------|
| 1 | 0.73717 |
| 2 | 0.74184 |
| 3 | 0.73940 |
| 4 | 0.73735 |
| 5 | 0.74042 |
| **OOF 전체** | **0.73921** |

---

## 📊 주요 발견 & 인사이트

### 1. 배아 이식 경과일 — 가장 강력한 단일 시그널
```
Day0: 25.1% / Day1: 18.7% / Day2: 21.2% / Day3: 25.9%
Day4: 34.4% / Day5★: 40.4% / Day6: 30.0%

이식 없음(NaN): 2.4% ← Day5 대비 13배 차이
```

### 2. ICSI 수정률 — 역U자 패턴
```
0%:     0.54%  (전체 실패)
51~70%: 29.15%
71~90%: 32.93% ← 최고
91~100%: 27.93%
```
> 높을수록 좋다는 직관이 깨짐 — 남성요인 × ICSI 2×2 매트릭스 확인 필요

### 3. 총 생성 배아 수 — 역U자 패턴 (freeze-all 전략)
| 그룹 | 전체 성공률 | 이식 후 성공률 |
|------|------------|---------------|
| 0~6개 (저반응) | 22.07% | 26.15% |
| 7~20개 (정상) | 35.03% ⭐ | 39.62% |
| 21개+ (과반응) | 22.64% | **43.74%** ⭐ |

> 과반응 그룹의 낮은 전체 성공률은 의도적 **freeze-all 전략** 때문 (48% 당기 이식 포기)

### 4. Adversarial Validation
```
Train vs Test 분류 AUC = 0.501 → 분포 동일 확인 ✅
```

---

## 🛠️ 실행 방법

```bash
# 의존성 설치
pip install lightgbm xgboost catboost scikit-learn pandas numpy

# 훈련 + 예측
python fertility_model.py \
  --train_path data/train.csv \
  --test_path  data/test.csv \
  --output_dir outputs/
```

---

## 📚 참고 논문

| 논문 | 활용 |
|------|------|
| Templeton et al. (1996) *Lancet* | ever_delivered, repeated_3plus 피처 근거 |
| Barnett-Itzhaki et al. (2020) | prev_birth_rate 피처 근거 |
| Kawwass et al. (2013) | elderly_donor_egg 피처 근거 |
| Bologna Criteria / Sunkara et al. | 배아 수 그룹(Poor/Normal/High/Hyper) 기준 |
| Olivennes et al. (2021) *Human Reprod Update* | AI 임상 적용 성능 벤치마크 |
| Wang et al. (2020) *Human Reprod Update* | IVF 예측 모델 체계적 리뷰 |

---

## 💡 핵심 교훈

> **"복잡한 모델이 아니라, 데이터의 한계를 이해하고 검증 가능한 개선만 남기는 것이 성능을 만든다"**

1. **데이터가 성능을 결정한다** — 좋은 피처를 더 만드는 것보다, 기존 피처를 제대로 쓰는 것이 중요
2. **성능의 핵심은 안정성이다** — 높은 점수보다, 재현 가능한 점수가 더 중요
3. **검증 전략이 결과를 좌우한다** — 최종 판단은 Public LB 기준으로 수행
4. **모델보다 중요한 것은 데이터 이해** — 8종 모델 상관계수 0.99+ → 알고리즘이 달라도 같은 신호 학습

---

## 🏆 최종 결과

| 지표 | 값 |
|------|-----|
| OOF AUC | 0.74119 |
| Public LB AUC | **0.7422861508** |
| 실제 의료 AI 수준 비교 | 전통 임상 데이터 0.60~0.72 대비 **동등 이상** |

> AUC 0.74 → *"실용 가능한 수준"* — 의료 예측 모델로서 의미 있는 성능  
> *(AUC 0.7~0.8: 실용 가능, 0.8~0.9: 높은 성능 — Hosmer et al., 2013)*

---

<div align="center">

**난임 예측은 정답을 맞추는 문제가 아니라,**  
**불확실성을 얼마나 잘 정량화하느냐의 문제였다.**

</div>
