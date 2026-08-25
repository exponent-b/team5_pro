# ✈️ Airline Passenger Satisfaction

### 서비스 평가·운항 정보·승객 특성을 활용한 만족도 이진분류 및 승객 군집 분석

> 129,880명의 항공 승객 데이터를 기반으로 만족 여부를 예측하고, 모델 해석과 탐색적 군집 분석을 통해 항공 서비스 개선 가설을 도출한 팀 프로젝트입니다.

[최종 통합 Notebook](team5_final_pro.ipynb) · [개인 프로젝트 결과](개인프로젝트결과/) · [데이터 출처](https://www.kaggle.com/datasets/teejmahal20/airline-passenger-satisfaction)

---

## 프로젝트 개요

이 프로젝트는 승객 특성, 여행 정보, 운항 지연시간, 서비스 평가를 이용해 `satisfaction`을 예측하는 **이진분류 문제**입니다.

- `0`: `neutral or dissatisfied` — 중립 또는 불만족
- `1`: `satisfied` — 만족
- **분류 목적:** 만족 여부를 안정적으로 예측하고 주요 예측 요인을 파악
- **군집 목적:** 만족도 타깃 없이 승객 경험을 탐색적으로 세분화하여 서비스 개선 가설 제안

분류 모델의 성능뿐 아니라 교차검증 안정성, 과적합 격차, 오분류 유형, 설명 가능성을 함께 평가했습니다.

## 핵심 결과

| 항목 | 최종 결과 |
|---|---:|
| 최종 모델 | LightGBM |
| Test Accuracy | **0.9651** |
| Test Precision | **0.9765** |
| Test Recall | **0.9433** |
| Test F1-score | **0.9596** |
| Test ROC-AUC | **0.9955** |
| 공식 분류 임계값 | **0.50** |
| 탐색적 군집 수 | **4개** |
| k=4 Silhouette Score | **0.1713** |

최종 혼동행렬은 **TN 14,314 / FP 259 / FN 647 / TP 10,756**이었습니다. 공식 test는 모델·파라미터·임계값을 확정한 뒤 마지막에 한 번 평가했습니다.

## 분석 목표

1. 승객의 만족 여부를 얼마나 안정적으로 예측할 수 있는가?
2. 만족도 예측에 주요하게 활용된 서비스와 승객 특성은 무엇인가?
3. 임계값을 조정했을 때 FP와 FN은 어떻게 달라지는가?
4. 승객을 서비스 경험과 운항 특성에 따라 어떤 유형으로 나눌 수 있는가?

## 데이터 소개

| 구분 | 내용 |
|---|---|
| 출처 | [Kaggle Airline Passenger Satisfaction](https://www.kaggle.com/datasets/teejmahal20/airline-passenger-satisfaction) |
| Train | 103,904행 × 25열 |
| Official test | 25,976행 × 25열 |
| 전체 승객 수 | 129,880명 |
| 식별자 제거 후 컬럼 수 | 23개(입력 22개 + 타깃 1개) |
| 모델 입력 | 수치형 18개, 범주형 4개 |
| 타깃 | `satisfaction` |
| 결측 변수 | `Arrival Delay in Minutes` |
| Train 결측률 | 0.2984%(310건) |
| Test 결측률 | 0.3195%(83건) |

| 변수 그룹 | 주요 변수 |
|---|---|
| 승객 특성 | `Gender`, `Age`, `Customer Type` |
| 여행 특성 | `Type of Travel`, `Class`, `Flight Distance` |
| 서비스 평가 | `Online boarding`, `Inflight wifi service`, `Seat comfort` 등 |
| 운항 정보 | `Departure Delay in Minutes`, `Arrival Delay in Minutes` |
| 타깃 | `satisfaction` |

### 데이터 해석 시 주의사항

- `Unnamed: 0`과 `id`는 식별자이므로 모델 입력에서 제외했습니다.
- 서비스 평가의 `0`은 공식 설명만으로 모든 항목에서 의미를 일관되게 확정하기 어렵습니다. 따라서 결측치나 오입력으로 단정해 대체·삭제하지 않고 원자료 값을 유지했으며, 해석 시 이 불확실성을 함께 고려했습니다.
- 지연시간의 `0`은 지연 없음, 만족도 타깃의 `0`은 중립·불만족 클래스를 의미합니다.
- 출발·도착 지연시간의 상관계수는 0.9655로 높아 중요도 해석 시 중복 정보를 고려했습니다.

## 프로젝트 진행 흐름

```mermaid
flowchart LR
    A[데이터 확인] --> B[EDA]
    B --> C[Train / Validation 분할]
    C --> D[Pipeline 전처리]
    D --> E[모델 비교]
    E --> F[교차검증 및 튜닝]
    F --> G[최종 Test 평가]
    G --> H[Feature Importance / SHAP]
    H --> I[K-means 군집 분석]
    I --> J[서비스 개선 가설]
```

## EDA 핵심 인사이트

- 중립·불만족 56.7%, 만족 43.3%로 심각한 클래스 불균형은 아니었습니다.
- 재표본화 없이 **Accuracy와 F1-score를 함께** 핵심 평가 지표로 사용했습니다.
- 만족·불만족 집단의 서비스 평균 차이가 큰 상위 항목은 `Online boarding`, `Inflight entertainment`, `Seat comfort`였습니다.
- 여행 목적, 좌석 등급, 고객 유형에 따라 만족률 차이가 나타났습니다.
- 비행 거리와 지연시간은 오른쪽으로 치우쳐 있으나 실제 운항 상황일 수 있어 임의로 삭제하지 않았습니다.
- 서비스 항목 사이의 높은 상관관계로 인해 개별 중요도가 여러 변수에 나뉘어 나타날 수 있습니다.

> EDA에서 확인한 차이는 **관련성**이며 서비스 변경의 인과효과를 의미하지 않습니다. 또한 서비스 평가 `0`의 의미가 항목별로 다를 가능성이 있으므로 집단 평균을 해석할 때 주의가 필요합니다.

## 데이터 전처리

| 처리 항목 | 적용 방법 | 근거 |
|---|---|---|
| 식별자 | `Unnamed: 0`, `id` 제거 | 예측 대상과 무관한 행 식별 정보 |
| 데이터 분할 | Train/Validation 80:20, `stratify=y` | 클래스 비율 유지 |
| 수치형 결측치 | 중앙값 대체 | 지연시간의 치우친 분포 고려 |
| 범주형 변수 | One-Hot Encoding | 순서가 없는 명목형 범주 |
| 로지스틱 회귀 | StandardScaler 적용 | 변수 단위 차이 보정 |
| 트리 모델 | 표준화 미적용 | 분할 기반 모델은 단위 변화에 덜 민감 |
| 누수 방지 | Pipeline, ColumnTransformer | 각 CV fold의 학습 데이터에서만 전처리 학습 |
| 재현성 | `random_state=10` | 동일 실험 결과 재현 |

서비스 평가 `0`은 결측치로 명시되지 않았고 의미도 확정하기 어려워 최종 모델에서는 원자료 값을 유지했습니다. 향후 데이터 사전을 확보한 뒤 `Not Applicable` 여부를 별도 변수로 분리하고, 원값 유지·결측 처리 방식과의 민감도 분석을 수행할 수 있습니다.

## 모델링 및 검증

모든 모델을 동일한 **Stratified 5-Fold**, `scoring="f1"`, `random_state=10` 조건에서 비교했습니다.

| 모델 | CV 평균 F1 | 표준편차 | Validation Accuracy | Validation F1 | Train F1 | 과적합 격차 |
|---|---:|---:|---:|---:|---:|---:|
| LightGBM | **0.9577** | 0.0017 | 0.9631 | 0.9565 | 0.9614 | 0.0049 |
| Random Forest | 0.9562 | **0.0015** | 0.9629 | 0.9564 | 1.0000 | 0.0436 |
| XGBoost | 0.9540 | 0.0020 | 0.9609 | 0.9540 | 0.9581 | 0.0041 |
| Decision Tree | 0.9360 | 0.0014 | 0.9454 | 0.9372 | 1.0000 | 0.0628 |
| Logistic Regression | 0.8529 | 0.0016 | 0.8758 | 0.8538 | 0.8528 | -0.0010 |
| DummyClassifier | 0.0000 | 0.0000 | 0.5667 | 0.0000 | 0.0000 | 0.0000 |

LightGBM과 Random Forest의 CV 평균 F1 차이는 0.0015로 작았습니다. Random Forest는 학습 F1이 1.0이고 일반화 격차가 큰 반면, LightGBM은 유사한 검증 성능에서 과적합 격차가 작았습니다. 따라서 LightGBM을 압도적으로 우수하다고 표현하기보다 **성능과 일반화 안정성의 균형**을 선정 근거로 삼았습니다.

## 하이퍼파라미터 튜닝

기본 비교에서 성능이 높고 과적합 격차가 상대적으로 안정적인 LightGBM과 XGBoost만 제한된 GridSearchCV로 튜닝했습니다. 각 모델은 8개 조합 × 5개 fold, 총 40회 학습으로 비교했습니다.

| 모델 | CV 평균 F1 | 표준편차 | Validation F1 | Train F1 | 과적합 격차 |
|---|---:|---:|---:|---:|---:|
| Tuned LightGBM | **0.9589** | **0.0019** | **0.9576** | 0.9640 | **0.0063** |
| Tuned XGBoost | 0.9582 | 0.0020 | 0.9573 | 0.9805 | 0.0232 |

최종 LightGBM 파라미터:

```python
{
    "learning_rate": 0.05,
    "max_depth": -1,
    "n_estimators": 150,
    "num_leaves": 63
}
```

## 임계값 분석

임계값은 validation 데이터에서만 비교했습니다.

| 임계값 | FP | FN | Accuracy | Precision | Recall | F1-score | FPR |
|---:|---:|---:|---:|---:|---:|---:|---:|
| **0.50** | 185 | **562** | **0.9641** | 0.9786 | **0.9376** | **0.9576** | 0.0157 |
| 0.60 | 135 | 655 | 0.9620 | 0.9841 | 0.9273 | 0.9548 | 0.0115 |
| 0.70 | 68 | 768 | 0.9598 | 0.9918 | 0.9147 | 0.9517 | 0.0058 |
| 0.75 | **36** | 840 | 0.9578 | **0.9956** | 0.9067 | 0.9491 | **0.0031** |

실제 FP·FN 비용자료가 없으므로 Accuracy와 F1-score의 균형이 좋은 **0.50을 공식 임계값**으로 사용했습니다. 0.75는 실제 불만족 고객을 만족으로 판단하는 FP의 비용이 큰 운영 환경을 가정한 별도 시나리오입니다. 임계값이 높아지면 FP는 줄지만 FN이 늘고 Recall과 F1-score가 낮아지는 상충관계가 있습니다.

## 공식 Test 평가

| Accuracy | Precision | Recall | F1-score | ROC-AUC |
|---:|---:|---:|---:|---:|
| **0.9651** | **0.9765** | **0.9433** | **0.9596** | **0.9955** |

```text
Confusion Matrix
[[TN 14314, FP 259],
 [FN   647, TP 10756]]
```

- **FP:** 실제 중립·불만족 승객을 만족으로 예측한 259건
- **FN:** 실제 만족 승객을 중립·불만족으로 예측한 647건
- 전체 25,976건 중 25,070건을 올바르게 분류했습니다.

선정 단계의 F1은 Train 0.9640, Validation 0.9576이었습니다. 모델 확정 후 전체 train으로 재학습한 모델의 official test F1은 0.9596이었습니다.

## 모델 해석

### Feature Importance와 SHAP의 차이

- **Feature Importance:** 모델이 분할 과정에서 변수를 얼마나 활용했는지 확인
- **SHAP:** 각 변수 값이 만족·불만족 예측 방향에 어떻게 기여했는지 확인

선정 단계 LightGBM과 validation 표본 최대 2,000개를 사용한 평균 절대 SHAP 기준 상위 변수는 다음과 같습니다.

| 순위 | 변수 | 평균 \|SHAP\| |
|---:|---|---:|
| 1 | Type of Travel — Business travel | 1.8092 |
| 2 | Inflight wifi service | 1.6113 |
| 3 | Online boarding | 1.5618 |
| 4 | Customer Type — Loyal Customer | 0.7440 |
| 5 | Class — Business | 0.4903 |

> Feature Importance와 SHAP은 모델 예측에 대한 기여도를 나타내며 실제 인과관계를 의미하지 않습니다.

## 승객 군집 분석

만족도 타깃을 제외한 14개 변수에 중앙값 대체와 StandardScaler를 적용한 뒤 K-means의 k=2~6을 비교했습니다. PCA는 2차원 시각화에만 사용했으며, K-means는 표준화된 원래 14차원 공간에서 학습했습니다.

| k | Inertia | Silhouette | Calinski-Harabasz | Davies-Bouldin | 최소 군집 비율 |
|---:|---:|---:|---:|---:|---:|
| 2 | 1,199,911.9 | **0.1756** | **22,058.7** | 2.0502 | 0.4737 |
| 3 | 1,091,198.7 | 0.1621 | 17,303.8 | 2.0937 | 0.2801 |
| **4** | **978,970.3** | **0.1713** | **16,828.5** | **1.7883** | **0.0363** |
| 5 | 923,218.0 | 0.1452 | 14,952.1 | 1.9128 | 0.0347 |
| 6 | 882,843.1 | 0.1370 | 13,458.9 | 1.9599 | 0.0338 |

수치적 분리만 보면 k=2가 유리하지만, 서비스 전략을 세분화하기 위한 해석 가능성을 함께 고려해 k=4를 탐색적으로 선택했습니다. k=4의 실루엣 점수가 약 0.17이므로 군집을 명확한 자연 집단으로 일반화하지 않습니다.

| 군집 | 승객 수 | 만족도 비율 | 탐색적 유형 |
|---:|---:|---:|---|
| 0 | 27,581 | 17.37% | 객실환경 저평가형 |
| 1 | 43,604 | 70.92% | 전반적 서비스 고평가형 |
| 2 | 3,774 | 35.24% | 지연 경험 집중형 |
| 3 | 28,945 | 27.57% | 운영서비스 개선형 |

군집 번호 자체에는 의미가 없으며 초기값이나 데이터가 달라지면 번호가 바뀔 수 있습니다.

## 서비스 관점의 인사이트

- 기내 Wi-Fi와 온라인 탑승 경험은 만족 예측에서 우선적으로 점검할 가치가 있습니다.
- 객실환경 저평가형에는 좌석 편안함, 청결, 기내 엔터테인먼트 개선을 검토할 수 있습니다.
- 운영서비스 개선형에는 탑승 서비스, 수하물 처리, 기내 서비스의 접점 품질을 우선 점검할 수 있습니다.
- 지연 경험 집중형에는 지연 자체뿐 아니라 사전 안내, 대체편, 보상 절차를 함께 검토할 수 있습니다.

이 제안은 모델과 군집 분석에서 도출한 **탐색적 가설**입니다. 실제 정책 적용 전 A/B 테스트 또는 후속 기간 데이터 검증이 필요합니다.

## 프로젝트 구조

현재 GitHub 저장소의 파일 구조입니다.

```text
team5_pro/
├── README.md
├── team5_final_pro.ipynb                 # 실행 결과가 저장된 최종 통합 분석본
└── 개인프로젝트결과/
    ├── 항공만족_배근우.ipynb
    ├── 항공만족_배지수.ipynb
    └── 항공만족_김양일/
        ├── 01_EDA.ipynb
        ├── 02_전처리_모델링.ipynb
        ├── 03_클러스터링.ipynb
        ├── train.csv
        └── test.csv
```

- `team5_final_pro.ipynb`: 세 개인 프로젝트의 장점을 선별해 구성한 최종 통합본
- `개인프로젝트결과/`: 팀원별 원본 분석 Notebook과 김양일 분석에 사용한 CSV

> Kaggle 페이지의 라이선스 표시는 `Other (specified in description)`입니다. 원본 CSV를 공개 저장소에 재배포하기 전 데이터 페이지의 세부 이용 조건을 확인하세요. 데이터 재배포가 불확실하다면 CSV 대신 다운로드 안내만 제공하는 방식을 권장합니다.

## 실행 방법

1. 저장소를 내려받습니다.

```bash
git clone https://github.com/exponent-b/team5_pro.git
cd team5_pro
```

2. 다음 라이브러리를 설치하고 Jupyter Notebook을 실행합니다.

```bash
pip install pandas numpy scikit-learn xgboost lightgbm shap matplotlib seaborn jupyter
jupyter notebook
```

3. [최종 통합 Notebook](team5_final_pro.ipynb)을 엽니다.

4. GitHub에서 다시 실행할 때는 데이터 경로 설정 셀을 아래처럼 지정한 뒤 위에서 아래로 실행합니다.

```python
from pathlib import Path

DATA_DIR = Path("개인프로젝트결과") / "항공만족_김양일"
TRAIN_PATH = DATA_DIR / "train.csv"
TEST_PATH = DATA_DIR / "test.csv"
```

> Notebook에 저장된 출력은 최종 실행 결과입니다. 다른 위치에서 실행할 경우 위 상대경로를 사용하면 저장소 안의 CSV를 읽을 수 있습니다.

Notebook에 기록된 실행 환경:

| 도구 | 버전 |
|---|---:|
| Python | 3.14.6 |
| pandas | 3.0.3 |
| NumPy | 2.5.1 |
| scikit-learn | 1.9.0 |
| XGBoost | 3.4.0 |
| LightGBM | 4.7.0 |
| SHAP | 0.52.0 |
| Matplotlib | 3.11.1 |
| Seaborn | 0.13.2 |

## 재현성

- 모든 주요 실험의 `random_state=10`
- Stratified 5-Fold 교차검증
- Pipeline과 ColumnTransformer를 통한 fold 내부 전처리
- validation 기반 모델·파라미터·임계값 결정
- 공식 test는 최종 결정 후 평가
- SHAP 분석 표본은 최대 2,000개로 제한
- Notebook 자동 품질 검사 전 항목 통과

## 팀 소개

| 팀원 | 주요 발표 및 정리 역할 |
|---|---|
| 배지수 | 문제 정의, 데이터 구조, EDA, 발표자료 통합 |
| 배근우 | 데이터 전처리, 모델 비교, 교차검증 및 튜닝 |
| 김양일 | 최종 평가, 임계값, SHAP, 군집 분석 및 결론 |

모든 팀원이 데이터 확인부터 결과 해석까지 전체 분석 과정을 함께 검토했으며, 위 역할은 주요 정리 및 발표 범위를 나타냅니다.

## 과제 평가 기준 체크리스트

- [x] 문제 정의와 타깃 클래스 명시
- [x] 데이터 출처 명시
- [x] EDA 및 시각화
- [x] 결측치와 이상치 처리 근거
- [x] 데이터 누수 방지
- [x] 2개 이상의 분류 모델 비교
- [x] Stratified 교차검증
- [x] 하이퍼파라미터 튜닝
- [x] Accuracy와 F1-score 평가
- [x] Confusion Matrix와 오분류 분석
- [x] Feature Importance와 SHAP
- [x] K-means 군집 분석
- [x] 한계와 향후 개선 방향
- [x] `random_state` 고정 및 재현성 확인

## 한계 및 향후 개선

- 설문 점수만으로 구체적인 불만 사유와 실제 인과효과를 확인할 수 없습니다.
- 항공사, 노선, 항공권 가격, 예약 시점, 운항 상황 등의 추가 변수가 필요합니다.
- 서비스 평가 0을 별도 `Not Applicable` 지표로 분리하는 처리 방법을 추가 비교할 수 있습니다.
- 실제 FP·FN 비용자료가 없어 운영 임계값은 실제 고객 이탈 및 대응 비용을 반영해 다시 결정해야 합니다.
- k=4의 최소 군집 비율이 작고 실루엣 점수도 약 0.17이므로 군집 결과는 탐색적으로 해석해야 합니다.
- 향후 다른 변수 구성, 기간 외 검증, 다른 군집 알고리즘과의 비교가 필요합니다.

## 참고자료 및 라이선스

- Dataset: [Kaggle — Airline Passenger Satisfaction](https://www.kaggle.com/datasets/teejmahal20/airline-passenger-satisfaction)
- 데이터 제공자: TJ Klein (`teejmahal20`)
- Kaggle 표시 라이선스: `Other (specified in description)`
- [scikit-learn documentation](https://scikit-learn.org/stable/)
- [XGBoost documentation](https://xgboost.readthedocs.io/)
- [LightGBM documentation](https://lightgbm.readthedocs.io/)
- [SHAP documentation](https://shap.readthedocs.io/)

프로젝트 코드에 별도 라이선스가 명시되어 있지 않으므로 임의의 오픈소스 라이선스를 선언하지 않습니다.

---

## English Summary

This project predicts airline passenger satisfaction using passenger profiles, flight information, delay records, and service ratings. Logistic Regression, Decision Tree, Random Forest, XGBoost, and LightGBM were compared under stratified cross-validation. LightGBM was selected based on predictive performance and generalization stability, achieving a test F1-score of **0.9596** and ROC-AUC of **0.9955**. Feature Importance and SHAP were used for model interpretation, while K-means clustering provided exploratory passenger segments for service-improvement hypotheses.
