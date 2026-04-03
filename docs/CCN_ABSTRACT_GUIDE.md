# CCN 2026 Extended Abstract 작성 가이드

## 목적

이 문서는 CCN 2026 Extended Abstract 작성에 필요한 모든 합의사항, 실험 구조, 결과 해석, 스토리 범위를 정리한다. 다른 채팅이나 세션에서도 이 문서만 읽으면 바로 abstract 작성을 이어갈 수 있도록 한다.

---

## 1. 제출 형식

- **학회:** CCN (Cognitive Computational Neuroscience) 2026
- **트랙:** Extended Abstract
- **Abstract:** 최대 300단어 (web form과 동일)
- **본문:** 최대 2페이지 (references 제외)
- **리뷰:** Double-blind
- **템플릿:** `ccn/ccn-template-main 2/ccn_extended_abstract.tex`
- **Storypack:** `ccn/ccn_extended_abstract_storypack_20260328/` (figure/table 후보, 분석 리포트, quicklook 파일 정리됨)

---

## 2. 핵심 질문 (Research Question)

> 인간은 하나의 쌍이 두 가지 관계에 동시에 참여하는 mixed-relation triangle에서 triangle inequality를 위반한다.
> LLM도 sequential in-context learning만으로 동일한 위반을 보일 수 있는가?
> 그리고 그 위반의 표상적 기반은 무엇인가?

---

## 3. 배경: Triangle Inequality Violation이란

### 3.1 기본 개념

Triangle inequality는 metric 공간의 공리 중 하나다:

```
d(A,B) + d(B,D) ≥ d(A,D)
```

Parallelogram model을 비롯한 vector-space account는 개체를 공간의 점으로, 관계를 차이벡터로 표현한다. 이 설명이 고정된 metric 공간 위에서 성립한다면, relational similarity도 triangle inequality를 따라야 한다.

### 3.2 인간 실험 (Selin Samra Thesis)

- **참가자:** 30명
- **자극:** 18개 문항, 각 문항에 4개 **쌍(pair)** (A, B, C, D). 각 쌍은 두 개의 항목으로 구성 (예: Q1의 A = dog→puppy)
- **자극 형태:** 언어적 편향을 최소화하기 위해 단어 대신 **clipart 이미지**로 제시
- **제시 방식:** 매 trial마다 **두 쌍을 나란히 제시**하고, 두 쌍의 **관계가 얼마나 유사한지** 평가. 양방향 (AB, BA) 모두 측정하여 순서 효과 통제
- **측정:** 7점 Likert 척도 (1 = 전혀 유사하지 않음, 7 = 매우 유사함)
- **순차적 문맥 누적 없음** — 각 판단이 독립적. 한 trial의 결과가 다음 trial에 영향을 주지 않음
- **Trial 구조 예시 (Q1, 참가자 sunsik):**
  ```
  Trial:1_1  a,b → 6점  (A와 B의 관계 유사도)
  Trial:1_2  a,c → 7점  (A와 C의 관계 유사도)
  Trial:1_3  b,c → 5점
  Trial:1_4  a,d → 7점
  Trial:1_5  b,d → 2점  ← B-D 관계 없으므로 낮음
  Trial:1_6  c,d → 2점
  Trial:1_7  b,a → 6점  (역방향 — 순서 효과 통제)
  Trial:1_8  d,a → 7점
  ...
  ```
- **데이터 위치:** `datasets/csv_data/` (30개 CSV 파일, 참가자별 1개)

### 3.3 쌍의 관계 구조 (TRIANGLE_RELATION_ROLE_MAP)

A, B, C, D는 각각 **두 항목으로 이루어진 쌍(pair)**이다. 각 쌍 내부에 하나의 관계가 존재한다.

- **쌍 A:** 두 가지 관계를 동시에 가짐 (관계1 + 관계2). 이것이 실험의 핵심 — A가 ambiguous함
- **쌍 B:** A와 관계1만 공유
- **쌍 C:** A, B와 관계1 공유 (consistent control)
- **쌍 D:** A와 관계2만 공유
- **B-D:** 공유하는 관계 없음

예시 (Q1):
```
쌍 A: dog → puppy     관계: parent-descendant / produce (두 가지!)
쌍 B: cow → calf      관계: parent-descendant (관계1만)
쌍 C: cat → kitten    관계: parent-descendant (관계1, control)
쌍 D: cow → milk      관계: produce (관계2만)
B-D: calf과 milk 사이에 공유 관계 없음
```

참가자가 "A와 B의 관계가 유사한가?" → 높음 (관계1 공유)
참가자가 "A와 D의 관계가 유사한가?" → 높음 (관계2 공유)
참가자가 "B와 D의 관계가 유사한가?" → 낮음 (공유 관계 없음)

관계 맵 전체: `docs/TRIANGLE_RELATION_ROLE_MAP.md`

### 3.4 두 가지 Triangle

| | ABC (Consistent) | ABD (Mixed) |
|---|---|---|
| 구성 | A-B, A-C, B-C | A-B, A-D, B-D |
| 관계 | 세 edge 모두 **같은 관계** 공유 | A-B는 관계1, A-D는 관계2, B-D는 **관계 없음** |
| 예측 | 삼각형 성립 → violation 없음 | A가 양쪽에 높고 B-D가 낮음 → violation |
| 역할 | **Control** | **Test** |

### 3.5 Product Test 공식

인간과 LLM 모두 동일한 공식 사용:

```python
PT(x, y, z) = (x * y * z) / min(x, y, z)²
```

- x, y, z = 세 edge의 normalized similarity 값 [0, 1]
- **PT > 1 → triangle inequality VIOLATION (위반)**
- **PT ≤ 1 → violation 없음 (기하학적 제약 유지)**

인간: score를 `(score - 1) / 6`으로 [0, 1] 정규화
LLM: target logit probability를 percentile 정규화 `(logprob - p5) / (p95 - p5)` clipped to [0, 1]

코드: `scripts/compute_product_test_bootstrap_humans.py`, `scripts/compute_product_test_bootstrap_unified.py`

---

## 4. 왜 Violation이 생기는가 — 직관적 설명

### ABC (Control) — violation 안 생김:
```
A ──관계1──▶ B
A ──관계1──▶ C
B ──관계1──▶ C
→ 세 쌍 모두 유사도 높음 → 삼각형 성립 → PT ≈ 1
```

### ABD (Test) — violation 생김:
```
A ──관계1──▶ B  (높음: A가 관계1을 가지고 있으니까)
A ──관계2──▶ D  (높음: A가 관계2도 가지고 있으니까)
B ──???──▶ D   (낮음: B와 D는 관계가 없으니까)
→ AB 높고, AD 높고, BD 낮음 → PT >> 1 → VIOLATION
```

핵심: **A가 두 가지 다른 관계를 동시에 가지고 있기 때문에**, B에게도 D에게도 높은 유사도를 보이지만, B와 D 사이에는 아무 관계가 없어서 삼각형이 무너진다.

### Product Test 공식의 수학적 직관

```python
PT(x, y, z) = (x * y * z) / min(x, y, z)²
```

구체적 숫자 예시 (Q1 인간 데이터):
```
sim(AB) = 0.83  (A-B 관계 유사, 높음)
sim(AD) = 1.00  (A-D 관계 유사, 높음)
sim(BD) = 0.17  (B-D 관계 없음, 낮음)

min = 0.17 (BD)
PT = (0.83 × 1.00 × 0.17) / 0.17² = 0.1411 / 0.0289 = 4.88 >> 1 → VIOLATION
```

반대로 ABC (consistent):
```
sim(AB) = 0.83  (같은 관계)
sim(AC) = 1.00  (같은 관계)
sim(BC) = 0.67  (같은 관계)

min = 0.67 (BC)
PT = (0.83 × 1.00 × 0.67) / 0.67² = 0.5561 / 0.4489 = 1.24 ≈ 1 → violation 약함
```

직관: **세 edge 중 하나만 유독 낮으면** (즉 min이 매우 작으면) 분모가 매우 작아져서 PT가 크게 올라감 → violation. ABD에서는 BD가 그 역할.

---

## 5. 인간 결과

`results/pt_analysis/pt_human_bootstrap_summary.csv` (30명, bootstrap 10000회)

| Q | ABC mean | ABD mean | ABD p>1 | delta (ABD-ABC) |
|---|---|---|---|---|
| Q1 | 0.91 | **1.50** | 1.00 | +0.60 |
| Q4 | 0.78 | **2.29** | 1.00 | +1.51 |
| Q8 | 0.86 | **3.19** | 1.00 | +2.34 |
| Q10 | 0.97 | **2.37** | 1.00 | +1.40 |
| Q11 | 0.73 | **1.33** | 0.95 | +0.60 |
| Q15 | 0.69 | **1.58** | 0.99 | +0.89 |
| Q16 | 0.91 | **1.32** | 1.00 | +0.41 |

Violation이 약하거나 없는 문항도 있음:

| Q | ABC mean | ABD mean | ABD p>1 | 비고 |
|---|---|---|---|---|
| Q5 | 0.62 | 0.65 | 0.00 | ABD violation 없음 |
| Q9 | 0.89 | 0.88 | 0.10 | ABD violation 없음 |
| Q17 | 0.73 | 0.79 | 0.01 | ABD violation 없음 |
| Q18 | 0.98 | 0.81 | 0.09 | ABD violation 없음 |

패턴:
- **ABC:** 거의 모든 문항에서 PT < 1 (violation 없음)
- **ABD:** 18문항 중 다수에서 PT > 1 (violation), 일부 문항은 violation 없음
- **결론:** 인간은 mixed-relation triangle에서 체계적으로 violation을 보이지만, 모든 문항에서 나타나는 것은 아님. 문항별 차이는 관계 구조의 명확도에 따름

---

## 6. LLM 실험 설계

### 6.1 모델
- **Llama-3.1-70B** (4bit quantization, bf16)
- Function vector 추출 가능한 ICL 세팅

### 6.2 PT 측정 — Pure (Base) Condition

LLM에서는 5개 edge (AB, AC, AD, BC, BD)를 모두 측정하여 ABC triangle과 ABD triangle 모두 product test를 수행한다. 인간 실험과 동일한 구조.

LLM에서 각 edge를 측정하는 방법:

```
AB edge: [A예시1] [A예시2] ... [A예시N] B의 source → ?
         → B의 target에 대한 logit 측정
         예: "dog→puppy, cat→kitten, ... , cow→?" → "calf" logit

AC edge: [A예시1] [A예시2] ... [A예시N] C의 source → ?
         → C의 target에 대한 logit 측정
         예: "dog→puppy, cat→kitten, ... , cat→?" → "kitten" logit

AD edge: [A예시1] [A예시2] ... [A예시N] D의 source → ?
         → D의 target에 대한 logit 측정
         예: "dog→puppy, cat→kitten, ... , cow→?" → "milk" logit

BC edge: [B예시1] [B예시2] ... [B예시N] C의 source → ?
         → C의 target에 대한 logit 측정
         예: "cow→calf, horse→foal, ... , cat→?" → "kitten" logit

BD edge: [B예시1] [B예시2] ... [B예시N] D의 source → ?
         → D의 target에 대한 logit 측정
         예: "cow→calf, horse→foal, ... , cow→?" → "milk" logit
```

- Shot 수: 1, 3, 5, 7, 9
- Bootstrap 10000회
- Product test 공식은 인간과 동일
- Similarity 정규화: `target_s_norm = (logprob - p5) / (p95 - p5)` clipped to [0, 1]

**왜 이렇게 측정하나:**
- A 예시를 반복하면 모델이 A의 관계들을 학습함
- B를 query하면 관계1로 해석할 수 있는지 확인 (logit 올라가면 그 관계를 파악한 것)
- D를 query하면 관계2로도 해석할 수 있는지 확인 (logit 올라가면 그 관계도 파악한 것)
- B 예시 후 D를 query하면 B-D 관계 확인 (관계 없으니 logit 안 올라감)

**인간과의 구조적 대응:**
- 인간: 두 쌍의 관계를 직접 비교 → 유사도 rating (1-7)
- LLM: ICL로 관계를 학습시킨 후 target prediction → logit probability
- 측정 방식은 다르지만, 동일한 product test 공식으로 triangle inequality violation 여부를 판정

### 6.3 PCA 측정 — Mixed Context Conditions

representational level에서 A가 문맥에 따라 어떻게 이동하는지 확인:

```
AAA:    A 예시만 반복 (pure A)
BBB:    B 예시만 반복 (pure B)
DDD:    D 예시만 반복 (pure D)
BABA:   B와 A 교대 제시 (mixed AB context)
DADA:   D와 A 교대 제시 (mixed AD context)
```

- Function vector를 통해 각 condition의 relational representation 추출
- Common PCA 공간에 투영하여 조건 간 위치 비교
- Trial 데이터: `results_fv/relation_condition_qwise/.../Q1/_trials/`에 각 condition JSON 존재

---

## 7. LLM 결과

### 7.1 PT (Behavioral) — 5-edge 측정, ABC & ABD 모두 평가

데이터 소스:
- **5-edge run:** `/scratch/sunsik/my_fv_project/pt_analysis/llama31_70b_20260310_104803/pt_bootstrap_summary.csv` (ABC + ABD 모두 포함)
- **unified recompute run:** `/scratch/sunsik/my_fv_project/pt_analysis/llama31_70b_pt_candidate_recompute_full_q8grain_20260325_033335/pt_unified_bootstrap_summary.csv` (ABD만 포함)

CSV 컬럼 설명 (5-edge run):
- `pt_abc_mean`, `pt_abc_p_gt1`: ABC (consistent) triangle product test 값과 violation 확률
- `pt_abd_mean`, `pt_abd_p_gt1`: ABD (mixed) triangle product test 값과 violation 확률
- `delta_mean`: ABD - ABC 차이

#### LLM ABC vs ABD 비교 (10-shot)

| Q | ABC mean | ABC p>1 | ABD mean | ABD p>1 | 패턴 |
|---|---|---|---|---|---|
| Q1 | 1.00 | 0.55 | **3.33** | 1.00 | ABD >> ABC |
| Q3 | 1.00 | 0.56 | **8.97** | 1.00 | ABD >> ABC |
| Q4 | 1.60 | 1.00 | **1.81** | 1.00 | ABD > ABC |
| Q5 | 1.11 | 0.99 | 1.10 | 0.97 | 비슷 |
| Q6 | 1.48 | 1.00 | **21.16** | 1.00 | ABD >> ABC |
| Q7 | 1.27 | 1.00 | **1.77** | 1.00 | ABD > ABC |
| Q8 | **4.97** | 1.00 | 0.79 | 0.02 | ABC >> ABD (역전) |
| Q9 | **4.49** | 1.00 | 2.61 | 1.00 | ABC > ABD (역전) |
| Q10 | 1.78 | 1.00 | **2.71** | 1.00 | ABD > ABC |
| Q11 | 1.55 | 1.00 | **2.11** | 1.00 | ABD > ABC |
| Q13 | 1.08 | 1.00 | **15.45** | 1.00 | ABD >> ABC |
| Q16 | 1.23 | 1.00 | 1.04 | 0.90 | ABC > ABD |
| Q17 | 1.01 | 0.99 | 1.06 | 1.00 | 비슷 |
| Q18 | 1.04 | 1.00 | **5.33** | 1.00 | ABD >> ABC |

**핵심 관찰:**
- **다수 문항 (Q1, Q3, Q6, Q10, Q11, Q13, Q18):** ABD >> ABC — 인간과 동일한 패턴. Mixed triangle에서만 강한 violation
- **일부 문항 (Q8, Q9):** ABC > ABD로 역전. 이 문항에서는 consistent triangle에서도 LLM이 큰 violation을 보임
- **일부 문항 (Q5, Q17):** 양쪽 모두 약한 violation, 차이 미미
- LLM의 ABC violation이 인간보다 전반적으로 높은 편 (인간 ABC는 대부분 < 1)

**인간과의 비교:**
- 인간: ABC < 1 (violation 없음) vs ABD > 1 (violation) — 깔끔한 대비
- LLM: ABC도 일부 > 1이지만, 대다수 문항에서 ABD가 ABC보다 확연히 크다 → **ABD-ABC 차이 방향은 인간과 일치**

### 7.1.1 Shot 수에 따른 ABD 변화 (Q1 예시)

| shot | base AB | base AD | base BD | base ABD | p>1 |
|---|---|---|---|---|---|
| 1 | 0.84 | 0.91 | 0.47 | **1.63** | 1.00 |
| 3 | 0.93 | 0.87 | 0.15 | **5.73** | 1.00 |
| 5 | 0.96 | 0.88 | 0.16 | **5.40** | 1.00 |
| 7 | 0.97 | 0.87 | 0.20 | **4.25** | 1.00 |
| 9 | 0.98 | 0.88 | 0.26 | **3.31** | 1.00 |

관찰:
- **AB, AD는 shot이 늘수록 1에 가까워짐** — 모델이 관계를 더 잘 파악
- **BD는 계속 낮음** (0.15~0.47) — B와 D 사이에 관계가 없으므로
- **ABD는 모든 shot에서 > 1** — 일관되게 violation
- Shot 1에서도 이미 violation이 시작되지만, shot 3-5에서 가장 강해지고 이후 약간 수렴

### 7.2 PCA (Representational) — Mixed Context

`/scratch/sunsik/my_fv_project/results_fv/relation_condition_qwise/.../Q1/_pca_common/`

**AAA_ref PCA (Q1):**
```
AAA ↔ BBB:  4.52  (먼 거리 — 서로 다른 관계)
AAA ↔ BABA: 2.61  (중간 — A가 B 쪽으로 이동)
BABA ↔ BBB: 2.42  (중간 — 아직 BBB는 아님)
```

해석:
- **AAA ↔ BBB = 4.52:** pure A와 pure B는 멀리 떨어져 있음. 서로 다른 관계를 표상
- **AAA ↔ BABA = 2.61:** BABA context에서 representation이 AAA에서 BBB 방향으로 이동. pure A에 머무르지 않음
- **BABA ↔ BBB = 2.42:** 하지만 완전히 BBB가 되지도 않음. **중간 상태**

이것이 의미하는 것: **A의 representation은 고정된 점이 아니다.** AB 문맥이 주어지면 B 쪽으로 이동하고, AD 문맥이 주어지면 D 쪽으로 이동한다 (DADA condition이 존재하지만 아직 AAA_ref PCA에 포함되지 않음 — 추가 분석 가능).

이것이 PT violation과 연결되는 논리:
1. A 예시를 반복하면 모델 내부에서 A의 dual-relation representation이 활성화됨
2. 이 representation은 B 방향으로도, D 방향으로도 이동할 수 있는 잠재력을 가짐
3. B를 query하면 관계1 방향의 해석이 활성화 → logit 높음
4. D를 query하면 관계2 방향의 해석이 활성화 → logit 높음
5. 그러나 B와 D 사이에는 이런 이동 경로가 없음 → logit 낮음
6. 결과: AB 높고, AD 높고, BD 낮음 → PT >> 1 → violation

**BD_ref PCA (Q1) — abstract 범위 밖이지만 참고:**
```
BDBDBD_D ↔ DBDBDB_B: 0.53  (매우 가까움 — 두 mixed 조건이 수렴)
BBB ↔ DDD:            4.80  (매우 멀음 — pure endpoint는 분리)
BBB ↔ BDBDBD_D:       3.26
BBB ↔ DBDBDB_B:       3.79
BDBDBD_D ↔ DDD:       2.75
DBDBDB_B ↔ DDD:       2.57
```

해석: mixed BD context의 두 조건은 pure endpoint(BBB, DDD)와 달리 서로 매우 가까이 위치. 이는 모델이 B-D 사이에서도 어떤 공통 상태를 형성할 수 있음을 시사하지만, 해석은 아직 탐색적 수준. 추후 discussion에서 "모델이 latent shared structure를 추론할 수 있는가?"로 확장 가능.

---

## 8. Abstract 스토리 구조

### 8.1 논리 흐름

```
1. [배경] 유추 → parallelogram model → 고정 metric → triangle inequality 따라야 함
2. [인간] consistent triangle에서는 유지, mixed triangle에서는 위반 → context-sensitive
3. [질문] LLM은 고차원 벡터 공간에서 관계 표상 → 제약 따를 것으로 예상
         그러나 sequential ICL에서 표상 재조직되면 violation 가능?
4. [방법] Llama-3.1-70B, pure ICL, 인간과 동일한 product test (ABC + ABD triangle)
5. [결과-행동] ABC vs ABD 비교: 대다수 문항에서 ABD >> ABC → 인간과 동일 방향
6. [결과-표상] PCA: mixed context에서 A의 representation이 문맥에 따라 이동
7. [결론] 고정된 context-invariant metric으로 불충분, context가 representation 재조직
```

### 8.2 Abstract에서 각 부분의 역할

| 부분 | 역할 | 데이터 | 비고 |
|---|---|---|---|
| PT ABC vs ABD | **behavioral 증거**: LLM도 인간처럼 ABD > ABC violation 패턴을 보인다 | pt_abc_mean vs pt_abd_mean | 메인 결과 |
| PCA (mixed condition) | **representational 증거**: 왜 그런지 — A가 문맥에 따라 움직인다 | AAA_ref centroid distances | 메커니즘 해석 |
| 인간 ABC vs ABD 비교 | **배경**: 인간에서도 동일 패턴 확인 (ABC < 1, ABD > 1) | 인간 데이터 | 비교 기준 |

---

## 9. 범위 제한 — 명시적으로 빠지는 것

회의 합의와 현재 분석 상태에 따라 abstract에서 **반드시 빼야 하는 것들:**

| 항목 | 빠지는 이유 |
|---|---|
| **Multi-feature reweighting** | 회의에서 기각됨 (3월 26일) |
| **Reasoning model 비교** | 분석 미완, 별도 논문으로 |
| **BD 확장 해석** (latent shared structure) | abstract 범위 밖, 추후 discussion/exploratory finding |
| **Tversky를 강한 결론으로** | 3월 26일 "not yet convinced" |
| **ctx (mixed alternating) PT 결과** | abstract에서는 pure 결과만 다룸 |
| **Causes 수준의 인과 주장** | PCA는 "is consistent with" 수준까지만 |

---

## 10. 톤 가이드라인

### 10.1 써야 하는 톤
- "고정된 metric만으로는 **설명되지 않는다**" (부정이 아니라 불충분)
- "sequential context가 representation을 **재조직할 수 있음을 시사한다**"
- PCA 결과는 "context-dependent representational movement와 **consistent하다**"

### 10.2 피해야 하는 톤
- ✗ "LLM이 인간처럼 사고한다" → 이 논문은 그 주장이 아님
- ✗ "고정된 geometry를 반박한다" → 불충분하다고 말하는 것이지 반박이 아님
- ✗ "AI를 더 인간처럼 만들었다" → 2월 회의에서 명시적으로 경고됨
- ✗ "violation의 원인이 representational movement이다" → causes가 아니라 consistent with

### 10.3 회의록 기반 프레이밍 기준점
- `docs/meeting_txt/12월미팅.txt:291` — sequential processing이 스토리의 출발점
- `docs/meeting_txt/2월 미팅.txt:527,539` — "인간의 context-sensitive violation을 이해하는 데 무엇이 필요한가"
- `docs/meeting_txt/3월19일 미팅.txt:174` — Shepard-like geometry와 contextual feature weighting 연결
- `docs/meeting_txt/3월26일미팅.txt:378` — Tversky 연결은 "not yet convinced"

---

## 11. 핵심 파일 위치

### 데이터
| 항목 | 경로 |
|---|---|
| 인간 rating 데이터 (30명) | `datasets/csv_data/*.csv` |
| 인간 PT bootstrap 결과 | `results/pt_analysis/pt_human_bootstrap_summary.csv` |
| LLM PT bootstrap (5-edge, ABC+ABD) | `/scratch/.../pt_analysis/llama31_70b_20260310_104803/pt_bootstrap_summary.csv` |
| LLM PT bootstrap (unified, ABD only) | `/scratch/.../pt_analysis/llama31_70b_pt_candidate_recompute_full_q8grain_20260325_033335/pt_unified_bootstrap_summary.csv` |
| 관계 구조 맵 | `docs/TRIANGLE_RELATION_ROLE_MAP.md` |
| ICL trial 데이터 | `/scratch/.../relation_condition_qwise/.../Q1/_trials/condition_*.json` |

### PCA 결과
| 항목 | 경로 |
|---|---|
| AAA_ref PCA distances | `/scratch/.../Q1/_pca_common/AAA_ref/distance_summary.json` |
| AAA_ref scatter | quicklook symlink: `q1_aaa_ref_scatter.png` |
| BD_ref PCA distances | `/scratch/.../Q1/_pca_common/BD_ref_BD_compare/distance_summary.json` |
| BD_ref scatter | quicklook symlink: `q1_bd_ref_scatter.png` |

### 코드
| 항목 | 경로 |
|---|---|
| 인간 PT 계산 | `scripts/compute_product_test_bootstrap_humans.py` |
| LLM PT 계산 | `scripts/compute_product_test_bootstrap_unified.py` |
| Target logit scoring | `scripts/score_cross_relation_target_logit.py` |
| Condition PCA | `scripts/run_condition_common_pca.py` |

### 참고 문서
| 항목 | 경로 |
|---|---|
| Thesis 전문 | `docs/thesis_selin_samra.md` |
| 회의록 (12월~3월) | `docs/meeting_txt/` |
| BD PCA sanity check | `reports/2026-03-22-bd-pca-sanity-check-report.md` |
| CCN 템플릿 | `ccn/ccn-template-main 2/ccn_extended_abstract.tex` |
| Storypack 읽기 순서 | `ccn/ccn_extended_abstract_storypack_20260328/README.md` |

---

## 12. 현재 한국어 Abstract 초안 (최신)

> 유추는 한 쌍에서 성립하는 관계를 다른 쌍에 적용해 새로운 대응을 찾는 추론이다. Parallelogram model을 비롯한 고전적 vector-space account는 개체를 공간의 점으로, 관계를 차이벡터로 나타내며, relational similarity를 고정된 기하학적 거리로 설명해 왔다. 이러한 설명이 고정된 metric 공간 위에서 성립한다면, relational similarity는 triangle inequality와 같은 기하학적 제약을 따라야 한다. 실제로 세 쌍이 동일한 관계를 공유하는 consistent-relation triangle에서는 인간의 판단도 이 제약을 대체로 유지한다. 그러나 하나의 쌍이 서로 다른 두 관계에 동시에 참여하는 mixed-relation triangle에서는 인간이 이 제약을 체계적으로 위반한다. 이는 관계 유사도가 고정된 거리가 아니라, 어떤 관계가 활성화되느냐에 따라 달라지는 context-sensitive한 판단임을 보여준다.
>
> 본 연구는 대규모 언어모델에서도 동일한 위반이 나타나는지를 묻는다. Llama-3.1-70B에 순차적 in-context learning으로 동일 관계의 예시만 반복 제시한 뒤, 인간 실험과 동일한 product test를 ABC (consistent)와 ABD (mixed) triangle 모두에 적용했다. 그 결과, consistent triangle에서는 product test가 1 근처를 유지한 반면 mixed triangle에서는 체계적으로 1을 초과하여, 인간과 구조적으로 동일한 violation 패턴이 나타났다. 이어 function vector를 기반으로 조건별 relational representation을 추출하고 common PCA 공간에 투영한 결과, 두 관계가 교대로 제시되는 mixed context에서 A의 표상은 하나의 고정된 점에 머무르지 않고 문맥에 따라 서로 다른 방향으로 이동했다. 이러한 결과는 LLM에서도 relational similarity가 고정된 context-invariant metric만으로는 설명되지 않으며, sequential context가 relational representation 자체를 재조직할 수 있음을 시사한다.

---

## 13. 미해결 이슈 / 결정 필요 사항

| 이슈 | 상태 | 영향 |
|---|---|---|
| LLM ABC도 일부 문항에서 violation | Q8, Q9에서 ABC > ABD (역전) | "consistent triangle 유지"를 단정적으로 쓰면 안 됨. "대다수 문항에서 ABD >> ABC" 정도로 서술 |
| DADA condition이 AAA_ref PCA에 미포함 | trial 존재하지만 PCA에 아직 안 넣음 | "A가 D 방향으로도 이동"을 직접 보여주려면 DADA를 PCA에 추가해야 함 |
| Q별 violation 강도 차이 | 일부 Q에서 인간은 violation 없고 LLM만 있음 (Q5, Q9) | 모든 Q에서 패턴이 동일하다고 과잉 주장하면 안 됨 |
| Product test 공식 선택 근거 | PT = xyz/min² 사용 중 | 왜 이 공식인지에 대한 justification이 본문에 필요할 수 있음 |
| 인간-LLM 직접 비교의 한계 | 측정 방식이 다름 (rating vs logit) | "구조적으로 동일한 패턴"까지는 가능하지만 "동일한 크기"는 주장 불가 |

## 14. 다음 단계

1. **미해결 이슈 결정** — 특히 ABC 주장을 어떻게 처리할지
2. 한국어 초안 최종 검토 후 영어 300단어 abstract 전환
3. CCN tex 파일에 abstract 삽입
4. 본문 2페이지 구성: Introduction → Methods → Results → Discussion
5. Figure 선정 (quicklook에서 후보 확인)
6. References 정리 (ccn_references.bib)
