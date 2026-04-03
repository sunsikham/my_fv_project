# Plan: Relation-Aligned Bundle Validation

## Summary

- `feature reweighting`만으로는 어떤 bundle이 곧바로 `relation`이라고 말할 수 없다.
- 현재 strongest claim은 global semantic relation이 아니라, **q-local B/D endpoint와 정렬된 relation-aligned bundle**이다.
- 따라서 `multi-feature reweighting` 다음 단계는:
  - `clean endpoint alignment`
  - `B vs D discrimination`
  - `behavioral relevance`
  - optional `causal intervention`
  를 통해 bundle의 의미를 더 강하게 검증하는 것이다.

## 1. Feature And Relation Are Not The Same

- `G_A`, `U_B`, `U_D`에서 나오는 축들은 기본적으로 data-driven basis다.
- 즉 이 축들은
  - PCA/SVD로 뽑은 방향
  - delta에서 반복적으로 나타나는 변화 방향
  - high-variance / high-repeatability structure
  이지, 자동으로 semantic relation label이 붙는 것은 아니다.
- 따라서
  - `feature = relation`
  이라고 바로 주장하면 과하다.

## 2. What Can Be Claimed Right Now

### Level 1: Mechanistic Claim

- `BABABA`나 `DADADA`에서 특정 bundle의 coefficient가 커진다.
- `PR`이 1로 collapse하지 않고 여러 축이 동시에 active하다.
- 이는 `one-axis replacement`가 아니라 `multi-feature reweighting`이라는 기계적 주장이다.

### Level 2: q-Local Relation-Aligned Claim

- 어떤 bundle이 이 q에서 정의된 clean `B` endpoint와 정렬된다.
- 그리고 competing `D` endpoint와는 구분된다.
- 이 수준의 주장은 현재 데이터 구조에서 가능하다.

### Level 3: Global Relation Claim

- 여러 q에서 동일한 semantic relation family가 반복된다는 주장.
- 이건 q마다 `B relation`의 의미가 다르면 바로 할 수 없다.
- relation family annotation이나 cross-q semantic grouping이 따로 필요하다.

## 3. The Right Claim For The Current Project

- 지금 가장 정직한 주장은 다음이다.

> In each q, B context increases a q-local, behavior-relevant, B-aligned feature bundle rather than collapsing A into a single axis.

- 한국어로는 다음처럼 표현할 수 있다.

> 각 q에서 B context는 A를 단일 축으로 교체하는 것이 아니라, 그 q의 clean B endpoint와 정렬된 여러 feature bundle의 가중치를 증가시킨다.

## 4. How To Show That A Bundle Is B-Related

## 4.1 Clean Endpoint Alignment

- 각 q에서 clean anchor를 정의한다.
  - `a = mean(AAAAAA)`
  - `b = mean(BBBBBB)`
  - `d = mean(DDDDDD)`
- 이때 q-local clean B direction은

`b - a`

- stepwise change 또는 bundle reconstruction이 이 방향과 얼마나 정렬되는지 본다.

예:

`align_B(t) = cos(Δ_t^{BAB}, b - a)`

또는

`align_B_bundle(t) = cos(Δhat_t^{BAB}, b - a)`

- 기대 패턴:
  - `BABABA`에서 step이 갈수록 `align_B` 증가
  - `DADADA`에서는 같은 값이 더 작음

- 이게 성립하면 그 bundle은 적어도
  - `q-local clean B endpoint`
  와 정렬된다고 말할 수 있다.

## 4.2 B-vs-D Discrimination

- alignment만으로는 부족하다.
- 같은 bundle에 대해 B와 D 양쪽 정렬을 동시에 본다.

예:

`cos(Δhat_t^{BAB}, b - a)` vs `cos(Δhat_t^{BAB}, d - a)`

- 기대 패턴:
  - `BABABA`의 bundle은 `B` 정렬이 더 큼
  - `DADADA`의 bundle은 `D` 정렬이 더 큼

- 이게 되면 bundle은 단순한 generic change가 아니라
  - `B/D`를 구분하는 q-local relation-relevant signal
  로 해석할 수 있다.

## 4.3 Behavioral Relevance

- feature reweighting만으로는 부족하다.
- 그 bundle이 실제 output behavior와 연결되어야 한다.

예:

- bundle strength가 커질수록
  - `B target probability`가 올라가는가
  - `B answer rank`가 올라가는가
  - answer-vs-distractor score가 좋아지는가

- 예시 metric:

`F_B(t) = Σ_k alpha_{t,k}^{B 2}`

- 기대 패턴:
  - `F_B(t)`가 클수록 B-like behavior가 강해진다

- 이게 되면 그 bundle은
  - 단순 분산축
  이 아니라
  - behaviorally relevant B bundle
  이라고 말할 수 있다.

## 4.4 Causal Intervention (Optional Strong Test)

- 가장 강한 증거는 causal intervention이다.
- 예:
  - `AAAAAA`의 A state에 B-bundle 방향 성분을 더한다
  - 또는 `BABABA` state에서 그 성분을 제거한다

예:

`v' = v + λ Δhat_B`

또는

`v' = v - λ Δhat_B`

- 그리고 output이
  - 더 B-like해지는지
  - 덜 B-like해지는지
  를 본다.

- 이게 성립하면 그 bundle은
  - 단순 정렬
  이 아니라
  - 실제로 B answer를 미는 causal relation-aligned structure
  라고 훨씬 강하게 주장할 수 있다.

## 5. What Is Not Enough

- 아래만으로는 아직 부족하다.
  - `Δc_t`가 커진다
  - `PR > 1`
  - residual이 남는다

- 이건
  - multi-feature reweighting
  자체는 지지하지만,
  - 그 bundle이 `B relation`과 연결된다는 건 직접 보이지 않는다.

## 6. What Should Be Reported

- 현재 단계에선 아래 표현이 가장 안전하다.

### Recommended Safe Phrasing

- `B-aligned bundle`
- `q-local B-endpoint-aligned bundle`
- `behavior-relevant B bundle`
- `relation-aligned update bundle`

### Avoid For Now

- `this feature is the B relation`
- `this PCA component is the semantic relation itself`
- `the same relation bundle is shared across all q`

## 7. Stronger Project Statement

- 네 가설을 현재 프로젝트 수준에서 가장 강하게 쓰면 다음과 같다.

> B context does not simply replace A with a single axis. Instead, it increases a q-local, behavior-relevant bundle of representation changes that is more aligned with the clean B endpoint than with the competing D endpoint.

- 한국어 버전:

> B context는 A를 단일 축으로 교체하는 것이 아니라, 그 q의 clean B endpoint와 더 잘 정렬되고 competing D endpoint와는 구분되는, behaviorally relevant한 변화 bundle의 가중치를 증가시킨다.

## 8. Minimal Validation Checklist

- `multi-feature reweighting` 결과가 먼저 있어야 한다
  - `Δc_t`
  - `PR`
  - `active_count`
  - retained-A / inside-change metrics
- 그 다음 아래를 추가해야 한다
  - `clean B alignment`
  - `clean D alignment`
  - `behavior correlation`
- 가능하면 마지막에
  - `intervention`

## 9. Bottom Line

- `feature reweighting`은 필요하지만 충분하지 않다.
- relation-aligned bundle claim을 하려면, 그 bundle이
  - q-local clean B direction과 정렬되고
  - D와 구분되며
  - behavior와 연결되고
  - 가능하면 intervention에도 causal하게 작동한다
  는 것을 보여야 한다.

