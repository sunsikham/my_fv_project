# Plan: Representation Priority Realignment

## Summary

- 현재 단계에서 다음 representation 분석의 우선순위는 다시 정렬하는 것이 좋다.
- 핵심은 단순 magnitude가 아니라,

> inside-A와 outside-A 중 누가 실제 endpoint-aligned shift를 설명하느냐

를 먼저 보는 것이다.

- 따라서 우선순위는 다음 세 단계로 재정렬한다.
  1. `inside/outside endpoint contribution`
  2. `inside/outside joint decomposition`
  3. `signed contribution structure`

## Why Reprioritize

- 지금까지는
  - `outside-A`가 존재한다
  - stepwise에선 `inside-A`가 더 크다
까지는 확인했다.
- 하지만 이것은 importance가 아니라 **magnitude**만 본 것이다.

- 중요한 건:
  - 큰데도 endpoint와 무관할 수 있다
  - 작아도 endpoint 방향에 정확히 정렬되면 중요할 수 있다

- 따라서 representation 단계의 다음 질문은

> 단순히 inside/outside가 얼마나 큰가가 아니라,  
> inside/outside가 intended endpoint alignment를 얼마나 설명하는가

여야 한다.

## Priority 1: Inside/Outside Endpoint Contribution

- 먼저

`Δ = Δ_in + Δ_out`

로 나눈다.

여기서

- `Δ_in = P_A Δ`
- `Δ_out = (I - P_A) Δ`

- 그리고 각각에 대해 따로 본다.
  - intended endpoint alignment
  - cross endpoint alignment
  - norm

### Basis Choice

- main basis는 `matched basis`로 고정한다.
- 이유:
  - `Δ_t` 자체가 matched baseline으로 정의되어 있다
  - `BABABA/DADADA`와 직접 대응되는 A-space를 쓰는 것이 inside/outside 해석에 가장 엄격하다
- `all basis`는 robustness / sensitivity check로만 둔다

### Inside / Outside Endpoint Directions

- inside-A 방향은 A-space 안으로 투영한 endpoint direction을 쓴다.

`u_B^in = normalize(P_A (b-a))`

`u_D^in = normalize(P_A (d-a))`

- outside-A 방향은 A-space 밖 성분으로 정의한 endpoint direction을 쓴다.

`u_B^out = normalize((I - P_A)(b-a))`

`u_D^out = normalize((I - P_A)(d-a))`

- 즉
  - inside 변화는 inside용 endpoint direction과 비교
  - outside 변화는 outside용 endpoint direction과 비교
해야 한다.

예:

- `cos(Δ_in, u_B^in)`
- `cos(Δ_in, u_D^in)`
- `cos(Δ_out, u_B^out)`
- `cos(Δ_out, u_D^out)`

### Why This Comes First

- 이 단계가 먼저 있어야 아래 같은 해석이 가능해진다.

예:

- inside-A가 거의 모든 intended alignment를 설명한다
- outside-A는 존재하지만 endpoint-selective하지 않다
- outside-A가 작지만 sharpening에 기여한다

즉

> outside-A의 역할을 크기(magnitude)가 아니라 기능(function)으로 설명

할 수 있게 된다.

## Priority 2: Inside/Outside Joint Decomposition

- cosine만 보면 다시 separate projection 문제로 돌아갈 수 있다.
- B축과 D축이 비직교일 수 있으므로, inside/outside 각각에 대해 joint decomposition을 적용하는 것이 좋다.

예:

`Δ_in ≈ α_B^in u_B + β_D^in u_D + ε_in`

`Δ_out ≈ α_B^out u_B + β_D^out u_D + ε_out`

### Why This Is Important

- 이렇게 해야 다음과 같은 문장을 정당하게 말할 수 있다.

예:

- B-like part는 mostly inside-A에서 온다
- D contamination은 outside가 아니라 inside의 mixed facet에서 온다
- outside는 B/D 둘로 잘 설명 안 되는 extra restructuring이다

즉

> outside-A의 역할을 “있다/없다”가 아니라 “무엇을 하느냐”로 설명

할 수 있게 된다.

## Priority 3: Signed Contribution Structure

- 이건 reweighting hypothesis와 직접 연결되는 분석이다.
- 핵심 질문:

> B-like shift가  
> B-aligned feature를 올려서 생긴 건지,  
> anti-B feature를 내려서 생긴 건지,  
> 둘 다인지

- 이 분석은 다음 값을 본다.

`contrib_k^B(t) = Δc_{t,k} a_k^B`

`contrib_k^D(t) = Δc_{t,k} a_k^D`

### Why This Matters

- 만약 inside-A main mechanism이 진짜라면,
  inside-A 내부에서 이미 존재하던 feature weight의 재배치가 보여야 한다.
- 따라서 signed contribution은 단순 보조 분석이 아니라,

> “reweighting”이라는 말의 가장 직접적인 representational 증거

가 된다.

### Practical Note

- Priority 3는 1,2가 끝난 뒤 바로 이어서 보는 것이 좋다.
- 사실상 1,2,3은 한 묶음의 representation explanation stack이다.

## Recommended Priority Stack

### 1. Inside/Outside Endpoint Selectivity

- `matched basis`를 main으로 사용
- `Δ_in`, `Δ_out` 각각의 intended/cross alignment
- inside는 `u_B^in, u_D^in`
- outside는 `u_B^out, u_D^out`

### 2. Inside/Outside Joint Decomposition

- `Δ_in ≈ α_B^in u_B^in + β_D^in u_D^in + ε_in`
- `Δ_out ≈ α_B^out u_B^out + β_D^out u_D^out + ε_out`

### 3. Signed Contribution Structure

- inside-A 기준 signed contribution
- 필요하면 outside-A contribution도 별도 정리

## Why This Order Is Better

- 이 순서는
  - magnitude
  - mechanism
  - reweighting structure
를 논리적으로 분리해준다.

- 기존보다 더 잘 답하는 질문:

> one-axis failure의 원인이  
> inside-A structured reweighting인지,  
> outside-A reshaping인지,  
> 또는 둘의 조합인지

## Bottom Line

- 현재 representation 단계에서 가장 중요한 질문은

> inside-A와 outside-A 중 누가 endpoint-aligned shift를 실제로 설명하느냐

이다.

- 따라서 다음 우선순위는
  1. inside/outside endpoint contribution
  2. inside/outside joint decomposition
  3. signed contribution structure

로 재정렬하는 것이 가장 타당하다.
