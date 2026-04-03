# Tech Spec: Stepwise Endpoint-Aligned Contributions

## Summary

- 이 문서는 stepwise reweighting metrics 위에 한 단계 더 얹는 **endpoint-aligned contribution analysis**를 정의한다.
- 목적은 단순히
  - 여러 feature가 바뀌었다
  - B/D dominance가 커진다
를 넘어서,

> 실제로 stepwise 변화가 A-space 안에서 B/D endpoint 방향으로 정렬되는가,  
> 그리고 그 정렬이 어떤 feature들의 signed contribution으로 만들어지는가

를 보는 것이다.

## Primary Basis Choice

- main analysis는 `matched basis`를 사용한다.
- 이유:
  - stepwise `Δ_t` 자체가 matched baseline으로 정의되어 있다
  - mixed prompt와 직접 대응되는 A-space를 쓰는 것이 가장 엄격하다
- `all basis`는 robustness / sensitivity check로만 둔다.

## Why This Layer Is Needed

- 기존 stepwise reweighting은 아래를 잘 보여준다.
  - `Δc_t`
  - `PR_deltac`
  - `R_t`
  - `inside/outside_change`
  - `M(t)`
- 하지만 이것만으로는 아직 부족하다.
- 왜냐하면
  - “여러 feature가 바뀐다”
  와
  - “그 변화가 실제로 B endpoint 쪽을 향한다”
  는 다른 주장이다.

- 따라서 다음 단계는
  - feature reweighting
  - endpoint alignment
를 직접 연결하는 지표가 필요하다.

## Core Idea

- A-basis `G_A` 위에서 stepwise coefficient drift `Δc_t`를 이미 계산했다면,
- 각 A-feature axis `g_k`가 B/D endpoint 방향과 얼마나 정렬되는지 먼저 계산한다.
- 그리고 각 step의 `Δc_{t,k}`와 곱해서,

> 어떤 feature가 실제로 B-like shift에 얼마나 기여했는가

를 signed하게 본다.

## 1. Use A-Space Projected Endpoint Directions

- raw `b-a`, `d-a`를 그대로 쓰지 않는다.
- 먼저 A-space로 endpoint direction을 투영한다.

`u_B^(A) = normalize(P_A (b-a))`

`u_D^(A) = normalize(P_A (d-a))`

여기서
- `P_A`는 A-basis `G_A` 위로의 projection
- `b-a`, `d-a`는 q-local clean endpoint direction

이 정의가 중요한 이유:
- 네 가설은
  - A 구조를 유지한 채
  - A-space 내부 feature들이 재가중된다는 것
이므로,
- feature alignment도 **A-space 안에서** 정의해야 한다.

## 1.1 Inside/Outside Split Uses Different Endpoint Directions

- inside/outside decomposition을 할 때는 같은 endpoint direction을 그대로 재사용하지 않는다.
- inside와 outside는 서로 다른 공간에 있으므로, 각각 그 공간에 맞는 endpoint direction을 써야 한다.

### Inside directions

`u_B^in = normalize(P_A (b-a))`

`u_D^in = normalize(P_A (d-a))`

### Outside directions

`u_B^out = normalize((I - P_A)(b-a))`

`u_D^out = normalize((I - P_A)(d-a))`

의미:

- inside 변화는 inside용 B/D 방향과 비교
- outside 변화는 outside용 B/D 방향과 비교

이 정의를 써야 inside/outside가 일관되게 비교된다.

## 2. Feature-Level Alignment Weights

- 각 A-basis feature axis `g_k`에 대해

`a_k^B = <g_k, u_B^(A)>`

`a_k^D = <g_k, u_D^(A)>`

를 계산한다.

의미:
- `a_k^B`
  - feature `k`가 B endpoint 방향과 얼마나 정렬되는가
- `a_k^D`
  - feature `k`가 D endpoint 방향과 얼마나 정렬되는가

이 값들은 hard split label이 아니라 **soft alignment weight**다.

## 3. Main Readout: Signed Total Alignment

- 각 step의 A-space 변화는

`Δ_A(t) = G_A Δc_t`

로 볼 수 있다.

- 그러면 B 방향 정렬의 전체량은

`T_B(t) = <Δ_A(t), u_B^(A)>`

가 된다.

- basis로 풀면

`T_B(t) = Σ_k Δc_{t,k} a_k^B`

마찬가지로

`T_D(t) = Σ_k Δc_{t,k} a_k^D`

### Why This Is The Main Metric

- `T_B(t)`는
  - 어떤 feature가 얼마나 바뀌었는지
  - 그 feature가 B와 얼마나 정렬되는지
를 한 번에 묶는다.
- 즉

> 전체 변화량 자체가 A-space 안에서 B endpoint 방향으로 얼마나 signed하게 정렬되었는가

를 바로 보여준다.

### Expected Pattern

- `BABABA`
  - `T_B(t)` 증가
  - `T_D(t)`는 더 작거나 cross 정도
- `DADADA`
  - `T_D(t)` 증가
  - `T_B(t)`는 더 작음

## 4. Feature-Level Signed Contribution

- 각 feature의 실제 B 기여도는

`contrib_k^B(t) = Δc_{t,k} a_k^B`

- D 쪽은

`contrib_k^D(t) = Δc_{t,k} a_k^D`

의미:
- `Δc_{t,k}`
  - step마다 feature `k`가 얼마나 바뀌었나
- `a_k^B`
  - 그 feature가 B 방향과 얼마나 정렬되는가
- `contrib_k^B(t)`
  - 그 feature가 실제 B-like shift에 얼마나 기여했는가

이 지표는

> “많이 바뀐 feature가 실제로 B endpoint와 정렬된 feature들이냐?”

를 가장 직접적으로 답해준다.

## 5. Positive vs Negative Contribution Split

- 단순 total contribution만 보면,
  - B-aligned feature가 증가한 건지
  - B-aligned feature가 반대로 움직인 건지
가 섞일 수 있다.

- 그래서 positive / negative split을 같이 본다.

`T_B^+(t) = Σ_k max(0, contrib_k^B(t))`

`T_B^-(t) = Σ_k max(0, -contrib_k^B(t))`

마찬가지로

`T_D^+(t), T_D^-(t)`

### Interpretation

- `T_B^+`가 크고 step 따라 커지면
  - B endpoint 쪽으로 실제로 누적되는 기여가 커진다
- `T_B^-`가 작으면
  - B 방향을 거스르는 feature contribution은 상대적으로 작다

## 6. Soft-Weighted Energy (Secondary)

- hard split energy보다 soft-weighted energy가 더 좋다.

`E_B(t) = Σ_k (a_k^B)^2 (Δc_{t,k})^2`

`E_D(t) = Σ_k (a_k^D)^2 (Δc_{t,k})^2`

의미:
- B/D와 정렬된 feature들 위에서 변화량의 크기 자체가 얼마나 커졌는가

### Role

- `T_B(t)` / `T_D(t)`가 메인
- `E_B(t)` / `E_D(t)`는 보조

이유:
- `E_B`는 부호를 잃기 때문
- signed endpoint accumulation은 `T_B`가 더 잘 보여준다

## 7. Multi-Feature Contributor Concentration

- 네 논지는 one-axis replacement가 아니라 multi-feature reweighting이다.
- 그러면 `T_B(t)`를 만든 기여가 여러 feature에 퍼져 있는지도 봐야 한다.

추천 지표:

`PR_{B,contrib}(t) = (Σ_k |contrib_k^B(t)|)^2 / Σ_k (contrib_k^B(t))^2`

`PR_{D,contrib}(t) = (Σ_k |contrib_k^D(t)|)^2 / Σ_k (contrib_k^D(t))^2`

### Interpretation

- `PR ≈ 1`
  - 사실상 한 feature collapse
- `PR > 1`
  - 여러 feature가 같이 기여

즉 이 지표는

> endpoint-aligned shift가 one-feature effect인지, 여러 feature의 누적인지

를 직접 보여준다.

## 8. Recommended Analysis Order

### A. Bundle-level signed alignment

먼저 본다:

- `T_B(t)`
- `T_D(t)`
- `T_B(t) - T_D(t)`

이게 메인 readout이다.

### B. Feature-level signed contribution

그다음 본다:

- `contrib_k^B(t)`
- `contrib_k^D(t)`

이걸로 어떤 feature들이 실제 endpoint shift를 만들었는지 본다.

### C. Positive/negative split

- `T_B^+(t)`, `T_B^-(t)`
- `T_D^+(t)`, `T_D^-(t)`

### D. Concentration / diversity

- `PR_{B,contrib}(t)`
- `PR_{D,contrib}(t)`

### E. Secondary energy

- `E_B(t)`
- `E_D(t)`

## 9. Main Expected Signature

### `BABABA`

- `T_B(t)` 증가
- `T_D(t)`보다 큼
- `T_B^+(t)` 증가
- `PR_{B,contrib}(t)`가 1로 collapse하지 않음

### `DADADA`

- `T_D(t)` 증가
- `T_B(t)`보다 큼
- `T_D^+(t)` 증가
- `PR_{D,contrib}(t)`가 1로 collapse하지 않음

## 10. Stronger Project Claim Enabled By This Layer

- 기존 문장:

> multi-feature reweighting이 stepwise로 누적된다

- 이 layer가 추가되면 더 강하게 이렇게 말할 수 있다.

> B context does not simply replace A. Instead, it stepwise accumulates signed contributions from multiple A-space features that are aligned with the q-local B endpoint.

- 한국어:

> B context는 A를 단순히 교체하지 않는다. 대신 A-space 안에서 q-local B endpoint와 정렬된 여러 feature들의 signed contribution을 stepwise하게 누적시킨다.

## 11. Recommended Scope

- main basis:
  - `matched_5slot_basis`
- robustness:
  - `all_10slot_basis`

- main reference:
  - `AAA_ref`

## 12. Bottom Line

- feature를 hard하게 B/D/neutral로 나누기 전에,
  먼저 **A-space projected endpoint direction**을 만들고
  그 위에서

`T_B(t) = Σ_k Δc_{t,k} a_k^B`

라는 **signed endpoint-aligned total contribution**을 메인 readout으로 두는 것이 더 정확하고 더 강한 분석이다.
