# Tech Spec: Strict Same-State Inside/Outside Intervention At A_query

## Summary

- 이 문서는 `inside-A`와 `outside-A`의 역할 분리를 **strict same-state intervention**으로 테스트하는 스펙이다.
- 여기서 strict하다는 뜻은 다음과 같다.
  - vector source가 `A_query`
  - intervention target도 같은 `A_query`
  - 둘 다 같은 extracted state space 안에 있다
- 따라서 이 문서의 메인 질문은

> `Q1`의 실제 `A_query` state 자체에 inside/outside component를 직접 더하거나 빼면,  
> 같은 state 안에서 어떤 부분이 endpoint direction을 만들고 어떤 부분이 selection/sharpening을 담당하는가

를 보는 것이다.

- 이 문서는 **behavioral steering / portability test**가 아니다.
- 즉 아래는 이 문서의 main scope가 아니다.
  - unified PT low-shot prompt family
  - final prediction-site residual injection
  - target probability / final-token margin을 main readout으로 두는 실험
  - candidate layer sweep

## What This Spec Is And Is Not

### This Spec Is

- `Q1`, `AAA_ref`, `matched`, `A_query`에서 정의된 **같은 state object**를 직접 조작하는 실험이다.
- intervention은 model의 다른 prompt site에 vector를 옮겨 꽂는 것이 아니라,
  - 이미 추출된 `A_query` summed state
  - 그 state에서 계산된 inside/outside component
  를 같은 measurement space 안에서 그대로 조작한다.

### This Spec Is Not

- low-shot PT behavior prompt에 vector를 옮겨 넣는 steering test가 아니다.
- source는 `A_query`인데 target은 다른 prompt의 final prediction token인 실험이 아니다.
- single-layer residual hook으로 portability를 보는 실험이 아니다.

### Important Boundary

- 이 v1은 **strict within-state intervention in the extracted summed-state space**다.
- 즉
  - source state
  - target state
  - readout state
  가 모두 같은 extracted `A_query` representation space에 있다.
- native model 내부에서 per-layer/per-head로 exact causal patching을 하는 spec은 별도 v2 주제다.

## Core Logic

- representation 분석에서 이미 보인 것:
  - `inside`는 더 크고 main carrier에 가깝다
  - `outside`는 더 작지만 endpoint-selective하다
- strict intervention에서 보고 싶은 것은:
  - 이 둘이 실제로 **같은 A_query state 안에서 서로 다른 역할**을 하는가

### Expected role split

- `inside-only` intervention:
  - intended endpoint direction을 더 많이 복구해야 한다
  - target mixed state의 핵심 inside part를 더 많이 설명해야 한다
  - 즉 content / direction / answer identity 쪽이어야 한다

- `outside-only` intervention:
  - 전체 이동량은 inside보다 작을 수 있다
  - 하지만
    - intended-vs-competing selectivity
    - outside-space endpoint sharpening
    - bundle dominance sharpening
    를 더 선명하게 만들 수 있어야 한다

- `inside + outside` intervention:
  - 실제 mixed target state를 가장 잘 재구성해야 한다
  - role split이 맞다면 둘 중 하나만으로는 안 잡히는 부분까지 함께 설명해야 한다

## State Objects

- main reference:
  - `Q1`
  - `AAA_ref`
  - `matched`
  - `slot = A_query`

### Per-trial states

- `v_A(i)`:
  - `AAA`의 matched `A_query` summed state for trial `i`
- `v_B(i)`:
  - `BABA`의 matched `A_query` summed state for trial `i`
- `v_D(i)`:
  - `DADA`의 matched `A_query` summed state for trial `i`

### Branch deltas

- `Δ_B(i) = v_B(i) - v_A(i)`
- `Δ_D(i) = v_D(i) - v_A(i)`

### Inside/Outside decomposition

- `P_A`:
  - matched `A-basis`로 만든 projector
- `Δ_in_B(i) = P_A Δ_B(i)`
- `Δ_out_B(i) = (I - P_A) Δ_B(i)`
- `Δ_in_D(i) = P_A Δ_D(i)`
- `Δ_out_D(i) = (I - P_A) Δ_D(i)`

이 정의는 이미 stepwise reweighting / inside-outside analysis와 동일한 object를 재사용한다.

## Intervention Objects

이 문서에서는 두 intervention mode를 분리한다.

### 1. Primary: trial-exact mode

- 가장 엄격한 main mode다.
- 같은 trial의 실제 decomposition component를 그대로 쓴다.

#### B branch

- `inside_B(i) = Δ_in_B(i)`
- `outside_B(i) = Δ_out_B(i)`
- `full_B(i) = Δ_B(i) = inside_B(i) + outside_B(i)`

#### D branch

- `inside_D(i) = Δ_in_D(i)`
- `outside_D(i) = Δ_out_D(i)`
- `full_D(i) = Δ_D(i) = inside_D(i) + outside_D(i)`

의미:

- source도 같은 `A_query`
- target도 같은 trial의 `A_query`
- intervention vector도 그 exact trial에서 나온 component

즉 이 mode가 strict same-state intervention의 main mechanistic test다.

### 2. Secondary: mean-vector mode

- robustness / compact summary를 위한 보조 mode다.
- trial별 component를 평균해 shared vector를 만든다.

#### B branch

- `inside_B_mean = mean_i Δ_in_B(i)`
- `outside_B_mean = mean_i Δ_out_B(i)`
- `full_B_mean = inside_B_mean + outside_B_mean`

#### D branch

- `inside_D_mean = mean_i Δ_in_D(i)`
- `outside_D_mean = mean_i Δ_out_D(i)`
- `full_D_mean = inside_D_mean + outside_D_mean`

### Important Interpretation Rule

- `trial-exact`는 main result다.
- `mean-vector`는
  - compactness
  - across-trial stability
  - population-level tendency
  를 보기 위한 secondary result다.
- main claim을 mean-vector mode만으로 세우지 않는다.

## Intervention Targets

intervention target은 언제나 **같은 extracted `A_query` state**다.

### Sufficiency target

- baseline target state:
  - `v_A(i)`
- 질문:
  - `AAA`의 `A_query` state에 inside/outside를 넣으면 실제 mixed endpoint state 쪽으로 어떻게 이동하는가

### Necessity target

- mixed target state:
  - B branch: `v_B(i)`
  - D branch: `v_D(i)`
- 질문:
  - 실제 mixed `A_query` state에서 inside/outside를 빼면 어떤 부분이 무너지는가

### Explicit Non-Goal

- target은 unified PT prompt의 마지막 prediction site가 아니다.
- target은 다른 prompt family의 final token도 아니다.
- 따라서 이 스펙에는 shot axis도, layer sweep도 없다.

## Intervention Conditions

모든 조건은 state arithmetic으로 정의한다.

### B branch: sufficiency on `AAA A_query`

- no intervention:
  - `v_syn = v_A(i)`
- `+ inside_B`
  - `v_syn = v_A(i) + α * inside_B(*)`
- `+ outside_B`
  - `v_syn = v_A(i) + α * outside_B(*)`
- `+ full_B`
  - `v_syn = v_A(i) + α * full_B(*)`

### D branch: sufficiency on `AAA A_query`

- no intervention:
  - `v_syn = v_A(i)`
- `+ inside_D`
  - `v_syn = v_A(i) + α * inside_D(*)`
- `+ outside_D`
  - `v_syn = v_A(i) + α * outside_D(*)`
- `+ full_D`
  - `v_syn = v_A(i) + α * full_D(*)`

### B branch: necessity on real `BABA A_query`

- no intervention:
  - `v_syn = v_B(i)`
- `- inside_B`
  - `v_syn = v_B(i) - α * inside_B(*)`
- `- outside_B`
  - `v_syn = v_B(i) - α * outside_B(*)`
- `- full_B`
  - `v_syn = v_B(i) - α * full_B(*)`

### D branch: necessity on real `DADA A_query`

- no intervention:
  - `v_syn = v_D(i)`
- `- inside_D`
  - `v_syn = v_D(i) - α * inside_D(*)`
- `- outside_D`
  - `v_syn = v_D(i) - α * outside_D(*)`
- `- full_D`
  - `v_syn = v_D(i) - α * full_D(*)`

### Meaning Of `(*)`

- in `trial-exact` mode:
  - `inside_B(*)` means `inside_B(i)` from the same trial
- in `mean-vector` mode:
  - `inside_B(*)` means `inside_B_mean`

## Alpha Schedule

- main reported alpha:
  - `α = 1.0`
- supporting sweep:
  - `α = 0.0, 0.5, 1.0, 1.5`

이유:

- `α = 1.0`은 exact component recovery / removal 해석이 가장 직접적이다
- `α = 0.5, 1.5`는 partial / overshoot behavior를 본다
- `α = 0.0`은 identity sanity check다

## Primary Readouts

이 문서의 main readout은 **state-level**이다.

## 1. Target-State Reconstruction Error

strict same-state intervention의 첫 번째 readout은

> intervened state가 실제 mixed target state를 얼마나 잘 재구성하는가

이다.

### Sufficiency

- B branch:
  - `err_B_target(v_syn) = ||v_syn - v_B(i)||`
- D branch:
  - `err_D_target(v_syn) = ||v_syn - v_D(i)||`

### Necessity

- B branch:
  - `rem_B(v_syn) = ||v_syn - v_A(i)||`
- D branch:
  - `rem_D(v_syn) = ||v_syn - v_A(i)||`

### Inside/Outside error split

- `err_in = ||P_A (v_syn - v_target)||`
- `err_out = ||(I - P_A) (v_syn - v_target)||`

의미:

- inside-only가 target의 inside part를 얼마나 복구하는가
- outside-only가 target의 outside part를 얼마나 복구하는가

## 2. Endpoint-Aligned State Movement

inside/outside는 서로 다른 공간에 있으므로, 각각 그 공간에 맞는 endpoint direction을 쓴다.

### Inside directions

- `u_B^in = normalize(P_A (b-a))`
- `u_D^in = normalize(P_A (d-a))`

### Outside directions

- `u_B^out = normalize((I - P_A)(b-a))`
- `u_D^out = normalize((I - P_A)(d-a))`

### Synthetic change

- sufficiency:
  - `δ_syn = v_syn - v_A(i)`
- necessity:
  - `δ_syn = v_syn - v_A(i)`

즉 necessity도 "ablation 후 남은 change"를 `AAA` 기준으로 본다.

### Alignment readouts

- inside:
  - `T_in^B = <P_A δ_syn, u_B^in>`
  - `C_in^B = <P_A δ_syn, u_D^in>`
  - `T_in^D = <P_A δ_syn, u_D^in>`
  - `C_in^D = <P_A δ_syn, u_B^in>`
- outside:
  - `T_out^B = <(I - P_A) δ_syn, u_B^out>`
  - `C_out^B = <(I - P_A) δ_syn, u_D^out>`
  - `T_out^D = <(I - P_A) δ_syn, u_D^out>`
  - `C_out^D = <(I - P_A) δ_syn, u_B^out>`

### Selectivity margins

- `S_in^B = T_in^B - C_in^B`
- `S_out^B = T_out^B - C_out^B`
- `S_in^D = T_in^D - C_in^D`
- `S_out^D = T_out^D - C_out^D`

의미:

- inside-only가 intended direction 자체를 얼마나 잘 복구하는가
- outside-only가 competing direction과 구분되는 선택성을 얼마나 강화하는가

## 3. Joint Selectivity In Each Subspace

inside/outside는 B/D 방향이 완전히 직교하지 않을 수 있으므로 joint decomposition을 같이 본다.

### B branch

- inside:
  - `P_A δ_syn ≈ α_B^in u_B^in + β_D^in u_D^in + ε_in`
- outside:
  - `(I - P_A) δ_syn ≈ α_B^out u_B^out + β_D^out u_D^out + ε_out`

### D branch

- inside:
  - `P_A δ_syn ≈ α_D^in u_D^in + β_B^in u_B^in + ε_in`
- outside:
  - `(I - P_A) δ_syn ≈ α_D^out u_D^out + β_B^out u_B^out + ε_out`

main margins:

- B branch:
  - `J_in^B = α_B^in - β_D^in`
  - `J_out^B = α_B^out - β_D^out`
- D branch:
  - `J_in^D = α_D^in - β_B^in`
  - `J_out^D = α_D^out - β_B^out`

이 지표는 단순 cosine보다 더 엄격한 selectivity readout이다.

## 4. Bundle Dominance On The Synthetic Change

기존 endpoint-anchored bundle을 synthetic change에 그대로 적용한다.

- `F_B(δ_syn) = ||U_B^T δ_syn||^2`
- `F_D(δ_syn) = ||U_D^T δ_syn||^2`

dominance:

- B branch:
  - `M_B(δ_syn) = F_B(δ_syn) - F_D(δ_syn)`
- D branch:
  - `M_D(δ_syn) = F_D(δ_syn) - F_B(δ_syn)`

의미:

- outside-only가 전체 이동량은 작아도 bundle dominance를 sharper하게 만들 수 있는지 본다

## 5. Norm-Normalized Efficacy

- inside vector는 보통 더 크다.
- 따라서 모든 main readout은
  - raw value
  - intervention norm으로 나눈 normalized value
  를 둘 다 보고한다.

예:

- `S_out^B / ||(I-P_A)δ_syn||`
- `M_B(δ_syn) / ||δ_syn||`
- `err_reduction / ||intervention||`

이 normalized view는

> outside가 작지만 선택적으로 효율적인가

를 보기 위해 필요하다.

## What Counts As Support For The Role Split

### First: what does NOT count as evidence

- `trial-exact` mode에서
  - `v_A(i) + full_B(i) = v_B(i)`
  - `v_B(i) - full_B(i) = v_A(i)`
  - `v_A(i) + full_D(i) = v_D(i)`
  - `v_D(i) - full_D(i) = v_A(i)`
  는 **정의상 성립하는 sanity check**다.
- 이것만으로 role split evidence라고 주장하지 않는다.

### If `inside` is content / direction

- sufficiency on `AAA`:
  - `inside-only` should reduce `err_in` more than `outside-only`
  - `inside-only` should increase `T_in` and `J_in` more than `outside-only`
  - `inside-only` should recover more of the intended target-state core
- necessity on real mixed state:
  - removing `inside` should collapse intended inside-direction more strongly
  - removing `inside` should reduce remaining intended progress more strongly

### If `outside` is selection / sharpening

- sufficiency on `AAA`:
  - `outside-only` may show smaller total movement
  - but should improve `S_out`, `J_out`, and normalized `M_B/M_D`
  - especially per unit norm, outside should look more selective
- necessity on real mixed state:
  - removing `outside` may leave some intended direction alive
  - but should reduce selectivity / sharpening metrics more than it reduces raw total movement

### If both are needed together

- `full` should give:
  - the smallest target-state reconstruction error
  - the strongest joint intended alignment
  - the cleanest overall bundle dominance

## Scope

- main q:
  - `Q1`
- main reference:
  - `AAA_ref`
- main basis scope:
  - `matched`
- main slot:
  - `A_query`
- main mode:
  - `trial-exact`

### Not main in v1

- `union_ref`
  - robustness only
- mean-vector mode
  - secondary only
- other slots
  - appendix only
- next-token behavior readout
  - appendix only

## Inputs

- extracted state input:
  - `stepwise_a_states_AAA_ref.npz`
- reweighting / projector / endpoint anchor input:
  - `stepwise_reweighting_arrays_AAA_ref.npz`
- inside/outside decomposition input:
  - `stepwise_inside_outside_endpoint_arrays.npz`
- optional joint readout helper input:
  - `stepwise_inside_outside_joint` outputs or equivalent recomputation

## Implementation Shape

- add one strict state-intervention prep layer
  - loads `A_query` trial states
  - loads `P_A`, `u_B^in`, `u_D^in`, `u_B^out`, `u_D^out`
  - materializes trial-exact and mean intervention objects

- add one strict state-intervention runner
  - no model loading
  - no prompt rebuilding
  - no residual hook
  - constructs `v_syn` offline in the same state space
  - scores all branch × mode × condition × alpha combinations

- add one summarizer
  - collapses raw rows into branch × mode × alpha summary tables
  - emits interpretation-ready markdown

## Outputs

- `inside_outside_state_intervention_vectors_Q1.npz`
  - stores trial-exact and mean vectors
- `inside_outside_state_intervention_raw_rows_Q1.csv`
- `inside_outside_state_intervention_summary_Q1.csv`
- `inside_outside_state_intervention_synthetic_states_Q1.npz`
- `inside_outside_state_intervention_report_Q1.md`

## Test Plan

- vector sanity
  - for every trial:
    - `inside + outside = full`
  - for mean vectors:
    - `inside_mean + outside_mean = full_mean`

- identity sanity
  - `α = 0` reproduces the original state exactly

- exact reconstruction sanity in `trial-exact`, `α = 1`
  - `AAA + full_B = BABA`
  - `AAA + full_D = DADA`
  - `BABA - full_B = AAA`
  - `DADA - full_D = AAA`

- target-space sanity
  - no shot axis appears
  - no layer axis appears
  - no unified PT family appears in runner outputs

- readout sanity
  - reconstruction errors are finite
  - inside/outside alignment metrics are finite when norms are non-zero
  - normalized metrics handle zero-norm safely

- role split sanity
  - `inside-only` has larger effect on intended inside-direction recovery
  - `outside-only` has relatively larger effect on selectivity / sharpening metrics
  - `full` gives best overall reconstruction

- robustness sanity
  - mean-vector mode should preserve the same qualitative direction as trial-exact mode, even if weaker

## Assumptions

- v1 is `Q1` only.
- main reference is `AAA_ref`.
- main slot is `A_query`.
- main result is `trial-exact` same-state intervention.
- `mean-vector` is robustness only.
- this spec is strict with respect to the extracted `A_query` state space.
- behavioral steering on another prompt/site is a different experiment and should live in a different document.
