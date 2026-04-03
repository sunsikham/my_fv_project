# Tech Spec: Stepwise A-State Extraction For PCA Trajectory And Multi-Feature Reweighting

## Summary

- 목표는 `AAAAAA`, `BABABA`, `DADADA`에서 **A가 등장하는 각 위치의 state**를 다시 추출해서 저장하는 것이다.
- 이 추출은 두 가지 다운스트림 분석을 동시에 지원해야 한다.
  - `stepwise PCA trajectory`
  - `A-basis / multi-feature reweighting`
- 중요한 점:
  - **StepD head ranking을 다시 할 필요는 없다**
  - 하지만 **새 forward pass는 필요하다**
  - 이유는 현재 저장본이 거의 모두 `final query` 위치 한 점만 담고 있기 때문이다

## Why This Exists

- 현재 q-wise vector artifact는 대부분 `A_query` 한 점만 저장한다.
- 그래서 지금 가능한 것은:
  - final-query movement
  - final-query reweighting
- 아직 불가능한 것은:
  - `A_demo_1 -> A_demo_2 -> A_demo_3 -> A_demo_4 -> A_query` trajectory
  - matched stepwise `Δ_t`
  - stepwise `Δc_t`
  - stepwise PCA GIF
- 따라서 먼저 필요한 것은 **stepwise state extraction layer**다.

## StepD vs Re-Extraction

- `StepD`가 하는 일:
  - 중요한 head를 찾는 ranking / selection
- 이번 단계가 하는 일:
  - 이미 선택된 head pool을 고정한 채
  - 같은 trial prompt를 다시 forward해서
  - `A`가 나타나는 여러 위치의 state를 저장
- 결론:
  - **head ranking 재계산은 불필요**
  - **state 재추출 forward는 필요**

## Immediate Scope

- v1은 `Q1`을 먼저 지원한다.
- 그 다음 같은 구조로 `Q8`, `Q9`, `Q10`, `Q11`로 확장한다.
- v1에서는 아래 두 reference를 동시에 저장한다.
  - `AAA_ref`
  - `union_ref`

## Head Pools

### `AAA_ref`

- source:
  - existing `top_heads_ref_AAA.json`
- 의미:
  - A-only baseline 해석에 가장 깔끔한 reference
- 장점:
  - `G_A`와 coefficient drift 해석이 자연스럽다
- 한계:
  - `B`/`D` 쪽에서 중요하지만 `AAA`에는 약한 head를 놓칠 수 있다

### `union_ref`

- source:
  - union of existing condition-level top-head selections
- canonical definition:
  - `union(top20_AAA, top20_BBB, top20_BABA, top20_DDD, top20_DADA)`
- 의미:
  - cross-condition geometry를 같은 측정기에서 보기 위한 reference
- 장점:
  - `AAA/BABA/DADA`를 더 공정하게 비교할 수 있다
- 한계:
  - q별 artifact completeness가 불균일할 수 있다

### Important Rule

- **step마다 head pool을 다시 뽑지 않는다**
- **condition마다 다른 pool을 써서 stepwise trajectory를 만들지 않는다**
- 한 번 정한 ref별 head pool을 모든 step에 동일하게 쓴다

## Slot Definition

### Mixed Conditions

- `BABABA`와 `DADADA`에서 A는 아래 다섯 위치에 등장한다.
  - `A_demo_1`
  - `A_demo_2`
  - `A_demo_3`
  - `A_demo_4`
  - `A_query`

### `AAAAAA` Matched Baseline

- `AAAAAA`는 A demo가 더 많지만, matched baseline 비교에는 아래 다섯 위치만 쓴다.
  - `A_demo_2`
  - `A_demo_4`
  - `A_demo_6`
  - `A_demo_8`
  - `A_query`
- 이유:
  - `BABABA`/`DADADA`의 A 위치와 prefix depth를 더 잘 맞출 수 있기 때문이다

### Why Matched 5 Slots Matter

- `Δ_t = v_t^{mixed} - v_t^A`는 **동일한 역할 / 동일한 prefix depth**에서 비교해야 한다
- 따라서 `Δ_t`의 primary baseline은 반드시 **position-matched 5 slots**로 만든다

## Two Uses Of `AAAAAA`

### 1. Matched Baseline For `Δ_t`

- 목적:
  - `BABABA` 또는 `DADADA`에서 A가 어떻게 바뀌었는지 직접 비교
- 사용 위치:
  - matched 5 slots only

### 2. A-Only Basis Learning For `G_A`

- 목적:
  - 원래 A-only feature space를 학습
- 두 버전을 같이 지원한다.
  - `matched_5slot_basis`
  - `all_10slot_basis`

### Meaning Of The Two `G_A` Variants

- `matched_5slot_basis`
  - `AAAAAA`의 matched 5 slots만 써서 만든 basis
  - 해석이 더 엄격하다
  - mixed prompt와 직접 비교되는 A positions만 반영한다

- `all_10slot_basis`
  - `AAAAAA`의 모든 demo A + query A를 써서 만든 basis
  - 더 넓고 안정적인 A-only subspace를 반영한다
  - 보조 / 안정성 체크에 적합하다

## Conditions To Extract

- 반드시 추출:
  - `AAA`
  - `BABA`
  - `DADA`
- 이번 단계에서 optional:
  - `BBB`
  - `DDD`
- 이유:
  - stepwise A-state trajectory 자체는 `AAA/BABA/DADA`만 있으면 된다
  - `BBB/DDD`는 이후 `U_B/U_D` 또는 anchor overlay에서 보조적으로 유용하다

## What Must Be Saved

### Minimum Output For Trajectory

- `summed` state:
  - top heads를 합친 residual vector
- shape:
  - `(trial, step, dim)`
- 이 버전만 있어도:
  - stepwise common PCA
  - stepwise PCA GIF
  는 가능하다

### Required Output For Multi-Feature Plan

- `headwise` state:
  - top heads를 합치지 않은 per-head residual contribution
- shape:
  - `(trial, step, head, dim)`
- 이유:
  - 지금 당장 `Δc_t`는 summed로도 가능하지만
  - 이후 finer-grained feature / head-block analysis를 위해 raw head structure를 남겨야 한다

### Save Both

- v1 저장 원칙:
  - `summed`와 `headwise`를 둘 다 저장한다
- 이유:
  - `summed`는 PCA trajectory와 q-wise reweighting에 바로 쓰인다
  - `headwise`는 future fine-grained mechanistic analysis를 위해 필요하다

## Matching Unit

- 기본 matching key:
  - `q_id`
  - `trial_id`
  - `slot_name`
  - `ref`
- condition match:
  - `AAA ↔ BABA`
  - `AAA ↔ DADA`
- 즉 stepwise state extraction 결과는 나중에 trial-wise matched delta를 만들 수 있게 저장되어야 한다

## Output Artifacts

### Per-q Extraction Files

- `stepwise_a_states_AAA_ref.npz`
- `stepwise_a_states_union_ref.npz`
- `stepwise_a_states_meta.json`

### Suggested NPZ Keys

- `Q1__AAA__sum`
- `Q1__AAA__headwise`
- `Q1__BABA__sum`
- `Q1__BABA__headwise`
- `Q1__DADA__sum`
- `Q1__DADA__headwise`

### Suggested Shapes

- summed:
  - `(n_trials, 5, resid_dim)`
- headwise:
  - `(n_trials, 5, n_heads_selected, resid_dim)`

### Metadata Must Include

- `q_id`
- `ref`
- `slot_names`
  - `["A_demo_1", "A_demo_2", "A_demo_3", "A_demo_4", "A_query"]`
- `aaa_matched_slot_names`
  - `["A_demo_2", "A_demo_4", "A_demo_6", "A_demo_8", "A_query"]`
- `head_pool_source`
- `head_pool_size`
- `head_list`
- `trial_ids_by_condition`
- `slot_token_indices_by_condition`
- `available_conditions`
- `missing_artifacts`

## What This Enables Immediately

### 1. Stepwise PCA Trajectory

- pooled matrix:
  - all `AAA`, `BABA`, `DADA` summed states across trials and steps
- then:
  - fit one common PCA
  - project all points to the same PCA space
- this supports:
  - static stepwise PCA plots
  - animated PCA GIF
  - centroid trajectory overlays

### 2. Stepwise Matched Deltas

- from saved states we can compute:
  - `Δ_t^{BAB} = v_t^{BAB} - v_t^A`
  - `Δ_t^{DAD} = v_t^{DAD} - v_t^A`
- this is the minimum object needed for:
  - stepwise feature reweighting
  - inside/outside A-space decomposition
  - update-bundle analysis

### 3. Future `G_A` / `Δc_t` Analysis

- with extracted states we can build:
  - `matched_5slot_basis`
  - `all_10slot_basis`
- and then compute:
  - `c_t^A`
  - `c_t^{BAB}`
  - `c_t^{DAD}`
  - `Δc_t`
  - retained-A metrics
  - `U_B/U_D` bundle metrics

## What This Does Not Yet Do

- this extraction stage does **not** itself compute:
  - `G_A`
  - `Δc_t`
  - retained-A
  - bundle dominance
  - participation ratio
- it only creates the stored state layer needed for those analyses

## Relation To Shot Trajectory

- stepwise trajectory:
  - x-axis = `prompt-internal A position`
  - shows within-prompt accumulation
- shot trajectory:
  - x-axis = `number of demos / shots`
  - compares different prompts
- this spec supports **stepwise trajectory now**
- shot-sweep trajectory would require additional extraction over multiple shot settings
- but it would reuse the same design:
  - fixed head pool
  - saved states
  - common PCA downstream

## Implementation Order

1. implement `AAA_ref` stepwise extraction on `Q1`
2. implement `union_ref` stepwise extraction on `Q1`
3. save both `summed` and `headwise`
4. verify slot-token alignment manually on a few trials
5. build a simple stepwise common PCA plot/GIF from `AAA/BABA/DADA`
6. then move to `G_A`, `Δc_t`, and reweighting metrics

## Test Plan

- verify `AAA`, `BABA`, `DADA` each produce 5 aligned A slots
- verify `AAA_ref` and `union_ref` both save outputs when available
- verify trial ordering is deterministic and metadata agrees with arrays
- verify `summed.shape == (n_trials, 5, dim)`
- verify `headwise.shape == (n_trials, 5, n_heads, dim)`
- verify a common PCA over saved summed states can be fit without re-running model forward
- verify saved arrays are sufficient to compute matched `Δ_t` offline

## Core Claim Of This Spec

- 이 extraction은 **PCA trajectory용 임시 산출물**이 아니다.
- 이 extraction은 **multi-feature reweighting plan의 입력층**이다.
- 즉 지금 저장하는 state는 동시에 두 역할을 한다.
  - trajectory visualization
  - feature reweighting / bundle analysis

