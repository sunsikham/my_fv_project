# Tech Spec: A-Basis Reweighting Analysis With Future Stepwise Expansion

## Summary

- 구현 목표는 `ABABAB`와 `DADADA`를 **one-axis movement**가 아니라, **matched A-only baseline 대비 feature reweighting**으로 분석하는 것이다.
- v1은 **final-query-only**로 간다. 현재 저장된 벡터에서 바로 가능한 범위다.
- `retained-A`는 두 단계로 나눈다.
  - `v1`: `K_A=5`의 exploratory top-K A-basis
  - `v1.1` or main run: `K_A_eff = rank(Z_A)`까지 사용한 full A-only subspace
- 동시에 저장/코드 구조는 **stepwise (`A_1, A_2, A_3, A_query`) 확장 가능**하게 설계한다.
- 분석은 두 reference 공간에서 모두 수행한다.
  - `AAA_ref`
  - `union_ref`

## Key Changes

- 새 분석 엔트리포인트를 추가한다.
  - 예: `scripts/compute_reweighting_qwise.py`
- 입력은 기존 per-q vector artifacts를 재사용한다.
  - `trial_vectors_<ref>_AAA.npy`
  - `trial_vectors_<ref>_BBB.npy`
  - `trial_vectors_<ref>_DDD.npy`
  - `trial_vectors_<ref>_BABA.npy`
  - `trial_vectors_<ref>_DADA.npy`
  - `vector_extraction_meta.json`
- v1의 분석 단위는 `(q_id, ref, slot_name)`이다.
  - 기본 `slot_name = A_query`
  - stepwise 확장 시 같은 스키마에 `A_1, A_2, A_3`를 추가한다
- `q_id`는 v1에서 **family-level analysis unit**으로 취급한다.
  - 즉 `G_A`도 `q_id x ref x slot`마다 학습한다
  - 이후 q가 너무 좁다고 판단되면 family grouping을 별도 확장으로 둔다

## A-Only Basis (`G_A`)

- `G_A`는 `AAAAAA`의 A-only states만으로 만든다.
- 입력 행렬:
  - rows = `AAA` trial vectors for the chosen `slot_name`
  - columns = residual-space dimensions
- centering 규칙을 명시적으로 고정한다.
  - `μ_A = mean(V_AAA)`
  - centered matrix `Z_A = V_AAA - μ_A`
- basis 추출은 PCA/SVD로 한다.
  - `Z_A = U Σ V^T`
  - `G_A`는 top `K_A_eff` right singular vectors
  - columns of `G_A` are orthonormal
- effective rank fallback:
  - `K_A_eff = min(K_A, rank(Z_A), n_AAA_trials - 1)`
  - 기본 `K_A = 5`
- projector 정의:
  - because `G_A` is orthonormal, `P_A = G_A G_A^T`
- component sign stabilization:
  - 각 component sign은 `AAA -> BBB` mean contrast와의 내적이 양수가 되도록 정한다
  - 이를 `a_basis_meta.json`에 기록한다

## Matched Baseline And Coefficients

- 매칭 규칙은 고정한다.
  - same `q_id`
  - same `trial_id`
  - same `slot_name`
  - same `ref`
- branch-wise matching으로 분리한다.
  - `BAB` branch는 `AAA / BBB / BABA`의 공통 `trial_id`만 사용
  - `DAD` branch는 `AAA / DDD / DADA`의 공통 `trial_id`만 사용
  - five-condition global intersection은 사용하지 않는다
  - 이유: D-side missingness 때문에 BAB analysis까지 막히지 않게 하기 위함
- coefficient 정의:
  - `c_t^A = G_A^T (v_t^A - μ_A)`
  - `c_t^BAB = G_A^T (v_t^BAB - μ_A)`
  - `c_t^DAD = G_A^T (v_t^DAD - μ_A)`
- 핵심 drift:
  - `Δc_t^BAB = c_t^BAB - c_t^A`
  - `Δc_t^DAD = c_t^DAD - c_t^A`
- 저장은 trial-wise와 q-wise 둘 다 한다.
  - trial-wise arrays in NPZ
  - q-wise means in CSV

## Change Decomposition And Diagnostics

- state-level retained energy와 change-level decomposition을 둘 다 계산한다.

### State-level

- `v1` exploratory metric:
  - `R_BAB_topk = ||P_A^topk (v^BAB - μ_A)||^2 / ||v^BAB - μ_A||^2`
  - `R_DAD_topk = ||P_A^topk (v^DAD - μ_A)||^2 / ||v^DAD - μ_A||^2`
  - 여기서 `P_A^topk`는 top `K_A=5` basis projector
- `v1.1` / main run metric:
  - `R_BAB_full = ||P_A^full (v^BAB - μ_A)||^2 / ||v^BAB - μ_A||^2`
  - `R_DAD_full = ||P_A^full (v^DAD - μ_A)||^2 / ||v^DAD - μ_A||^2`
  - 여기서 `P_A^full`은 `K_A_eff = rank(Z_A)`까지 쓴 full A-only subspace projector
- 해석:
  - `R_topk`는 **retained top-K A-basis energy**라고 부른다
  - `R_full`은 **full A-only subspace retention**으로 해석한다
  - 발표/주요 해석은 `R_full`을 우선 사용하고, `R_topk`는 exploratory 보조지표로 둔다

### Change-level

- `Δ^BAB = v^BAB - v^A`
- `Δ^DAD = v^DAD - v^A`
- `inside_change_frac_BAB = ||P_A Δ^BAB||^2 / ||Δ^BAB||^2`
- `outside_change_frac_BAB = ||(I - P_A) Δ^BAB||^2 / ||Δ^BAB||^2`
- `inside_change_frac_DAD = ||P_A Δ^DAD||^2 / ||Δ^DAD||^2`
- `outside_change_frac_DAD = ||(I - P_A) Δ^DAD||^2 / ||Δ^DAD||^2`
- 이것이 문서에서 말한 “원래 A feature 재가중 vs A-only basis 밖 새 성분”을 직접 보여준다

### Δc-space diagnostics

- feature-wise drift:
  - `Δc_{t,k}^BAB`
  - `Δc_{t,k}^DAD`
- active-count threshold:
  - `τ_k = 0.1 * std(c_k^A over AAA trials)` for that `q_id, ref, slot`
- metrics:
  - `active_count_deltac_BAB`
  - `active_count_deltac_DAD`
  - `PR_deltac_BAB`
  - `PR_deltac_DAD`
- interpretation:
  - low active-count / `PR≈1` = one-axis-like collapse
  - larger active-count / `PR>1` = distributed reweighting

## Update Bundles (`U_B`, `U_D`)

- `U_B` and `U_D` are secondary update bases, not replacements for `G_A`.
- build inputs:
  - `Δ_B,clean = v_BBB - v_AAA`
  - `Δ_D,clean = v_DDD - v_AAA`
- fitting scope:
  - default `pooled_all_q` across eligible q and matched trials, within each `(ref, slot_name)`
  - metadata must record `bundle_fit_scope = pooled_all_q`
  - no leave-one-q-out in v1
- basis extraction:
  - PCA/SVD on pooled delta matrix
  - `K_B_eff = min(K_B, rank(Z_B), n_clean_B - 1)`
  - `K_D_eff = min(K_D, rank(Z_D), n_clean_D - 1)`
  - defaults `K_B = 3`, `K_D = 3`
- coefficients:
  - `alpha_BAB_B = U_B^T Δ^BAB`
  - `alpha_BAB_D = U_D^T Δ^BAB`
  - `alpha_DAD_B = U_B^T Δ^DAD`
  - `alpha_DAD_D = U_D^T Δ^DAD`

### Bundle-space metrics

- energies:
  - `F_B_BAB = Σ alpha_BAB_B^2`
  - `F_D_BAB = Σ alpha_BAB_D^2`
  - `F_B_DAD = Σ alpha_DAD_B^2`
  - `F_D_DAD = Σ alpha_DAD_D^2`
- dominance:
  - `M_BAB = F_B_BAB - F_D_BAB`
  - `M_DAD = F_D_DAD - F_B_DAD`
- diversity:
  - `active_count_bundle_BAB`
  - `active_count_bundle_DAD`
  - `PR_bundle_BAB`
  - `PR_bundle_DAD`
- threshold for active-count in bundle-space:
  - fixed `τ_bundle = 0.1 * sqrt(mean(alpha_clean^2))` per coefficient dimension
  - store this in metadata

## Output Artifacts

- `reweighting_manifest.csv`
  - `q_id, ref, slot_name, eligible, reason`
  - `n_trials_AAA, n_trials_BBB, n_trials_DDD, n_trials_BABA, n_trials_DADA`
  - `n_trials_BAB_intersection`
  - `n_trials_DAD_intersection`
  - `rank_A, rank_B, rank_D, K_A_eff, K_B_eff, K_D_eff`
- `reweighting_qwise_<ref>.csv`
  - one row per `q_id x slot_name`
  - includes:
    - `R_BAB_topk, R_DAD_topk`
    - `R_BAB_full, R_DAD_full`
    - `inside_change_frac_BAB, outside_change_frac_BAB`
    - `inside_change_frac_DAD, outside_change_frac_DAD`
    - `active_count_deltac_BAB, active_count_deltac_DAD`
    - `PR_deltac_BAB, PR_deltac_DAD`
    - `F_B_BAB, F_D_BAB, F_B_DAD, F_D_DAD`
    - `M_BAB, M_DAD`
    - `active_count_bundle_BAB, active_count_bundle_DAD`
    - `PR_bundle_BAB, PR_bundle_DAD`
- `reweighting_arrays_<ref>.npz`
  - `q__slot__mu_A`
  - `q__slot__G_A`
  - `q__slot__U_B`
  - `q__slot__U_D`
  - `q__slot__c_A_mean`
  - `q__slot__c_BAB_mean`
  - `q__slot__c_DAD_mean`
  - `q__slot__delta_c_BAB_mean`
  - `q__slot__delta_c_DAD_mean`
  - `q__slot__alpha_BAB_B_mean`
  - `q__slot__alpha_DAD_D_mean`
- current one-axis outputs remain untouched.
  - this analysis is additive, not a replacement

## Stepwise-Ready Extension

- v1 only computes `slot_name = A_query`
- but extraction and output schema must already support:
  - `A_1`
  - `A_2`
  - `A_3`
  - `A_query`
- to make that possible, add a new extraction spec now:
  - `extract_stepwise_a_states.py`
  - outputs `stepwise_a_states_<ref>.npz`
- in v1 this file may contain only `A_query`
- later we extend it without changing downstream schemas

## Test Plan

- Basis sanity
  - `G_A` reconstruction error on AAA trials is finite and not degenerate
  - component signs are deterministic across reruns
- Matching sanity
  - `AAA/BBB/BABA` align by `trial_id` for the BAB branch
  - `AAA/DDD/DADA` align by `trial_id` for the DAD branch
  - branch-specific dropped trials are logged in manifest
- Change decomposition sanity
  - `inside_change_frac + outside_change_frac ≈ 1`
  - `R_BAB_topk, R_DAD_topk, R_BAB_full, R_DAD_full` are in `[0,1]`
- Δc metrics
  - `Δc ≈ 0` if comparing `AAA` vs matched `AAA`
  - synthetic one-axis perturbation gives `PR_deltac ≈ 1`
  - synthetic multi-axis perturbation gives `PR_deltac > 1`
- Bundle metrics
  - `M_BAB > 0` when synthetic B-delta dominates
  - `M_DAD > 0` when synthetic D-delta dominates
- Dual-ref comparison
  - `AAA_ref` and `union_ref` outputs both generate successfully
  - q-wise directions are comparable and not obviously contradictory
- Real-data acceptance
  - with current stored vectors, at least `Q1,Q8,Q9,Q10,Q11` run end-to-end for `A_query`

## Assumptions

- v1 is final-query-only because stepwise A states are not yet stored.
- `q_id` is treated as family-level in v1.
- `AAA_ref` and `union_ref` are both worth computing; neither is assumed globally “correct.”
- `G_A` is the primary analysis basis; `U_B/U_D` are secondary update bases.
- summed trial vectors are sufficient for v1 q-wise reweighting, but finer headwise analyses are deferred until stepwise/headwise extraction exists.
- `v1` uses `K_A=5` for exploratory top-K A-basis analysis.
- `v1.1` or main run additionally computes the full A-only subspace version with `K_A_eff = rank(Z_A)`.
- BAB and DAD matching are intentionally separated; missing D-side trials must not block BAB-side analysis.
