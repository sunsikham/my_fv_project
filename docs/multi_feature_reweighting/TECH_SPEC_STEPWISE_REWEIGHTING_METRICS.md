# Tech Spec: Stepwise Reweighting Metrics

## Summary

- 이 문서는 `stepwise A-state extraction` 이후에 계산할 **stepwise reweighting 분석 지표**를 정의한다.
- 목적은 `BABABA`, `DADADA`가 `AAAAAA` baseline 대비 어떻게 바뀌는지를 **과정 단위(step-wise)** 로 측정하는 것이다.
- v1 primary reference는 `AAA_ref`다.
- `union_ref`는 main scope에서 제외하고 sensitivity / supplementary로만 다룬다.
- 분석은 세 층으로 나뉜다.
  - `A-basis reweighting`
  - `A-only space inside/outside decomposition`
  - `B/D bundle dominance`

## Scope

- 입력은 이미 저장된 stepwise state artifact다.
  - `stepwise_a_states_AAA_ref.npz`
  - `stepwise_a_states_meta.json`
- 우선 대상 step은 matched 5 slots다.
  - `A_demo_1`
  - `A_demo_2`
  - `A_demo_3`
  - `A_demo_4`
  - `A_query`
- 우선 대상 condition은
  - `AAA`
  - `BABA`
  - `DADA`

## Primary Reference

- v1 stepwise reweighting main analysis는 `AAA_ref`를 사용한다.
- 이유:
  - `AAA_ref`가 A-only baseline 해석에 가장 직접적이다
  - current `union_ref`는 이전 3-condition union 정의의 영향을 받아 해석이 덜 안정적이다
  - stepwise multi-feature 주장의 main evidence는 `AAA_ref`만으로 충분히 만들 수 있다
- 따라서 이 문서의 coefficient, retention, bundle metric 정의는 기본적으로 `AAA_ref` summed state를 기준으로 한다.

## Inputs

### Required State Objects

- `v_{i,t}^A`
  - `AAAAAA`에서 trial `i`, step `t`의 A state
- `v_{i,t}^{BAB}`
  - `BABABA`에서 trial `i`, step `t`의 A state
- `v_{i,t}^{DAD}`
  - `DADADA`에서 trial `i`, step `t`의 A state

### Matching Rule

- 같은 `q_id`
- 같은 `trial_id`
- 같은 `slot_name`
- 같은 `ref`

즉 stepwise 분석은 항상 matched baseline 방식으로 진행한다.

## Primary Basis: `G_A`

- `G_A`는 `AAAAAA`의 A-only states로 만든다.
- 두 버전을 같이 지원한다.

### 1. `matched_5slot_basis`

- `AAAAAA`의 matched 5 slots만 사용
  - physical positions: `2,4,6,8,query`

### 2. `all_10slot_basis`

- `AAAAAA`의 전체 A states 사용
  - `demo 1..9 + query`

### Common Basis Construction

- centered matrix:

`Z_A = V_A - μ_A`

- PCA/SVD로 basis를 만든다.
- 저장:
  - `μ_A`
  - `G_A_topk`
  - `G_A_full`

## Layer 1: A-Basis Reweighting

- 이 층이 메인 분석이다.
- 각 step `t`에서 아래 coefficient를 계산한다.

`c_t^A = G_A^+ (v_t^A - μ_A)`

`c_t^{BAB} = G_A^+ (v_t^{BAB} - μ_A)`

`c_t^{DAD} = G_A^+ (v_t^{DAD} - μ_A)`

- 그리고 drift를 계산한다.

`Δc_t^{BAB} = c_t^{BAB} - c_t^A`

`Δc_t^{DAD} = c_t^{DAD} - c_t^A`

### Core Metrics

#### 1. Feature-wise coefficient drift

- `Δc_{t,k}^{BAB}`
- `Δc_{t,k}^{DAD}`

해석:
- 어떤 A-feature가 step 따라 커지는가
- 어떤 것은 유지되는가
- 어떤 것은 줄어드는가

#### 2. `active_count_deltac(t)`

- threshold:

`τ_k = 0.1 * std(c_k^A over AAA baseline states)`

- count:

`active_count_deltac(t) = #{k : |Δc_{t,k}| > τ_k}`

해석:
- 몇 개 feature가 의미 있게 바뀌는가

#### 3. `PR_deltac(t)`

`PR_deltac(t) = (Σ_k Δc_{t,k}^2)^2 / Σ_k Δc_{t,k}^4`

해석:
- `PR ≈ 1`: one-axis-like collapse
- `PR > 1`: 여러 feature에 걸친 reweighting

### What This Layer Tests

- 네 가설의 핵심:
  - A가 통째로 교체되는 것이 아니라
  - 원래 A basis 위의 여러 feature coefficient가 step 따라 체계적으로 바뀌는가

## Layer 2: A-Only Space Inside/Outside Decomposition

- 이 층은 “원래 A feature가 얼마나 남는가”를 본다.

### State-level retention

`R_t^{BAB} = ||P_A (v_t^{BAB} - μ_A)||^2 / ||v_t^{BAB} - μ_A||^2`

`R_t^{DAD} = ||P_A (v_t^{DAD} - μ_A)||^2 / ||v_t^{DAD} - μ_A||^2`

해석:
- mixed A state가 A-only subspace 안에 얼마나 남아 있나

### Change-level decomposition

`Δ_t^{BAB} = v_t^{BAB} - v_t^A`

`Δ_t^{DAD} = v_t^{DAD} - v_t^A`

`inside_change_frac_t^{BAB} = ||P_A Δ_t^{BAB}||^2 / ||Δ_t^{BAB}||^2`

`outside_change_frac_t^{BAB} = ||(I - P_A) Δ_t^{BAB}||^2 / ||Δ_t^{BAB}||^2`

`inside_change_frac_t^{DAD} = ||P_A Δ_t^{DAD}||^2 / ||Δ_t^{DAD}||^2`

`outside_change_frac_t^{DAD} = ||(I - P_A) Δ_t^{DAD}||^2 / ||Δ_t^{DAD}||^2`

해석:
- 변화량 자체가 A-only basis 안의 재가중으로 얼마나 설명되나
- 변화량 중 A-only basis 밖 새 성분은 얼마나 되나

### What This Layer Tests

- 변화가 주로 `A 내부 재가중`인지
- 아니면 `A-only basis 밖의 새 방향 유입`이 큰지

## Layer 3: B/D Bundle Dominance

- 이 층은 보조지만 중요하다.
- step이 갈수록 intended B/D dominance가 실제로 커지는지 본다.

## Update Bundles

- `U_B`
  - B-like change bundle
- `U_D`
  - D-like change bundle

## Recommended Bundle Construction

- v1에서는 `U_B/U_D`를 **endpoint-anchored residual bundle** 방식으로 만든다.
- 목적:
  - bundle의 첫 축을 clean endpoint 의미에 고정하고
  - 추가 축만 stepwise residual 구조에서 학습한다

### `U_B` Construction

#### Step 1. Clean main axis

`u_B,1 = normalize(b - a)`

여기서
- `a = mean(A_query of AAAAAA)`
- `b = mean(A_query of BBBBBB)`

즉 `u_B,1`은 clean B endpoint axis다.

#### Step 2. Stepwise BAB residuals

각 matched trial/step에서

`Δ_t^{BAB} = v_t^{BAB} - v_t^A`

를 만든 뒤, main-axis 성분을 제거한 residual을 만든다.

`r_t^{BAB} = Δ_t^{BAB} - <Δ_t^{BAB}, u_B,1> u_B,1`

그리고 이 residual들을 trial × step 전체에서 모은다.

#### Step 3. Residual PCA

- residual matrix에 PCA/SVD를 적용해 반복적으로 나타나는 추가 축을 뽑는다.
- 예:
  - `u_B,2`
  - `u_B,3`

#### Final bundle

`U_B = [u_B,1, u_B,2, u_B,3]`

의미:
- `u_B,1`: clean B main axis
- `u_B,2`, `u_B,3`: stepwise BAB change에서 반복되는 추가 변화 축

### `U_D` Construction

- `U_D`도 같은 방식으로 만든다.

#### Step 1. Clean main axis

`u_D,1 = normalize(d - a)`

여기서
- `a = mean(A_query of AAAAAA)`
- `d = mean(A_query of DDDDDD)`

#### Step 2. Stepwise DAD residuals

`Δ_t^{DAD} = v_t^{DAD} - v_t^A`

`r_t^{DAD} = Δ_t^{DAD} - <Δ_t^{DAD}, u_D,1> u_D,1`

#### Step 3. Residual PCA

- residual PCA/SVD로
  - `u_D,2`
  - `u_D,3`
를 얻는다.

#### Final bundle

`U_D = [u_D,1, u_D,2, u_D,3]`

### Why This Construction Is Preferred

- 첫 축이 clean endpoint axis라서 해석이 명확하다
- 추가 축들은 실제 stepwise 변화의 structured residual을 담는다
- pure PCA-only bundle보다 해석이 좋다
- final-query clean-contrast only bundle보다 stepwise process를 더 잘 반영한다

- coefficients:

`alpha_t^B = U_B^T Δ_t`

`alpha_t^D = U_D^T Δ_t`

여기서 `Δ_t`는 branch에 따라
- `Δ_t^{BAB}`
- `Δ_t^{DAD}`
를 쓴다.

### Core Metrics

#### 1. `F_B(t)`

`F_B(t) = Σ_k alpha_{t,k}^{B 2}`

해석:
- B-bundle energy

#### 2. `F_D(t)`

`F_D(t) = Σ_k alpha_{t,k}^{D 2}`

해석:
- D-bundle energy

#### 3. `M(t)`

`M(t) = F_B(t) - F_D(t)`

branch-specific interpretation:
- `BAB` branch:
  - `M_BAB(t) > 0`이면 intended B dominance
- `DAD` branch:
  - `M_DAD(t) = F_D(t) - F_B(t)`로 대칭 정의 가능

해석:
- intended bundle이 competing bundle보다 우세한가

#### 4. `PR_bundle(t)`

`PR_bundle(t) = (Σ_k alpha_{t,k}^2)^2 / Σ_k alpha_{t,k}^4`

해석:
- bundle 내부에서도 one-axis collapse인지
- 여러 축이 함께 사는지

### What This Layer Tests

- step이 갈수록 intended B/D dominance가 실제로 강화되는가
- 동시에 bundle이 단일 축으로 collapse하지 않는가

## Recommended Minimum Stepwise Metrics

- 실제 분석에서 가장 먼저 볼 최소 세트는 이 다섯 개다.

1. `mean Δc_t`
2. `PR_deltac(t)`
3. `R_t`
4. `inside_change_frac_t`
5. `M(t)`

## Expected Signature If The Hypothesis Is Correct

- step이 갈수록
  - 일부 `Δc_{t,k}`는 증가
  - `PR_deltac(t)`는 1로 collapse하지 않음
  - `R_t`는 너무 낮지 않음
  - `inside_change_frac_t`가 충분히 큼
  - `M(t)`는 intended branch에서 증가

즉,

> 원래 A 구조는 꽤 남아 있으면서, 여러 feature가 같이 재가중되고, 그 총합이 점점 B/D 쪽 우세를 만든다

라는 패턴이다.

## Output Schema

### Trial-wise arrays

- `c_A_trials`
- `c_BAB_trials`
- `c_DAD_trials`
- `delta_c_BAB_trials`
- `delta_c_DAD_trials`
- `R_BAB_trials`
- `R_DAD_trials`
- `inside_change_frac_BAB_trials`
- `inside_change_frac_DAD_trials`
- `alpha_BAB_B_trials`
- `alpha_BAB_D_trials`
- `alpha_DAD_B_trials`
- `alpha_DAD_D_trials`

### Stepwise summary CSV

- one row per `q_id x ref x slot_name`
- include:
  - `R_t`
  - `inside/outside_change`
  - `active_count_deltac`
  - `PR_deltac`
  - `F_B`
  - `F_D`
  - `M`
  - `PR_bundle`

## Implementation Order

1. load stepwise state NPZ
2. build `G_A` from `AAA` stepwise states
3. compute `c_t` and `Δc_t`
4. compute `R_t` and `inside/outside change`
5. build endpoint-anchored residual `U_B`, `U_D`
6. compute bundle metrics
7. save trial-wise arrays and stepwise summary CSV

## Bottom Line

- stepwise에서 봐야 할 것은 단순히 “B 쪽으로 간다”가 아니다.
- 핵심은 세 층이다.
  - **A-basis coefficient drift**
  - **A-only subspace retention / inside-outside change**
  - **B/D bundle dominance**
- 이 세 층이 together 해야
  - one-axis replacement
  와
  - multi-feature reweighting
  을 구분할 수 있다.
