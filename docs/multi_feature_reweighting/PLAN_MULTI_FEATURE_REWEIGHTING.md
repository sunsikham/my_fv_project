# Multi-Feature Reweighting Plan

## Core idea

In `ABABAB`, the representation does not collapse to a single B-only feature.  
Instead, multiple original A-related features remain alive, while the total weight of the B-related feature bundle grows and makes the whole state look more B-like.

So the target claim is:

> Relative to the matched `AAAAAA` baseline, `ABABAB` increasingly projects onto a B-related feature bundle, while the change remains distributed across multiple feature axes rather than collapsing to a single axis.

---

## State Basis vs Update Basis

The key distinction is between:

- `g_k`: feature axes that describe the original A-only state
- `u_B,j`: feature axes that describe how B context changes A

So the hypothesis is not that `U_B` replaces A.  
Instead, the A-only state itself can be written as

`v_t^A = sum_k c_{t,k}^A g_k`

and the B-context state as

`v_t^BAB = sum_k c_{t,k}^{BAB} g_k + eta_t`

Interpretation:

- some coefficients `c_{t,k}` increase
- some stay roughly similar
- some decrease
- and `eta_t` can capture any new component outside the A-only basis

This means:

- `g_k` = the original A-only feature basis (**state basis**)
- `u_B,j` = directions of B-induced change (**update basis**)

So they are not competing explanations.  
They describe different levels:

- state basis: what A already is
- update basis: how B context pushes A away from its baseline

---

## Stronger Statement Of The Hypothesis

The intended hypothesis is not that `ABABAB` replaces A with a single B-only feature.

Instead:

> A in `AAAAAA` already contains multiple features.  
> When B context is added, those features do not all disappear.  
> Rather, a subset of B-related features grows in weight, while other original A-features remain present.

This can be written as:

`v_t^A = sum_k c_{t,k}^A g_k`

and

`v_t^BAB = sum_k c_{t,k}^{BAB} g_k + eta_t`

Interpretation:

- `g_k` = A-only feature axes
- `c_{t,k}^A` = how much A carries feature `k` at step `t`
- `c_{t,k}^{BAB}` = how much the mixed state carries the same A-feature at step `t`
- `eta_t` = any new component outside the A-only basis

So the real claim is:

> `ABABAB` becomes more B-like because the B-related subset of the original A-feature bundle is reweighted upward, not because all non-B structure vanishes.

---

## Build The A-Only Basis First

To really test the hypothesis, the analysis should start by building an A-only basis:

`G_A = [g_1, g_2, ..., g_K]`

This basis should be built from `AAAAAA` states only.

Conceptually:

- collect A-only states from the same A family
- ideally across multiple pairs and multiple steps
- extract the common axes of variation from those A-only states

This gives the feature basis that represents:

> the features A already has before B context is inserted

---

## Why This Comes Before `U_B`

`U_B` is still useful, but it is a different object.

- `G_A` describes what A already is
- `U_B` describes how B context changes A

So:

- `G_A` = state basis
- `U_B` = update basis

They are not competing explanations.  
The intended workflow is:

1. learn `G_A` from `AAAAAA`
2. express `BABABA` in `G_A`
3. study coefficient drift inside `G_A`
4. only then ask whether the remaining change aligns with `U_B`

In other words, `U_B` should be treated as a secondary analysis of the change vector, not as a replacement for the original A-only feature basis.

---

## Project Mixed States Back Into The A-Only Basis

Once `G_A` is available, express the mixed-context state in that basis.

At each step:

- project the `BABABA` A-state into the A-only basis
- project the matched `AAAAAA` A-state into the same basis

For example:

`c_t^BAB = G_A^+ v_t^BAB`

`c_t^A = G_A^+ v_t^A`

where `G_A^+` is the pseudoinverse.

Interpretation:

- `c_t^BAB` = coefficient profile of the mixed A-state in the original A-only feature basis
- `c_t^A` = coefficient profile of the matched A-only baseline in that same basis

So this is no longer a raw-state comparison.  
It is a comparison in the original A-feature coordinates.

---

## Then Look At Coefficient Change

Now define:

`Δc_t = c_t^BAB - c_t^A`

This is the key quantity.

It directly answers:

> relative to the A-only trajectory, how much did B context change each A-feature coefficient?

Interpretation:

- some coefficients increase
- some stay similar
- some decrease

This is the most direct operationalization of the hypothesis:

> B context does not erase all original A-features and replace them with one B-only direction.  
> Instead, it reweights the original A-feature profile, making some features stronger while others remain present.

---

## What Pattern Would Support The Hypothesis

Suppose the A-only coefficient profile is:

`c_t^A = [0.8, 0.7, 0.5, 0.2, 0.1]`

and under B context it becomes:

`c_t^BAB = [1.2, 0.9, 0.45, 0.18, 0.08]`

Then:

`Δc_t = [0.4, 0.2, -0.05, -0.02, -0.02]`

Interpretation:

- feature 1 and 2 increase
- feature 3, 4, 5 do not vanish
- the total state becomes more B-like because a subset is amplified, not because the rest is erased

That is the concrete shape of multi-feature reweighting.

---

## Decompose Change Inside vs Outside The A-Only Basis

Once `Δ_t^BAB = v_t^BAB - v_t^A` is available, split it into:

`Δ_t^BAB = P_A Δ_t^BAB + (I - P_A) Δ_t^BAB`

Interpretation:

- `P_A Δ_t^BAB` = reweighting of existing A-only features
- `(I - P_A) Δ_t^BAB` = newly introduced change outside the A-only basis

This distinction is important.

- if most of the change lies inside `P_A`, B context mainly reweights what A already had
- if `(I-P_A)Δ` is large, B context is pulling the state into genuinely new directions

This is the stronger version of the hypothesis:

> B context does not move A to a completely unrelated point; it mostly reweights existing A-only features, while optionally adding a smaller extra component.

---

## Two Additional Diagnostics

### Retained-A energy

`R_t = ||P_A v_t^BAB||^2 / ||v_t^BAB||^2`

Interpretation:

- large `R_t` = much of the mixed state still lies in the original A-only feature subspace
- small `R_t` = the mixed state has moved substantially outside the A-only basis

### B-amplified coefficient growth

For each A-only feature:

`Δc_{t,k} = c_{t,k}^{BAB} - c_{t,k}^A`

Interpretation:

- positive and growing `Δc_{t,k}` = that original A-feature is being amplified by B context
- near-zero `Δc_{t,k}` = that feature is retained but not strongly reweighted
- negative `Δc_{t,k}` = that feature is being suppressed

Together, these tell us whether B context is strengthening a subset of A’s original features rather than replacing the whole representation.

---

## Revised Analysis Order

The more accurate order is:

1. Build the A-only basis `G_A` from `AAAAAA`
2. Project matched `AAAAAA` and `BABABA` A states into `G_A`
3. Compute coefficient drift `Δc_t`
4. Measure which original A-features increase, which remain, and which decrease
5. Measure retained-A energy `R_t`
6. Optionally analyze the leftover change with `U_B`

This order tests the reweighting claim more directly than going straight to a B-update bundle.

---

## 1. Matched baseline first

For each family and each step `t`:

- A-only prefix, query A state:
  - `v_t^A`
- `ABABAB` prefix, query A state:
  - `v_t^BAB`

Then define:

`Δ_t^BAB = v_t^BAB - v_t^A`

Meaning:

- `Δ_t^BAB` is the additional change caused by inserting B context relative to the matched A-only baseline.

This is essential, because otherwise A-only self-drift and B-insertion effects are mixed together.

---

## 2. Build multiple B-feature axes, not just one

The feature axes here are not named semantic features.  
They are a data-driven basis describing how B context changes A.

Use training-family contrasts like:

`Δ_B,clean^(i) = v_B-rich^(i) - v_A-only^(i)`

Collect many such delta vectors, then build a small B-bundle:

- `u_B,1` = B-main axis
- `u_B,2` = B-subfeature 1
- `u_B,3` = B-subfeature 2

Likewise, if needed, define:

- `u_D,1`, `u_D,2`, ...

for D-related change directions.

At the beginning, a 2D or 3D B-bundle is enough:

- B-main axis
- B-subfeature 1
- B-subfeature 2

---

## 3. Read coefficients at each step

Let:

`U_B = [u_B,1, u_B,2, u_B,3]`

Then at each step:

`α_t^B = U_B^T Δ_t^BAB`

Interpretation:

- `α_t^B` is the coefficient vector of the B-bundle at step `t`
- The norm `||α_t^B||` measures how strongly the state moves into the B-bundle
- The distribution across coordinates tells us whether the shift is concentrated on one axis or spread across multiple axes

What to inspect:

- Does the norm of the B-bundle coefficient vector increase across steps?
- Do the coefficients collapse to one axis, or remain distributed across multiple axes?

---

## 4. Quantify “multiple features stay alive”

### A. Active coefficient count

Choose a threshold `τ` and define:

`N_t = #{ k : |α_t,k| > τ }`

Interpretation:

- `N_t ≈ 1` means almost one-axis
- `N_t >= 2` means multiple features are simultaneously active

This is the simplest and most intuitive diagnostic.

### B. Participation ratio

Define:

`PR_t = (sum_k α_t,k^2)^2 / sum_k α_t,k^4`

Interpretation:

- `PR_t ≈ 1` means effectively one axis
- `PR_t > 1.5, 2, ...` means multiple axes contribute

So under the multi-feature hypothesis, even when the state becomes more B-like, `PR_t` should not collapse to 1.

This is the key numerical summary of:

> “The state becomes more B-like, but multiple features remain active.”

---

## 5. Define B dominance

Let B-bundle energy be:

`F_B(t) = sum_{k in B-bundle} α_t,k^2`

and D-bundle energy be:

`F_D(t) = sum_{k in D-bundle} α_t,k^2`

Then define a dominance score:

`M(t) = F_B(t) - F_D(t)`

Interpretation:

- `M(t) > 0` means the B-bundle dominates the D-bundle
- If `M(t)` increases with step, B dominance is strengthening

If at the same time `PR_t` remains comfortably above 1, that means:

- B is becoming dominant
- but the state still occupies a multi-dimensional feature profile

So the multi-feature signature is:

- `M(t)` increases
- `PR_t` does not collapse to 1

---

## 6. Operationalized hypothesis

For `ABABAB`, the target claim is:

> Relative to `AAAAAA`, the additional change `Δ_t^BAB` projects more strongly onto the B-related feature bundle as steps increase, while remaining distributed across multiple feature axes rather than collapsing to a single one.

In short:

- Dominance: B gets stronger
- Diversity: multiple features remain active

---

## 7. Difference from current metrics

Current metrics are mostly one-axis:

- progress
- selectivity
- leakage
- residual

These are useful, but they still ask whether the representation moves along one main direction.

The multi-feature view goes one level deeper:

- residual is not treated as generic noise
- instead, the unexplained part is tested for structured, repeatable sub-axes

So current results can show:

> “This is not just one-axis interpolation.”

The next step is to test:

> “Is the non-one-axis part structured multi-feature reweighting?”

---

## 8. Minimal realistic experiment

Start small:

### Step 1

Choose one A family.

### Step 2

From `AAAAAA` vs `BBBBBB` clean contrast, build a small B-bundle:

- one B-main axis
- one or two B-subfeature axes from residual structure

That gives a 2D or 3D B-bundle.

### Step 3

For each step in `ABABAB`, compute:

`Δ_t^BAB = v_t^BAB - v_t^A`

### Step 4

At each step compute:

- B-bundle energy `F_B(t)`
- participation ratio `PR_t`

Desired pattern:

- `F_B(t)` increases with step
- `PR_t` stays above 1, ideally around 2 or more

Then the claim becomes:

> “B context makes A more B-like, but the change is not a single straight-line drift; it is a reweighting of multiple B-related latent features.”

---

## Short version

The multi-feature hypothesis says:

> In `ABABAB`, moving toward B does not mean that only one B-feature turns on.  
> Instead, multiple features remain alive, while the total weight of the B-related feature bundle increases and makes the state more B-like.

To test this:

1. Use the matched `AAAAAA` baseline and form `Δ_t`
2. Build multiple B-bundle axes, not just one
3. Measure B-bundle strength `F_B(t)`
4. Measure feature diversity `PR_t`

That is the most direct way to test the claim.
