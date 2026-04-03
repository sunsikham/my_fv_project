# Why the Residual Matters

This note explains why, in the current analysis, the residual should not be treated as a trivial error term or as mere noise.

## 1. What the residual is

In Stage 1, the first question is whether the mixed state moves toward the intended endpoint.

- `BABABA` is evaluated relative to `A -> B`
- `DADADA` is evaluated relative to `A -> D`

For `BABABA`, the total change is

\[
x_{bab} - a
\]

and the first quantity we compute is progress:

\[
p_B(x_{bab}) = \frac{(x_{bab}-a)^\top (b-a)}{\|b-a\|^2}
\]

This tells us how far the mixed state has moved along the `A -> B` axis.

The residual then measures how much of the total change is left over after removing that clean `A -> B` component:

\[
r_B(x_{bab})
=
\frac{\|(x_{bab}-a) - p_B(x_{bab})(b-a)\|}{\|b-a\|}
\]

So the residual is the off-axis part of the movement: the component of change that is not explained by simple movement along the intended endpoint direction.

## 2. What it would mean if the residual were small

If context simply replaced `A` with `B` or `D`, then a clean one-axis shift would be the most direct geometric explanation.

In that case:

- progress would be large
- residual would be small

This would support a simple interpolation-style picture:

> the state moves from `A` toward the intended endpoint mainly along a single direction.

So if the residual were very small, a one-axis substitution account would remain quite plausible.

## 3. But in our case the residual is not small

For Q1:

- `BABABA -> B progress = 0.528`
- `DADADA -> D progress = 0.509`

So the intended endpoint movement is clearly real.

However, the residuals are also non-trivial:

- `r_B = 0.343`
- `r_D = 0.301`

These values are not close to zero.

So the result is not:

> the state simply slides along a clean `A -> B` or `A -> D` line.

Instead, the state moves toward the intended endpoint, but a substantial amount of structured change remains outside that one-axis account.

## 4. Why the residual is difficult to dismiss as noise

To dismiss the residual as noise, we would want it to be:

- very small
- unstable across conditions
- unrelated to the endpoint-directed movement

But that is not what we see.

1. The endpoint progress is clearly present.  
   So the movement is not random.

2. The residual is still substantial.  
   So the total change is not exhausted by a one-axis explanation.

3. Later analyses show structured inside-A feature reweighting.  
   That makes the residual look less like random leftover variation and more like a trace of internal restructuring that a one-axis summary cannot capture.

So the residual is better interpreted as:

> structured change that is not captured by a simple endpoint-axis description.

## 5. Why the residual directly motivates the current hypothesis

The current working hypothesis is:

> context does not simply replace `A` with `B` or `D`;  
> instead, it reweights multiple features already present inside `A`.

If this hypothesis is right, then the movement should not be expected to collapse onto a single straight line.

A more natural picture is:

- `A` already contains multiple latent facets
- context changes the weights of those facets
- the resulting state becomes more B-like or D-like

Under that view, the expected pattern is:

- real endpoint progress
- but also a meaningful residual

because multi-feature reweighting will generally not compress into a single axis.

So the residual can be read as:

> the measurable trace of internal redistribution within `A`, rather than evidence of failure.

## 6. Why a straight-line account would be simpler, but is not sufficient

If the model were solving the problem through a simple one-axis substitution, then a straight-line movement toward the intended endpoint would be the simplest geometric account.

That would be the cleaner and more direct explanation:

- move from `A`
- toward `B` or `D`
- mostly along one axis

But this is not what the data look like.

The model does move toward the intended endpoint, but the residual remains substantial.

That is exactly why the residual matters: it tells us that a simple straight-line substitution account is not enough.

This is what motivates the current hypothesis that the model preserves much of the original `A` structure while redistributing emphasis across multiple features inside it.

## 7. What the joint residual adds

There is also a second issue with the raw cross-score.

The `A -> B` and `A -> D` directions are not perfectly orthogonal, so a mainly B-directed movement can still produce some positive projection onto the D axis.

That means raw separate projection can mix together:

- true cross-component
- geometric overlap between the axes

To address this, we use a joint decomposition:

\[
x-a \approx \alpha u_B + \beta u_D + \varepsilon
\]

Now the residual is what remains after explaining the change with the B and D components simultaneously.

So the joint residual is an even stronger statement:

> even after accounting for both intended axes together, there is still meaningful structure left over.

This makes it harder to explain the result as a mere projection artifact.

## 8. Current interpretation

The current picture is:

1. the mixed state really does move toward the intended endpoint
2. but the movement is not a clean one-axis straight shift
3. much of the change remains inside A-space
4. therefore, the residual is better understood as evidence of multi-feature reweighting than as meaningless leftover noise

So the residual matters because it tells us that the model is not simply replacing one state with another.

Instead, it is moving toward the endpoint while preserving and reorganizing structure already present in `A`.

## 9. One-sentence summary

The residual matters because it is not just leftover error: it is a measurable sign that endpoint movement is being produced through structured reweighting inside `A`, rather than through simple one-axis substitution.
