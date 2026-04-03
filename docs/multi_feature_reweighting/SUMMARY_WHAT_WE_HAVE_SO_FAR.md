# Summary: What We Have Computed So Far

## Core Thesis So Far

- 현재까지의 결과가 가장 강하게 지지하는 해석은 다음이다.

> A is not replaced by a single B/D axis.
> Instead, a substantial portion of the original A structure remains, while multiple features are reweighted stepwise so that the resulting state becomes increasingly aligned with the intended B/D endpoint.

- 한국어로는:

> A는 context를 받을 때 통째로 교체되지 않는다. 대신 A-only 구조를 상당 부분 유지한 채, 그 안의 여러 feature가 stepwise하게 재가중되고, 그 재가중의 signed contribution이 점점 B/D endpoint 쪽으로 정렬된다.

## Stage 1. Endpoint Movement

### What We Computed

- final query A only
- per q:
  - progress
  - cross-score
  - selectivity
  - residual
  - joint decomposition (`alpha`, `beta`, joint residual)

### What It Asked

- `BABABA`가 최종적으로 B 쪽으로 갔는가?
- `DADADA`가 최종적으로 D 쪽으로 갔는가?
- 그 이동이 A->B / A->D 한 직선으로 clean하게 설명되는가?

### Main Takeaway

- intended endpoint directionality는 분명하다.
- 그러나 residual이 남고 joint residual도 작지 않다.
- 즉 one-axis interpolation만으로는 충분하지 않다.

### Meaning

- context는 A를 intended B/D 쪽으로 밀지만,
- 그 변화는 단일 직선 이동 이상이다.

## Stage 2. Query-Only Reweighting

### What We Computed

- final query A only
- `AAAAAA`에서 `G_A` basis 생성
- `c^A`, `c^{BAB}`, `c^{DAD}`
- `Δc`
- `R_full`
- `inside/outside change`
- `PR_deltac`
- bundle dominance `M`

### What It Asked

- 최종 query에서 A가 통째로 교체되었는가?
- 아니면 A-only 구조를 상당 부분 유지한 채 multi-feature reweighting이 일어났는가?

### Main Takeaway

- mixed query state 안에 A-only 구조가 일부 남아 있다.
- 변화량의 큰 부분이 A-space 안 재가중으로 설명된다.
- 변화는 한 축이 아니라 여러 feature 축에 퍼져 있다.
- intended B/D bundle dominance는 존재한다.

### Meaning

- final query만 봐도 one-axis replacement보다는 multi-feature reweighting이 자연스럽다.

## Stage 3. Stepwise Reweighting

### What We Computed

- matched 5 slots:
  - `A_demo_1`
  - `A_demo_2`
  - `A_demo_3`
  - `A_demo_4`
  - `A_query`
- stepwise `G_A`
- stepwise `Δc_t`
- `R_t`
- `inside/outside`
- `M(t)`

### What It Asked

- multi-feature reweighting이 마지막에 갑자기 생기는가?
- 아니면 prompt 내부에서 stepwise하게 누적되는가?

### Main Takeaway

- B/D dominance는 첫 A부터 시작해 step이 갈수록 커진다.
- `PR_deltac(t)`가 높아서 one-axis collapse가 아니다.
- `R_t`와 `inside_change_frac_t`가 높아 A 구조가 stepwise 과정에서도 많이 남는다.

### Meaning

- 이 변화는 final-only effect가 아니라 stepwise accumulation이다.

## Stage 4. Endpoint-Aligned Contribution

### What We Computed

- `u_B^(A)`, `u_D^(A)`
- feature alignment:
  - `a_k^B`
  - `a_k^D`
- total signed alignment:
  - `T_B(t)`
  - `T_D(t)`
- feature-level signed contribution:
  - `contrib_k^B(t)`
  - `contrib_k^D(t)`
- `PR_contrib`

### What It Asked

- 여러 feature 변화가 실제로 B/D endpoint 방향과 연결되는가?
- 어떤 feature들이 그 방향 shift를 실제로 만들어내는가?

### Main Takeaway

- `BABABA`에서는 step이 갈수록 B endpoint alignment가 강화된다.
- `DADADA`에서는 step이 갈수록 D endpoint alignment가 강화된다.
- 이 shift는 한 feature collapse가 아니라 여러 feature의 signed contribution 누적이다.

### Meaning

- context effect는 단순 “여러 feature가 바뀐다”가 아니라,
- endpoint-aligned multi-feature reweighting으로 읽힌다.

## Stage 5. Inside/Outside Endpoint Decomposition

### What We Computed

- `Δ = Δ_in + Δ_out`
- `Δ_in = P_A Δ`
- `Δ_out = (I-P_A)Δ`
- inside/outside 각각에 대해:
  - norm
  - intended alignment
  - cross alignment
  - selectivity

### What It Asked

- inside-A와 outside-A 중 누가 endpoint-aligned shift를 실제로 더 설명하는가?

### Main Takeaway

- magnitude는 inside가 더 크다.
- Q1에서는 대략 inside가 75~80%, outside가 20~25% 정도의 bulk를 담당한다.
- 그런데 outside도 endpoint-selective하다.

### Meaning

- outside-A는 단순 noise라고 보기 어렵다.
- 하지만 bulk/main mechanism은 아니다.

## Stage 6. Inside/Outside Joint Decomposition

### What We Computed

- inside/outside 각각에 대해
  - `alpha_B`
  - `beta_D`
  - residual
를 joint decomposition으로 계산

### What It Asked

- inside/outside 각각에서 진짜 intended component와 cross component가 어떻게 생기는가?

### Main Takeaway

- inside가 intended B/D 성분의 bulk를 더 많이 가진다.
- outside도 intended direction에 selective하게 기여하지만 더 작다.
- inside = main carrier
- outside = smaller selective complement

### Meaning

- 현재 가장 안전한 해석은:

> inside-A structured reweighting is the main mechanism, and outside-A is a smaller but selective support component.

## Stage 7. Local Tangent / Curvature Check

### What We Computed

- `global`
- `step_local`
- `state_local`
세 방식으로 local tangent basis 비교
- inside/outside frac, norm, selectivity 비교

### What It Asked

- outside-A가 전역 linear basis artifact인가?
- 아니면 local tangent space로 가도 실제로 남는가?

### Main Takeaway

- local basis로 바꿔도 outside가 크게 사라지지 않았다.
- state-local에서는 outside가 오히려 더 크게 잡히기도 했다.
- local basis에서도 outside selectivity가 유지되었다.

### Meaning

- 현재 설정에선 outside-A를 단순 전역 PCA artifact로 보긴 어렵다.

## Current Best Interpretation

### Strongly Supported

- one-axis replacement는 아니다
- A structure는 상당 부분 남아 있다
- multi-feature reweighting이 stepwise하게 누적된다
- endpoint-aligned signed contribution이 실제로 커진다
- inside-A는 main carrier다
- outside-A는 smaller but selective complement다

### Not Yet Fully Proven

- outside-A가 true commitment / sharpening의 핵심 주역인지
- outside-A가 genuinely necessary한지
- outside-A 내부에도 distinct feature structure가 있는지
- behavior에서 inside와 outside의 역할이 정말 다르게 나타나는지

## What We Mean By “A Structure Remains”

- `A-only structure` means:
  - `AAAAAA` states define an A-basis `G_A`
  - mixed states are still substantially explainable in that basis
  - coefficient changes happen inside that basis

- 즉:
  - A가 사라진 것이 아니라
  - A-space 안의 feature weights가 재조정된다는 뜻이다.

## Current Working Hypothesis

- 현재까지 가장 잘 맞는 가설은:

> A already contains multiple relation facets.
> Context does not replace A wholesale.
> Instead, context reweights those facets inside the A-space, and adds a smaller but selective outside-A component that helps sharpen or stabilize the final B-like / D-like endpoint.

## Shepard vs. Tversky Interpretation

### What Is Shepard-Like In This Project

- `Shepard-like`하다는 말은
  - relation state들이 vector geometry 안에서
    - 위치
    - 방향
    - difference vector
  로 표현된다는 뜻이다.

- 현재 결과에서 Shepard-like한 부분은:
  - `a, b, d` anchor
  - `b-a`, `d-a` endpoint direction
  - progress / joint decomposition
  - stepwise endpoint alignment
  - local tangent / local geometry

- 즉:

> relation resolution의 바탕(substrate)에는 geometric structure가 있다

는 의미다.

### What Is Tversky-Like In This Project

- `Tversky-like`하다는 말은
  - relation judgment가 고정 metric distance 하나로 결정되지 않고
  - feature weighting / comparison rule에 따라 달라진다는 뜻이다.

- 현재 결과에서 Tversky-like한 부분은:
  - `Δc_t`
  - feature reweighting
  - endpoint-aligned signed contribution
  - inside/outside selective support

- 즉:

> context가 feature들의 가중치를 바꾸고, 그 weighted comparison이 B/D endpoint 쪽 판단을 만든다

는 의미다.

### Why They Fit Together

- 현재 결과는 Shepard와 Tversky 중 하나를 버리라는 뜻이 아니다.
- 가장 자연한 해석은:

> latent representation은 geometric하고 (Shepard-like substrate),
> 실제 relation choice는 context-dependent feature weighting으로 이루어진다 (Tversky-like readout).

- 따라서 triangle inequality violation이 나온다면,
  - 그건 geometry가 없어서라기보다
  - geometry 위의 effective comparison rule이 고정 metric이 아니기 때문이라고 해석할 수 있다.

### Bottom Line

- 이 프로젝트에서:
  - Shepard-like = endpoint direction과 state geometry가 실제로 존재한다는 점
  - Tversky-like = context가 feature weighting을 바꿔 relation choice를 만든다는 점

- 즉 현재 가장 좋은 framing은:

> LLM contains a geometric relational substrate, but ambiguous relation states are resolved through context-dependent feature reweighting rather than a fixed metric readout.

## Next Questions

- outside-A는 정말 commitment / sharpening 역할을 하는가?
- inside/outside를 strict intervention으로 나눠 넣고 빼면 역할이 다르게 드러나는가?
- low-shot behavior에서
  - inside-only는 direction/content를
  - outside-only는 margin/stability를
더 strongly 바꾸는가?

## Bottom Line

- 지금까지의 전체 그림은 다음 한 문장으로 요약할 수 있다.

> Context does not simply replace A with B or D. It preserves a substantial A-structure, reweights multiple features stepwise, and makes the resulting state progressively more B-like or D-like, with inside-A carrying most of the content and outside-A contributing a smaller but selective complement.
