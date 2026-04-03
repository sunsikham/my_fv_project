# Note: How To Interpret Outside-A Change

## Summary

- 현재 결과는 `outside-A` change가 존재하고, 단순 noise가 아니라 endpoint-selective한 방향으로 기여한다는 것을 보여준다.
- 하지만 이것이 곧바로
  - 완전히 새로운 semantic relation의 도입
  - 또는 outside-A가 main mechanism
를 뜻하는 것은 아니다.

- 가장 자연스러운 현재 해석은:

> main mechanism은 inside-A reweighting이고, outside-A는 더 작지만 선택적이며 endpoint-sharpening / disambiguation / binding을 돕는 보조 성분이다.

## 1. Important Clarification

- `outside-A`는

> A와 완전히 무관한 의미

를 뜻하지 않을 수 있다.

- 현재 `outside-A`의 정확한 의미는:

> 지금 만든 `A-only` linear basis로는 설명되지 않는 성분

이다.

- 따라서 outside-A는
  - 완전히 새로운 semantic factor
일 수도 있지만,
  - A-related structure의 extension / curvature
일 수도 있다.

## 2. Interpretation 1: A-Space Is Only A Linear Approximation

- `G_A`는 `AAAAAA`에서 본 A states로 만든 linear basis다.
- 실제 A representation은 더 넓은 manifold일 수 있다.
- context가 들어오면 이 manifold 위에서 곡선형 변화가 생길 수 있다.

- 이런 변화는
  - 여전히 A-related structure일 수 있지만
  - 현재 linear `G_A` 안에는 완전히 들어오지 않아
  - `outside-A`처럼 보일 수 있다.

### Consequence

- outside-A는 반드시 “새 semantic relation”이 아닐 수 있다.
- 오히려

> 현재 A-only basis가 다 담지 못한 A-related curvature / extension

일 수 있다.

## 3. Interpretation 2: Context Adds A Selection / Commitment Signal

- 네 가설대로 A 안에 이미
  - B facet
  - D facet
가 공존한다고 하자.

- 그 경우 context의 main 역할은 reweighting이다.
- 하지만 실제로 하나의 endpoint를 더 확정하려면
  - competition suppression
  - commitment
  - sharpening
같은 추가 신호가 필요할 수 있다.

- 이 보조 신호가 `outside-A`일 수 있다.

### Consequence

- inside-A는 content reweighting
- outside-A는 selection / commitment / sharpening

처럼 역할이 나뉠 수 있다.

## 4. Interpretation 3: Role / Context Binding

- `AAAAAA`의 A와 `BABABA` 안의 A는 semantic core는 비슷해도

> 어떤 문맥 아래에서 읽히는 A인가

라는 role 정보가 다를 수 있다.

- 모델은 단순히 relation facet만 바꾸는 것이 아니라,
  - “이 A는 지금 B-heavy context 아래에서 해석되어야 한다”
는 binding / routing 성분을 추가할 수 있다.

- 이런 종류의 변화는 A-only baseline에선 잘 드러나지 않아 outside-A에 잡힐 수 있다.

## 5. Why This Does Not Contradict The Main Hypothesis

- 현재 결과는
  - 네 가설이 틀렸다는 뜻이 아니다.
- 더 자연스러운 해석은:

### Main

- A 내부에 이미 존재하는 relation facet들이 재가중된다

### Secondary

- context가 선택/확정/sharpening용 작은 추가 성분도 붙인다

즉:

> mostly inside reweighting + smaller selective outside support

가 현재까지 결과와 가장 잘 맞는다.

## 6. Important Caution: “Used” Is Not The Same As “Necessary”

- 지금까지 우리가 보인 것은
  - outside-A가 존재한다
  - endpoint-selective한 방향으로 기여한다
까지다.

- 아직 보이지 않은 것은

> outside-A가 없으면 B-like / D-like shift가 무너지는가

이다.

- 즉 지금 단계에서 말할 수 있는 가장 정확한 문장은:

> outside-A is being used in a selective way

이지,

> outside-A is necessary

는 아니다.

## 7. Strongest Current Interpretation

- 현재 strongest interpretation은 다음과 같다.

> A contains multiple relation facets already, and the primary mechanism of context is to reweight those facets inside the A-space. In addition, context seems to introduce a smaller but selective outside-A component that sharpens, disambiguates, or binds the final B-like / D-like endpoint state.

- 한국어:

> A 안에는 이미 여러 relation facet가 들어 있고, context의 main mechanism은 그 facet들을 inside-A에서 재가중하는 것이다. 여기에 더해, context는 더 작지만 선택적인 outside-A 성분을 추가하여 최종 B-like / D-like endpoint를 더 선명하게 하거나 확정하는 것으로 보인다.

## 8. Bottom Line

- outside-A는 현재 결과만 놓고 보면
  - 단순 noise는 아니다
  - main mechanism도 아니다
- 가장 자연한 해석은:

> inside-A reweighting이 주된 메커니즘이고, outside-A는 더 작지만 endpoint-selective한 보조 성분이다.

