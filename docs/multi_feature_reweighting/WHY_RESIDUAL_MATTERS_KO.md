# 왜 residual이 의미가 있는가

이 문서는 현재 분석에서 residual을 왜 단순한 오차나 노이즈로 보지 않고, 해석적으로 중요한 신호로 보는지 정리한 메모이다.

## 1. residual이란 무엇인가

Stage 1에서 먼저 본 것은 mixed state가 intended endpoint 쪽으로 얼마나 이동했는가였다.

- `BABABA`는 `A -> B`
- `DADADA`는 `A -> D`

를 기준으로 보았다.

예를 들어 `BABABA`에 대해 말하면, 최종 변화는

\[
x_{bab} - a
\]

이고, 우리가 먼저 보고 싶은 것은 이 변화가 `A -> B` 방향으로 얼마나 갔는가이다.

그 값을 progress로 계산한다.

\[
p_B(x_{bab}) = \frac{(x_{bab}-a)^\top (b-a)}{\|b-a\|^2}
\]

그 다음 residual은, 이 전체 변화에서 `A -> B` 축 위 성분을 제거하고도 얼마나 변화가 남는지를 본다.

\[
r_B(x_{bab})
=
\frac{\|(x_{bab}-a) - p_B(x_{bab})(b-a)\|}{\|b-a\|}
\]

즉 residual은

- intended endpoint 쪽으로 간 성분을 빼고도
- 옆 방향으로 얼마나 변화가 남는가

를 나타내는 값이다.

## 2. residual이 작다면 무슨 뜻인가

만약 context가 정말로 `A`를 지우고, 거의 straight하게 `B` 또는 `D` 상태로 바꾼다면, 변화는 거의 한 축 위에서 일어나야 한다.

그러면:

- progress는 커지고
- residual은 작아진다

이 경우 해석은 비교적 단순하다.

> "상태가 거의 one-axis interpolation처럼 endpoint 쪽으로 이동했다."

즉 residual이 작으면,

- 단일 직선 이동 가설
- 단일 축 치환 가설

이 잘 맞는다고 볼 수 있다.

## 3. 그런데 지금 residual은 작지 않다

Q1에서:

- `BABABA -> B progress = 0.528`
- `DADADA -> D progress = 0.509`

즉 intended endpoint 방향 이동 자체는 분명하다.

그런데 residual도 동시에 남는다.

- `r_B = 0.343`
- `r_D = 0.301`

이 수치는 "거의 0"이라고 보기 어렵다.

즉 관찰된 변화는

- endpoint 방향으로 가고는 있지만
- 그 변화 전체가 한 직선 위에서 설명되지는 않는다

는 뜻이다.

바로 이 점 때문에 residual은 중요하다.

## 4. residual은 왜 단순 noise로 보기 어려운가

residual을 단순 noise로 보려면,

- 값이 매우 작거나
- 질문/조건마다 불안정하게 흔들리거나
- endpoint 방향성과 무관하게 랜덤하게 남아야 한다

그런데 지금은 다르게 보인다.

1. progress가 분명히 존재한다  
   즉 변화는 완전히 랜덤하지 않다.

2. residual이 동시에 작지 않다  
   즉 intended movement 외에도 구조적 변화가 남는다.

3. 이후 분석에서 inside-A 변화와 feature reweighting 패턴이 보인다  
   즉 residual은 "설명되지 않은 찌꺼기"라기보다, 단일 축 설명으로는 다 담기지 않는 내부 재구성의 흔적에 가깝다.

그래서 현재 해석에서는 residual을

> "noise라기보다 one-axis picture로는 설명되지 않는 structured change"

로 보는 것이 더 자연스럽다.

## 5. 왜 residual이 가설과 직접 연결되는가

현재 가설의 핵심은:

> context does not simply replace A with B or D.  
> Instead, it reweights multiple features already present in A.

만약 이 가설이 맞다면, 변화는 단순한 `A -> B` 직선 이동일 필요가 없다.

오히려 더 자연한 그림은:

- A 안에는 이미 여러 feature가 섞여 있고
- context가 그 feature들의 weight를 다르게 조정하면서
- 결과 상태를 더 B-like 또는 D-like로 만든다

는 것이다.

이 경우 기대되는 것은:

- endpoint 쪽 progress는 존재
- 하지만 residual도 동시에 존재

이다.

왜냐하면 여러 feature가 서로 다른 방식으로 재가중되면, 그 변화는 단일 축으로 압축되지 않기 때문이다.

즉 residual은

> "A가 wholesale replacement된 것이 아니라, 내부 feature 분포가 재조정되었다는 흔적"

으로 읽을 수 있다.

## 6. joint residual이 더 말해주는 것

raw cross-score에는 또 하나의 문제가 있다.

- `A -> B` 축과 `A -> D` 축이 완전히 직교하지 않아서
- B로만 간 변화도 D projection이 어느 정도 잡힐 수 있다

그래서 separate projection만으로는

- 진짜 cross-component
- 축 overlap 때문에 생긴 겉보기 cross-score

를 구분하기 어렵다.

이 문제를 줄이기 위해 joint decomposition을 쓴다.

\[
x-a \approx \alpha u_B + \beta u_D + \varepsilon
\]

이제 residual은

- B 성분
- D 성분

을 동시에 설명하고도 남는 부분이다.

즉 joint residual은 더 보수적으로 보아도

> "B/D 두 축만으로는 다 설명되지 않는 나머지 변화"

를 뜻한다.

Q1에서 joint residual이 여전히 작지 않다면, 그건 "축 overlap만 제거하면 다 해결되는 문제"가 아니라는 뜻이다.

## 7. 현재 해석에서 residual의 의미

현재 결과를 가장 자연스럽게 요약하면:

1. mixed state는 intended endpoint 방향으로 실제로 이동한다
2. 하지만 그 이동은 one-axis straight shift가 아니다
3. 변화의 상당 부분은 inside-A에서 일어난다
4. 따라서 residual은 단순 noise라기보다, multi-feature reweighting의 흔적이다

즉 residual의 의미는 다음과 같이 정리할 수 있다.

> residual이 남는다는 것은 모델이 endpoint 쪽으로 가긴 가지만, 그 과정을 단일 축 치환으로 수행하는 것이 아니라, A 안의 여러 feature를 재조정하면서 더 복합적인 방식으로 state를 바꾸고 있다는 뜻이다.

## 8. 한 문장 요약

Residual이 의미 있는 이유는, 그것이 "endpoint 방향 이동 외에 남는 무의미한 찌꺼기"가 아니라, one-axis substitution으로는 설명되지 않는 structured reweighting의 측정 가능한 흔적이기 때문이다.
