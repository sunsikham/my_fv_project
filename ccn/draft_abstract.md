# Abstract (Korean Draft)

유추의 계산적 기반을 설명하기 위해, vector-space account는 관계를 차이벡터로 표현하고 각 쌍에 문맥과 무관하게 고정된 하나의 표상을 가정한다. 그러나 이 고정 표상 가정이 타당한지, 아니면 관계 표상이 문맥에 따라 동적으로 재구성되는지는 열린 질문이다. Tversky(1977)가 항목 간 유사성에서 보인 triangle inequality 위반을 관계 유사성으로 확장한 Peterson et al.(2020)은, 하나의 쌍이 문맥에 따라 서로 다른 관계에 참여할 수 있는 경우에만 이 위반이 체계적으로 나타남을 보고했다. 이러한 조건적 비대칭은 관계 표상이 고정되어 있지 않을 가능성을 시사하지만, 행동 데이터만으로는 이것이 판단 수준의 편향인지 표상 자체의 변화인지를 구분할 수 없다.

본 연구는 벡터 공간 위에서 표상을 형성하면서도 내부 표상에 직접 접근할 수 있는 대규모 언어모델(Llama-3.1-70B)을 통해 이 질문을 검증한다. Triangle inequality를 진단 도구로 활용하여, 세 쌍이 동일한 관계를 공유하는 consistent triangle(ABC)과 ambiguous pair가 두 관계에 걸쳐 있는 triangle(ABD)을 비교한 결과, ABC에서는 triangle inequality가 유지된 반면 ABD에서는 인간과 동일한 방향의 체계적 위반이 관찰되었다. 위반의 표상적 원인을 규명하기 위해 in-context learning 과정에서 형성되는 관계 표상을 추출하고 공통 PCA 공간에 투영한 결과, ABD 조건에서 ambiguous pair의 표상이 문맥에 따라 서로 다른 관계 방향으로 이동함을 확인했다.

이 결과는 관계 표상이 문맥과 무관하게 고정되어 있다는 vector-space account의 핵심 가정에 도전한다. 관계 표상은 순차적 문맥에 의해 동적으로 재조직되며, triangle inequality 위반은 이러한 context-dependent re-representation의 귀결이다. 이는 관계 유사성에 대한 계산적 설명이 표상의 문맥 의존적 변화를 고려해야 함을 시사한다.
