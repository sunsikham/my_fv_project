# Results (Korean Draft)

**인간 행동 데이터.** Consistent triangle (ABC)의 product test는 18개 문항 중 17개에서 1 이하를 유지하여 triangle inequality가 대체로 성립하였다. 반면 mixed triangle (ABD)에서는 다수의 문항에서 PT가 1을 유의하게 초과하였다 (PT 범위: 1.05–3.19; bootstrap p > 1 ≥ 0.95). 이 비대칭은 ambiguous pair A가 참여하는 mixed triangle에서만 triangle inequality가 체계적으로 위반됨을 확인한다.

**LLM 행동 데이터.** Llama-3.1-70B에서도 동일한 방향의 비대칭이 관찰되었다 (Figure 2A). 10-shot 기준 14개 문항 중 9개에서 mixed triangle의 PT가 consistent triangle보다 유의하게 높았으며, 인간과 마찬가지로 위반이 mixed triangle에 집중되었다. Triangle inequality 위반의 조건적 비대칭이 벡터 공간 모델에서도 재현되었다.

**표상 분석.** PCA 투영 결과 (Figure 2B), 순수 조건의 centroid는 뚜렷이 분리된 반면 (Q1: AAA ↔ BBB = 4.52), 교차 조건 BABA의 centroid는 AAA와 BBB 사이에 위치하였다 (AAA ↔ BABA = 2.61, BABA ↔ BBB = 2.42). 이 패턴은 분석 대상 문항 전반에서 관찰되었으며, A의 relational representation이 고정된 점에 머무르지 않고 문맥에 따라 이동함을 보여준다.
