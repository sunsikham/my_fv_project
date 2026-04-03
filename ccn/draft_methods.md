# Methods (Korean Draft)

**자극 및 인간 데이터.** 18개 문항에서 각 4개의 쌍(A, B, C, D)을 구성하였다. 쌍 A는 두 관계에 동시에 참여하는 ambiguous pair이며, 이를 중심으로 consistent triangle (ABC: 모든 쌍이 동일한 관계 공유)과 mixed triangle (ABD: A가 B, D와 각각 다른 관계를 공유)을 구성하였다 (Figure 1). 30명의 참가자가 clipart 자극에 대해 모든 쌍 조합의 관계 유사성을 7점 Likert 척도로 평가하였다 (Peterson et al., 2020).

**LLM 실험.** 18개 문항 중 ambiguous pair에 대한 in-context 예시 구성이 가능한 14개 문항을 선정하였다. Llama-3.1-70B (4-bit NF4 quantization)에 순차적 in-context learning으로 관계 예시를 제시하였다. 5개 edge (AB, AC, AD, BC, BD) 각각에 대해 해당 관계의 예시를 1–10 shot 제시한 뒤, 해당 관계가 성립할 때 예상되는 target token의 logit probability를 관계 유사성의 측정치로 사용하였다. 각 shot 수에서 독립적으로 분석하였으며, 유사성 정규화는 문항 내 전체 logit 분포의 5th–95th percentile 기준 robust min-max를 적용하여 [0, 1]로 변환하였다.

**Product test.** Triangle inequality의 성립 여부를 판별하기 위해 product test를 적용하였다 (Peterson et al., 2020). 세 edge의 유사성 값으로부터 PT(x, y, z) = xyz / min(x, y, z)²을 산출하며, PT > 1이면 위반으로 판정한다. 인간과 LLM 모두에 동일한 공식을 적용하고, consistent triangle (ABC)과 mixed triangle (ABD) 각각에 대해 10,000회 bootstrap으로 신뢰구간을 산출하였다.

**표상 분석.** Function vector (Todd et al., 2024)를 통해 ICL 과정에서 형성되는 조건별 relational representation을 추출하였다. 순수 조건 (AAA, BBB, DDD: 단일 관계의 예시만 반복 제시)과 교차 조건 (BABA, DADA: 두 관계의 예시를 교대 제시)을 설정하고, activation patching을 통해 관계 처리에 대한 indirect effect가 가장 높은 상위 20개 attention head의 output projection 기여를 합산하여 trial-level 벡터를 구성하였다. 전 조건을 공통 PCA 공간 (3 components)에 투영하여 조건 간 centroid 거리를 비교하였다.
