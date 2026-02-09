1. 목적

본 문서는 M5를 **CPU 선행 단계(M5-A)**와 **GPU 실증 단계(M5-B)**로 분리했을 때, M5-A에서 수행할 구현/검증 범위와 산출물 계약을 정의합니다.

M5-A의 목표는 다음 2가지를 동시에 달성하는 것입니다.

현행 런타임 스키마(fv/model_spec.py)와 문서 스펙의 불일치를 제거하고, 모델 확장을 “스펙 추가”로만 가능하게 만든다.

모델군별로 토크나이저 동작(BOS/PAD 포함)을 CPU에서 사전 Audit하여, 입력 토큰열/타깃 토큰 위치가 흔들리지 않도록 “결정 근거(증적)”를 남긴다.

본 문서는 “문서만 그럴듯하게”가 아니라, 현재 코드가 실제로 소비하는 스키마를 Source of Truth로 삼습니다.

문서 상태 태그:

- `Current`: 현재 코드(`fv/model_spec.py`)에 이미 반영된 사실
- `Target(M5-A)`: M5-A 완료 시점 목표 상태(아직 미구현 가능)

2. 배경 및 핵심 원칙

Source of Truth(중요): M5-A는 현행 ModelSpec 런타임 스키마를 변경하지 않습니다.
즉, ModelSpec의 핵심 필드는 아래를 기준으로 합니다.

prepend_bos: bool

blocks_path: str

attn_path_in_block: str

out_proj_path_in_attn: str

문서에서 bos_policy / pad_policy / out_proj_path_template 같은 확장 필드는 ModelSpec “필수 필드”가 아니라:

(a) Audit 산출물(JSON)에서 기록하고

(b) 런타임에서는 기존 필드(prepend_bos 등)로 컴파일/매핑합니다.

회귀 방지 우선: GPT-2 기반 parity 경로(M1~M4)를 깨지 않는 것이 최우선입니다.

3. 범위(Scope)
3.1 In Scope (M5-A에서 반드시 수행)

아래 모델군 전부에 대해 ModelSpec registry 엔트리를 추가/정리

gpt2, gpt2-xl

gpt-j*

gpt-neox* / pythia*

gemma*

llama*

olmo*

모델군별 out_proj 개입 경로를 blocks_path + attn_path_in_block + out_proj_path_in_attn로 정규화

모델군별 Tokenizer Audit(BOS/PAD/EOS) 수행 + 결과를 산출물로 고정 저장

모델군별 Resolver Smoke(경로 해석 가능성) 수행 + 결과를 산출물로 고정 저장

CPU에서 가능한 모델군에 대해 minimal forward + hook attach/detach 스모크(권장)

3.2 Out of Scope (M5-A에서 하지 않음)

대형 모델을 GPU로 로드하여 FV 생성/주입/평가 end-to-end 실증 (M5-B로 이관)

FP16/BF16/quantization 수치 특성 검증 및 성능 최적화

relation StepD/E/6 대규모 실행

4. 용어 및 스키마 정의
4.1 ModelSpec 런타임 스키마(현행, 변경 없음)

ModelSpec dataclass 엔트리는 최소 아래 필드를 만족해야 합니다.

prepend_bos: bool

blocks_path: str

attn_path_in_block: str

out_proj_path_in_attn: str

registry key(`spec_key`)는 dataclass 필드가 아니라 `MODEL_SPECS` 딕셔너리의 키입니다.

out_proj_path_template는 별도 필드가 아니라, 아래처럼 파생(derived) 표현입니다.
out_proj_path_template := f"{blocks_path}.{{layer}}.{attn_path_in_block}.{out_proj_path_in_attn}"

4.2 BOS 정책(bos_policy)은 “Audit 결과”로만 보관

문서에서 사용하는 BOS 정책은 3상태 분류값이며, 이는 ModelSpec 필드가 아니라 Audit 산출물(JSON)의 필드입니다.

bos_policy ∈ { "none", "tokenizer", "manual" }

4.2.0 bos_policy 판정 규칙(기계적, 필수)

Tokenizer Audit에서 아래 순서로 판정합니다.

1) `bos_token_id is null` 이면 `bos_policy="none"`

2) 예외 규칙: `bos_token_id`와 `eos_token_id`가 모두 존재하고 `bos_token_id == eos_token_id`이면 `bos_policy="none"`

3) `bos_token_id is not null` 이고, `add_special_tokens=True` 인코딩의 첫 토큰이 BOS면 `bos_policy="tokenizer"`

4) `bos_token_id is not null` 이고, `add_special_tokens=True` 인코딩에서도 첫 토큰이 BOS가 아니면 `bos_policy="manual"`

4.2.1 bos_policy → prepend_bos 매핑 규칙(문서 계약)

현행 런타임은 prepend_bos: bool만 소비하므로, M5-A에서는 아래 매핑을 “공식 규칙”으로 둡니다.

bos_policy = "none" → prepend_bos = False

bos_policy = "tokenizer" → prepend_bos = True

bos_policy = "manual" → M5-A에서는 런타임 차등 구현이 없으므로 다음 중 하나를 문서로 고정해야 합니다.

정책 A(권장): manual을 tokenizer로 승격 처리하여 prepend_bos=True로 운용하고, “수동 prepend 구현은 M5-B 범위”로 명시

정책 B: manual을 M5-A에서 “불가/블로킹”으로 두고, 해당 모델군은 M5-B에서만 지원

본 문서에서는 **정책 A(권장)**를 기본으로 채택합니다.
즉, M5-A 단계에서는 manual이 감지되더라도 런타임은 prepend_bos=True로 운용하며, 차등 동작(진짜 수동 prepend)은 M5-B로 이관합니다.

재현성 고정 규칙(필수):

`decision_bos_policy="manual"` 이면 `decision_prepend_bos=True`를 JSON에 항상 기록합니다.

의미 분리 고정 규칙(필수):

`requires_manual_bos_injection := (decision_bos_policy=="manual")`를 `bos_audit_*.json` 및 `summary.rows[]`에 항상 기록합니다.

4.3 PAD 정책(pad_policy)도 “Audit 결과”로만 보관

많은 CausalLM 토크나이저는 pad_token이 없거나(e.g. GPT-2) 모델군별 기본값이 다릅니다.

M5-A에서는 pad 정책을 ModelSpec에 넣지 않고, Audit 산출물에 기록합니다.

예시 분류:

pad_policy = "use_eos_as_pad"

pad_policy = "tokenizer_provided_pad"

pad_policy = "none" (권장하지 않음; padding 필요 시 실패)

4.3.1 PAD 적용 계약(M5-A 고정)

M5-A는 **기록 중심(A안)**으로 고정합니다.

- Audit은 `decision_pad_policy`를 기록합니다.
- 실제 런타임 적용은 기존 loader/tokenizer 로직을 따릅니다(스펙에서 신규 강제 세팅 로직을 추가하지 않음).
- 단, `decision_pad_policy="use_eos_as_pad"`인 경우, 대표 스모크에서 pad_token_id가 eos_token_id와 일치하는지 검증 결과를 summary에 기록합니다.

5. 모델군 커버리지 및 registry key 규칙
5.1 spec_key 네이밍 규칙(필수)

spec_key는 소문자 + underscore만 사용합니다.

와일드카드 표기(gpt-j*, gpt-neox*)는 문서에서만 사용하고, registry key는 고정 문자열로 둡니다.

권장 key:

gpt2

gpt2_xl

gptj

neox (pythia 포함)

gemma

llama

olmo

5.2 모델군별 ModelSpec 초기값(1차 소스: 제공된 MODEL_CONFIG, `Target(M5-A)`)

아래 값들은 ModelSpec 필드로 직접 반영 가능한 “구조 정보”입니다.

| spec_key | prepend_bos(초기) | blocks_path | attn_path_in_block | out_proj_path_in_attn |
| --- | --- | --- | --- | --- |
| gpt2 | False | transformer.h | attn | c_proj |
| gpt2_xl | False | transformer.h | attn | c_proj |
| gptj | False | transformer.h | attn | out_proj |
| neox | False | gpt_neox.layers | attention | dense |
| gemma | True(초기) | model.layers | self_attn | o_proj |
| llama | True(초기) | model.layers | self_attn | o_proj |
| olmo | False(초기) | model.layers | self_attn | o_proj |

“초기 prepend_bos”는 제공된 코드의 가정을 그대로 옮긴 값이며, 최종 확정은 6장의 Tokenizer Audit 결과를 통해 결정합니다(단, 런타임 반영은 4.2.1 매핑 규칙에 따름).

6. 검증(Verification) 절차
6.1 Tokenizer Audit (필수, CPU만으로 가능)

목적: 모델군별로 BOS/PAD/EOS 동작을 확정하고, 그 결정 근거를 산출물로 남깁니다.

6.1.1 Audit 체크 항목(필수 기록)

각 모델군(실제 HF model id 1개 이상 대표 샘플)에 대해:

tokenizer 식별

tokenizer class / name_or_path

special token ids

bos_token_id, eos_token_id, pad_token_id

인코딩 동작

add_special_tokens=True일 때 첫 토큰이 BOS인지 여부

add_special_tokens=False일 때 첫 토큰이 BOS인지 여부

결론

bos_policy 판정

pad_policy 판정

런타임 적용값 prepend_bos (4.2.1 규칙에 따라)

6.2 Resolver Smoke (필수, CPU 로드 가능 여부에 따라 수준 다름)

목적: ModelSpec이 가리키는 경로가 실제 모듈 트리에서 해석 가능한지(또는 최소한 합리적인지) 확인하고 증적을 남깁니다.

6.2.1 수행 레벨 정의

Level-1 (CPU 토크나이저만 가능): tokenizer audit만 수행하고 resolver는 “deferred”로 기록

Level-2 (CPU 모델 로드 가능): 실제 모델 객체에서 아래를 확인

blocks_path 해석 성공 여부

샘플 레이어 다중 인덱스에서 out_proj 경로 해석 성공 여부

Level-2 다중 레이어 규칙(필수):

`layer_indices=[0, mid, last]`를 기본으로 사용하며, 유효 범위 내에서 dedup 후 검사합니다.

(예: n_layers=1이면 [0], n_layers=2이면 [0,1], n_layers>=3이면 [0, mid, last])

실패 시 “어느 segment에서 실패했는지”를 기록

CPU에서 모델 로드가 비현실적인 경우(대형 llama 등)는 M5-A에서 Level-1까지 완료로 인정하되, M5-B에서 Level-2를 수행하도록 “이관 기록”을 남겨야 합니다.

6.3 Minimal Forward + Hook Attach/Detach Smoke (권장)

CPU 로드 가능한 모델군(예: gpt2/gpt2_xl 등)에서:

아주 짧은 입력으로 forward 1회

hook attach/detach가 오류 없이 수행됨을 확인
(효과 측정/수치 동일성은 이 단계의 목적이 아닙니다.)

7. 산출물(Artifacts) 계약(필수)
7.1 저장 경로(필수)

artifacts/model_spec_audit/ 아래에 저장합니다.

7.2 산출물 목록(필수)

artifacts/model_spec_audit/bos_audit_<spec_key>.json

artifacts/model_spec_audit/resolver_smoke_<spec_key>.json

artifacts/model_spec_audit/summary.json

7.3 bos_audit JSON 스키마(필수 필드)

spec_key

hf_model_id (감사에 사용한 대표 모델 id)

tokenizer_class

tokenizer_name_or_path

bos_token_id, eos_token_id, pad_token_id

first_token_is_bos_add_special_tokens_true: bool

first_token_is_bos_add_special_tokens_false: bool

decision_bos_policy: "none"|"tokenizer"|"manual"

decision_pad_policy: str

decision_prepend_bos: bool (4.2.1 매핑 결과)

requires_manual_bos_injection: bool

7.4 resolver_smoke JSON 스키마(필수 필드)

spec_key

hf_model_id (스모크에 사용한 대표 모델 id; Level-1이면 동일 audit id)

smoke_level: "L1_tokenizer_only"|"L2_model_loaded"

blocks_path: str

checked_layer_indices: int[]

derived_out_proj_paths: string[] (checked_layer_indices와 1:1 대응)

blocks_path_ok: bool|null (L1이면 null 허용)

out_proj_path_ok: bool|null (L1이면 null 허용)

out_proj_module_class: string|null

out_proj_weight_shape: int[]|null

out_proj_bias_shape: int[]|null

실패 시:

failed_segment: str|null

exception_type: str|null

message: str|null

deferred_to_m5b: bool (L1이면 True 권장, L2면 False)

7.5 summary.json 스키마(필수)

모델군별 운영 요약을 한 파일에서 조회할 수 있도록 아래 필드를 포함합니다.

spec_key

decision_prepend_bos

decision_bos_policy

decision_pad_policy

requires_manual_bos_injection

smoke_level

deferred_to_m5b

status: "pass"|"fail"

8. 구현 작업(Tasks)
Task A1 — ModelSpec registry 정리(필수)

5.2 표의 모델군을 모두 registry에 등록

registry는 spec_key 기준으로 조회 가능해야 함

Task A2 — Tokenizer Audit 실행 및 산출물 생성(필수)

6.1 절차 수행

7.3 스키마에 맞게 bos_audit_*.json 생성

Task A3 — Resolver Smoke 실행 및 산출물 생성(필수)

CPU 가능 모델군은 Level-2 수행, 불가 모델군은 Level-1 수행 + 이관 표시

7.4 스키마에 맞게 resolver_smoke_*.json 생성

Task A4 — (권장) CPU minimal forward + hook smoke

최소 1개 이상 CPU 가능 모델군에서 수행 결과를 기록(로그/메모)

9. 완료 기준(DoD) — 단계형
DoD-A1 (필수): Tokenizer Audit 완료

모든 spec_key에 대해 bos_audit_<spec_key>.json이 존재하고 스키마 필수 필드를 충족

DoD-A2 (필수): Resolver Smoke 완료(예외 규정 포함)

모든 spec_key에 대해 resolver_smoke_<spec_key>.json이 존재

CPU 로드 가능 모델군은 smoke_level="L2_model_loaded"로 기록

CPU 로드 불가 모델군은 smoke_level="L1_tokenizer_only" + deferred_to_m5b=true로 기록

DoD-A3 (권장): CPU forward/hook smoke

CPU 가능 모델군 최소 1개에서 forward 1회 + hook attach/detach가 실패 없이 수행됨

10. 리스크 및 대응

BOS 정책 오판정 위험: tokenizer가 자동으로 BOS를 붙이는지 여부는 모델/버전별로 달라질 수 있으므로, 반드시 Audit 산출물로 근거를 남기고, 변경 시 diff가 추적 가능해야 합니다.

대형 모델 CPU 로드 불가: M5-A에서 Level-1로 완료 인정하되, “M5-B 이관”을 산출물에 명시하여 누락을 방지합니다.

문서-구현 불일치 재발: 본 문서는 런타임 스키마를 고정 Source of Truth로 선언했으므로, 향후 스키마 변경이 필요하면 “코드 변경 + 문서 변경”을 함께 수행해야 합니다.

11. 후속 문서(별도)

TECH_SPEC_M5_GPU.md (M5-B): 대형 모델 GPU 로드, resolver 실증(Level-2), 최소 E2E(FV/injection/eval) 검증을 다룹니다.
