# TECH_SPEC_M5_GPU.md

## M5-B (GPU 실증): Deferred 모델군 L2 승격 + 대형 모델 로딩/경로/토크나이징(BOS) 정합성 확정 + 최소 E2E 스모크(권장)

---

## 1. 목적

M5-B는 M5-A(CPU)에서 `deferred_to_m5b=true`로 남겨둔 모델군을 대상으로, GPU 환경에서 다음을 **실증적으로 확정**합니다.

1. **모델 로딩(대형/가속) 가능성**을 확보하고(권한/캐시/메모리 포함)
2. ModelSpec 기반 **Resolver Smoke를 L2로 승격**하여 실제 모듈 경로가 유효함을 증명하며
3. Tokenizer/BOS 계약이 런타임에서 **의미-동작 불일치 없이** 처리되도록 확정(특히 manual 케이스)하고
4. (권장) 최소 규모로 **FV/injection/eval E2E 스모크**로 “끝까지 돈다”를 확인합니다.

---

## 2. 범위(Scope)

### 2.1 In Scope (필수)

* M5-A summary에서 `deferred_to_m5b=true`인 모든 `spec_key`에 대해:

  * GPU에서 **모델 로딩 시도**
  * **Resolver Smoke L2 수행**(layer_indices 규칙 포함)
  * 성공 시 `resolver_smoke_<spec_key>.json`에 `smoke_level="L2_model_loaded"` 기록 및 `deferred_to_m5b=false`로 전환
* Tokenizer/BOS audit를 GPU 환경에서도 수행하고, M5-A와 **동일한 규칙/매핑**으로 판정 결과를 증적(JSON)으로 남김
* `requires_manual_bos_injection=true`인 모델군에 대해 의미-동작 정합성(7장)을 반드시 종결(해결 또는 명시적 실패)

### 2.2 Out of Scope

* 대규모 sweep, 성능 최적화, 장시간 벤치마크
* M6 전체 관계 파이프라인을 대형 모델로 완전 자동화하여 끝내는 작업

---

## 3. 사전조건(Prerequisites)

* NVIDIA GPU 사용 가능한 PyTorch/CUDA 환경
* (선택) bitsandbytes(4bit/8bit 로딩 시)
* Hugging Face gated 모델(예: 일부 Llama) 접근 시 권한/토큰 필요 가능
* 캐시 디렉토리 고정 권장(`--hf_cache_dir` 또는 환경변수)

---

## 4. 산출물(Artifacts) 계약

### 4.1 출력 디렉토리

* 기본 권장: `artifacts/model_spec_audit_m5b/`
* 실행별 분리를 위해 `--out_dir`로 변경 가능(권장)

### 4.2 필수 산출물(모든 spec_key)

* `bos_audit_<spec_key>.json`
* `resolver_smoke_<spec_key>.json`
* `summary.json`

### 4.3 GPU 메타데이터 필수/선택(파일별 고정)

GPU 메타데이터는 “실행/재현/회귀 비교”를 위해 아래처럼 **파일별 필수 여부를 고정**합니다.

| 파일                               | GPU 메타 필수? | 비고                               |
| -------------------------------- | ---------: | -------------------------------- |
| `summary.json`                   |     **필수** | 운영/자동화 파서 기준                     |
| `resolver_smoke_<spec_key>.json` |     **필수** | 로딩/경로/레이어 검증과 결합됨                |
| `bos_audit_<spec_key>.json`      |     선택(권장) | tokenizer-only 실행을 허용하려면 선택이 합리적 |

**GPU 메타 최소 필드(필수 대상 파일에 포함):**

* `device: "cuda"`
* `gpu_name`
* `torch_version`
* `torch_cuda_version`
* `transformers_version`
* `tokenizers_version`
* `dtype` (fp16/bf16/fp32)
* `load_mode` (예: `full`, `device_map_auto`, `bnb_4bit`, `bnb_8bit`)
* (선택) `device_map_summary`, `max_memory_summary`

---

## 5. 상태 머신(표준화: 파서 친화)

M5-B에서 상태는 아래 2개 필드 조합으로만 표현합니다.

* `status ∈ {"pass","fail"}` (**고정**)
* `deferred_to_m5b: bool` (미완료/추가조치 필요 표시)

**허용 조합(정의):**

* `status="pass" AND deferred_to_m5b=false`
  → 이번 단계에서 완료(이 spec_key는 M5-B 게이트 통과)
* `status="fail" AND deferred_to_m5b=true`
  → 이번 단계에서 완료 못 했고(권한/메모리/환경 등) 추가 조치가 필요(재시도 또는 환경 변경)
* `status="fail" AND deferred_to_m5b=false`
  → 이관이 아니라 “진짜 실패”(스펙/코드 문제 가능성이 커서 수정 후 재실행 필요)

**금지 조합(필수 규칙):**

* `status="pass" AND deferred_to_m5b=true`는 금지(의미 충돌)

> 운영 편의를 위해 summary에 아래 필드를 추가하는 것을 권장(선택):
> `failure_stage: "tokenizer_audit"|"model_load"|"resolver"|"manual_bos"|"e2e_smoke"|null`, `message: str|null`

---

## 6. Tokenizer/BOS 정책 판정 규칙 및 런타임 매핑(= M5-A와 동일)

GPU에서도 BOS 판정 및 `decision_prepend_bos` 매핑은 **M5-A와 완전 동일한 규칙**을 적용합니다.

### 6.1 BOS 정책 판정 규칙(우선순위)

* **Rule 0 (null 처리):** `bos_token_id == null` → `decision_bos_policy="none"`
* **Rule 1 (BOS=EOS 동치):** `bos_token_id != null AND eos_token_id != null AND bos_token_id == eos_token_id`
  → `decision_bos_policy="none"`
* **Rule 2 (tokenizer 삽입):** `add_special_tokens=True`에서 첫 토큰이 BOS면
  → `decision_bos_policy="tokenizer"`
* **Rule 3 (manual):** 위 규칙들에 해당하지 않으면
  → `decision_bos_policy="manual"`

### 6.2 파생 플래그(필수)

* `requires_manual_bos_injection := (decision_bos_policy=="manual")`

### 6.3 런타임 매핑(정책 A, M5-A와 동일) — **필수**

현행 런타임이 `prepend_bos: bool`만 소비하므로, M5-B에서도 아래 매핑을 **공식 규칙**으로 고정합니다.

* `decision_bos_policy="none"` → `decision_prepend_bos=False`
* `decision_bos_policy="tokenizer"` → `decision_prepend_bos=True`
* `decision_bos_policy="manual"` → `decision_prepend_bos=True`  (정책 A: manual 승격)

즉, 등가 표현으로는 아래가 성립합니다.

* `decision_prepend_bos := (decision_bos_policy != "none")`

> 주의: 위 매핑은 “운영상 knob(= special token 경로 사용)”을 의미합니다.
> manual 케이스에서 “실제 BOS 토큰 삽입” 정합성은 7장에서 **해결 또는 명시적 실패**로 종결해야 합니다.

---

## 7. manual BOS 케이스 정합성(필수 게이트)

`requires_manual_bos_injection=true`인 모델군은 GPU 단계에서 **의미-동작 정합성**을 반드시 종결해야 합니다.

### 7.1 요구사항(필수)

* manual 케이스에서 “실제로 입력의 첫 토큰이 BOS가 되도록” 보장할 것(또는 지원 불가를 명시)

### 7.2 처리 옵션(택1, 반드시 summary에 반영)

* **Option B1(권장): 실제 수동 BOS prepend 구현/검증**

  * 입력 토큰열에 BOS를 수동으로 prepend하여 정합성을 보장
  * (권장) `bos_realized=true` 또는 동등 필드로 증적
  * 이 경우 `status="pass", deferred_to_m5b=false` 가능
* **Option B2: 명시적 실패 처리**

  * manual 케이스를 지원하지 않으면 `status="fail"`로 표시
  * `failure_stage="manual_bos"` 및 `message`로 원인 기록
  * “기록만 하고 넘어감” 금지

---

## 8. 모델 로딩 계약(Model Loading Contract)

### 8.1 로딩 모드(권장)

* 소형/중형: `dtype=fp16` 또는 `bf16` 단일 GPU 로드
* 대형: `device_map="auto"` 기반 분산 로드
* 메모리 부족 시: `bnb_4bit`(또는 8bit) 로드

### 8.2 실패 시 기록(필수)

모델 로딩 실패 시:

* `resolver_smoke_<spec_key>.json`에

  * `failed_segment="model_load"`
  * `exception_type`, `message`
  * `status="fail"`, `deferred_to_m5b=true`
* `summary.json`에도 동일 반영

---

## 9. Resolver Smoke (L2) 계약(필수)

### 9.1 레이어 샘플링 규칙(필수)

* `layer_indices = [0, mid, last]` (유효 범위 내 dedup)

  * `mid = floor((n_layers-1)/2)`
  * `last = n_layers-1`

### 9.2 L2에서 확인/기록해야 할 것(필수)

* `smoke_level="L2_model_loaded"`
* `checked_layer_indices` (최종 dedup된 리스트)
* `derived_out_proj_paths` (레이어별 경로 리스트, 길이=checked_layer_indices)
* 경로 해석 결과:

  * `blocks_path_ok=true`
  * `out_proj_path_ok=true`

### 9.3 out_proj 모듈 정보 스키마(레이어별 배열로 고정)

L2에서는 아래 필드를 **레이어별 배열**로 기록합니다(길이=checked_layer_indices).

* `out_proj_module_class_by_layer: list[str|null]`
* `out_proj_weight_shape_by_layer: list[list[int]|null]`
* `out_proj_bias_shape_by_layer: list[list[int]|null]`

> 단, 모델 로딩 실패 등으로 L2가 불가하면 null/빈 배열을 허용하되, 이 경우 `status="fail"`로 처리해야 합니다.

### 9.4 L2 성공 시 상태 전환(필수)

L2가 성공하면:

* `resolver_smoke_<spec_key>.json`에서 `deferred_to_m5b=false`
* `summary.json`에서도 `deferred_to_m5b=false`, `status="pass"`로 기록

---

## 10. (권장) 최소 E2E 스모크: FV/Injection/Eval

### 10.1 목적

대표 1~2개 GPU 모델군(예: `neox`, `llama`)에 대해 최소 설정으로

* FV 생성 → Injection → Eval/score 산출
  까지 “끝까지 돈다”를 확인합니다.

### 10.2 통과 기준(권장)

* 오류 없이 종료
* 결과 파일 생성
* fixed_trials 메타의 prepend_bos 관련 값과 현재 결정이 충돌하면 summary에 경고(선택)

---

## 11. DoD(완료 기준) — 목표/예외 분리

### DoD-B1 (목표): Deferred 해소(가능한 범위 내)

* M5-A에서 `deferred_to_m5b=true`였던 `spec_key`에 대해, **가능한 한 많이**:

  * `resolver_smoke_<spec_key>.json`이 `L2_model_loaded`로 생성
  * `deferred_to_m5b=false`
  * summary에서 `status="pass"`

### DoD-B1E (허용 예외): 환경/권한/메모리 제한으로 인한 미종결

권한(gated), 메모리(OOM), 인프라 제약 등으로 **모델 로드 자체가 불가능**한 경우, 아래 증적을 남기면 “스펙 준수(예외로 인정)”로 간주합니다.

* `status="fail" AND deferred_to_m5b=true`
* `failure_stage="model_load"`(권장) 또는 동등한 실패 구간 식별
* `message`에 재현 가능한 원인(예: auth/oom/device_map) 기록
* `resolver_smoke_<spec_key>.json`, `summary.json` 존재 및 스키마 필수 필드 충족

### DoD-B2 (필수): manual BOS 정합성 종결

* `requires_manual_bos_injection=true`인 모델군은

  * Option B1로 해결하여 `pass + deferred=false` **또는**
  * Option B2로 `fail`로 명시(숨기지 않음)

### DoD-B3 (필수): 상태 머신 규칙 준수

* summary에서 금지 조합(`pass` + `deferred_to_m5b=true`)이 존재하지 않음

### DoD-B4 (권장): 최소 E2E 스모크 1회 이상

* 대표 GPU 모델 1개 이상에서 FV/injection/eval이 끝까지 수행됨

---

## 12. 리스크 및 대응(요약)

* gated 모델 접근 실패 → `fail + deferred=true`로 남기고 권한/환경 해결 후 재시도(DoD-B1E)
* 메모리 부족 → load_mode를 device_map/4bit로 전환하고 로딩 메타를 증적에 기록(DoD-B1E)
* manual BOS 불일치 → DoD-B2로 강제 게이트(기록만 하고 넘어가기 금지)
* 경로는 맞는데 모듈이 기대와 다름 → 레이어별 class/shape 증적으로 조기 탐지

---
