# TECH_SPEC_M0.md

## Scope (M0 only)
### In
- Determinism/runtime contract를 parity 검증 경로에 적용
- Fixed trials 기반 prompt/slot parity 검증 수행
- `scripts/verify_prompt_parity.py`, `scripts/verify_slot_parity_against_src.py` 실행 및 PASS 확인
- M0 전용 실행 래퍼(`scripts/run_m0_checks.sh`) 정의 및 사용

### Out
- `fv` 코어 로직 구현/수정(예: `compute_function_vector`, injection 구현)
- mean activations/indirect effect/FV 아티팩트 생성(M1+)
- relation 실험 파이프라인(M6)

## Determinism Contract 적용 방법 (src/fv 동일 강제)
M0에서는 아래 계약을 `src`/`fv` 양쪽 비교에 동일하게 적용한다.

- CPU
- float32
- `model.eval()`
- 고정 seed
- `torch.set_grad_enabled(False)`

적용 위치:
- M0 전용 실행 래퍼: `scripts/run_m0_checks.sh`에서 강제
- 개별 verify 스크립트 실행 전 환경/시드 고정
- 원칙: `src` 호출과 `fv` 호출 모두 동일 설정으로 실행

`run_m0_checks.sh` 최소 책임:
- `export PYTHONHASHSEED=0`
- `export TOKENIZERS_PARALLELISM=false`
- 고정 seed 값(`SEED=42`)을 명시하고 verify 커맨드에 동일하게 주입
- 동일 검증 세트를 연속 2회 실행하고 결과(`mismatch_count`)를 비교
- 불일치 또는 non-zero mismatch면 non-zero exit code 반환

## Fixed Trials 입력
기본 fixed trials 파일:
- `datasets/fixed_trials/fixed_trials_antonym_t10_s10_seed0.json`

M0에서 clean/corrupted는 “계약 존재 확인” 항목으로만 다룬다.
- 계약:
  - mean activations는 `prompt_data_clean`
  - indirect effect는 `prompt_data_corrupted`
- M0에서는 위 계약을 계산 단계가 아니라 검증 전제 조건으로 확인만 한다.

계약 존재 확인(필수):
- trials의 각 원소에 아래 키가 존재해야 한다.
  - `prompt_data_clean`
  - `prompt_data_corrupted`
  - `clean_prompt_str`
  - `corrupted_prompt_str`

검증 커맨드(권장, repo root):
```bash
python - <<'PY'
import json
from pathlib import Path
p=Path("datasets/fixed_trials/fixed_trials_antonym_t10_s10_seed0.json")
d=json.loads(p.read_text())
trials=d.get("trials", [])
assert trials, "trials missing/empty"
keys=("prompt_data_clean","prompt_data_corrupted","clean_prompt_str","corrupted_prompt_str")
for i,t in enumerate(trials):
    for k in keys:
        if k not in t:
            raise AssertionError(f"missing key: trial={i} key={k}")
print("fixed_trials schema check: PASS", "trials=", len(trials))
PY
```

## 실행 커맨드 (repo root)
아래 2개를 필수 실행한다.

1. Prompt parity
```bash
python scripts/verify_prompt_parity.py \
  --fixed_trials_path datasets/fixed_trials/fixed_trials_antonym_t10_s10_seed0.json \
  --max_trials 5 \
  --model_name_for_tokenizer gpt2
```

2. Slot parity (src vs fv)
```bash
python scripts/verify_slot_parity_against_src.py \
  --fixed_trials_path datasets/fixed_trials/fixed_trials_antonym_t10_s10_seed0.json \
  --max_trials 5 \
  --mode corrupted \
  --tokenizer_name gpt2 \
  --assert_zero
```

권장: `--mode clean`도 1회 추가 확인
```bash
python scripts/verify_slot_parity_against_src.py \
  --fixed_trials_path datasets/fixed_trials/fixed_trials_antonym_t10_s10_seed0.json \
  --max_trials 5 \
  --mode clean \
  --tokenizer_name gpt2 \
  --assert_zero
```

## Expected Output / PASS 기준 (DoD)
- 두 verify 스크립트 모두 종료 코드 0
- 출력에 `mismatch_count: 0`
- 동일 커맨드를 연속 2회 실행했을 때 결과 동일
  - 최소 기준: 두 번 모두 `mismatch_count: 0`
  - 권장: `checked_trials` 값도 동일
- fixed trials 계약 키 존재 체크 PASS

## Repo changes (최소 변경 원칙)
M0에서 변경은 최소화하되, 재현성 자동화를 위해 래퍼 1개 추가를 권장한다.

- 추가 파일: `TECH_SPEC_M0.md`
- 권장 추가 파일: `scripts/run_m0_checks.sh`
- 수정 파일: 없음(또는 최소)

## Troubleshooting
1. Tokenizer mismatch
- 증상: prompt/input_ids mismatch 다수 발생
- 확인: `--model_name_for_tokenizer`/`--tokenizer_name`가 fixed trials 메타와 같은지 확인

2. Seed/모드 불일치
- 증상: 재실행 시 결과 변동
- 확인: 실행 래퍼에서 seed 고정, `model.eval()`, `torch.set_grad_enabled(False)` 강제 여부 점검

3. Fixed trials path 오류
- 증상: 파일 없음/키 누락 예외
- 확인: `--fixed_trials_path` 경로 존재, JSON에 `trials`, `clean_prompt_str`, `corrupted_prompt_str`, `prompt_data_*` 키 존재

4. Slot mapping mismatch
- 증상: `idx_map`/`idx_avg` mismatch
- 확인: `--mode`(clean/corrupted)와 prefixes/separators/tokenizer 조합 일치 여부 확인

5. 의존성/환경 문제
- 증상: `transformers` import 또는 tokenizer 로드 실패
- 확인: 가상환경/패키지 설치 상태 및 네트워크/캐시 상태 점검
