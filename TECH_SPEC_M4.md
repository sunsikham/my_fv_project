# TECH_SPEC_M4.md

## Scope (M4 only)

### In
- M0~M3 parity 검증을 단일 엔트리포인트로 통합.
- 회귀(regression) 감지를 위한 고정 입력/고정 기준선 기반 harness 구성.
- 실패 시 earliest divergence stage를 즉시 식별 가능한 리포트/로그 표준화.

### Out
- `fv` 코어 알고리즘 신규 변경(M2/M3 로직 재설계)
- 모델 확장(`M5+`)
- relation 실험(`M6+`)

## Goal

- 한 번의 명령으로 parity 전체 상태를 `PASS/FAIL`로 판정.
- 실패 시 어디에서 처음 깨졌는지(stage + 핵심 diff)를 남긴다.

## Suite Composition (minimum required)

M4 parity suite는 아래 단계를 고정 순서로 실행한다.

1. M0: prompt parity
- `scripts/verify_prompt_parity.py`

2. M0: slot parity
- `scripts/verify_slot_parity_against_src.py`

3. M2: FV parity
- `scripts/verify_fv_parity.py`

4. M3: injection parity
- `scripts/verify_injection_parity.py`

실행 순서는 고정:
- `prompt -> slot -> fv -> injection`

## Canonical Inputs and Runtime Contract

- Fixed trials:
- `datasets/fixed_trials/fixed_trials_antonym_t10_s10_seed0.json`
- Golden artifacts (M1 canonical):
- `results/antonym/fixed_trials_antonym_t10_s10_seed0/`
- Runtime contract:
- CPU, float32, `model.eval()`, fixed seed, `torch.set_grad_enabled(False)`.

## Reference Config Normalization Policy

- `src` reference path에서 모델명 분기 누락이 있으면 parity runner에서만 reference config를 정규화할 수 있다.
- GPT-2 parity에서는 reference `src` out_proj를 `c_proj` canonical branch로 강제 매핑한다.
- 이 매핑은 검증 스크립트/스위트 오케스트레이션에만 존재해야 하며 `fv` core semantics를 변경하면 안 된다.

## M4 Runner Design

권장 신규 러너:
- `scripts/run_parity_suite.py`

필수 CLI 인자:
- `--dataset_name` (default: `antonym`)
- `--fixed_trials_path`
- `--fixed_trials_id`
- `--model_name` (default: `gpt2`)
- `--max_trials` (prompt/slot/injection 공통 샘플 수)
- `--n_top_heads` (fv/injection 공통)
- `--edit_layer` (injection)
- `--seed`
- `--fail_fast` (`1`이면 첫 실패에서 중단, `0`이면 끝까지 실행)

권장 실행 커맨드:
```bash
PYTHONPATH=. .venv/bin/python scripts/run_parity_suite.py \
  --dataset_name antonym \
  --fixed_trials_path datasets/fixed_trials/fixed_trials_antonym_t10_s10_seed0.json \
  --fixed_trials_id fixed_trials_antonym_t10_s10_seed0 \
  --model_name gpt2 \
  --max_trials 5 \
  --n_top_heads 10 \
  --edit_layer 9 \
  --seed 42 \
  --fail_fast 0
```

## Stage-level PASS Criteria

1. Prompt parity
- exit code 0
- `mismatch_count: 0`

2. Slot parity
- exit code 0
- `mismatch_count: 0`

3. FV parity
- exit code 0
- `mismatch_count: 0`
- `max_abs_diff: 0.0`

4. Injection parity
- exit code 0
- `mismatch_count: 0`
- clean/with logits diff max가 `0.0`

Suite PASS:
- 4개 stage 모두 PASS
- suite exit code `0`

Suite FAIL:
- 하나라도 FAIL이면 suite exit code non-zero
- earliest failed stage를 summary 최상단에 표기

## Required Outputs (M4)

- `results_fv/antonym/fixed_trials_antonym_t10_s10_seed0/parity_suite_report.json`
- `results_fv/antonym/fixed_trials_antonym_t10_s10_seed0/parity_suite_stages.csv`
- `results_fv/antonym/fixed_trials_antonym_t10_s10_seed0/parity_suite.log`

`parity_suite_report.json` 최소 필드:
- `status` (`PASS`/`FAIL`)
- `dataset_name`, `fixed_trials_id`, `model_name`, `seed`
- `earliest_failed_stage` (`null` if PASS)
- `stages[]`:
- `name`, `cmd`, `exit_code`, `mismatch_count`, `max_abs_diff`, `duration_sec`

## Regression Harness Rules

- 기준선은 fixed trials + canonical golden artifacts 조합으로 고정한다.
- 비교 대상(`results/` golden)은 read-only로 취급한다.
- suite는 매 실행에서 동일 인자/환경이면 동일 결과를 재현해야 한다.
- CI/로컬 모두 같은 단일 명령으로 실행 가능해야 한다.

## DoD (M4)

- `scripts/run_parity_suite.py`가 구현되어 단일 명령으로 4개 stage 실행.
- stage별 결과와 전체 결과가 파일로 저장됨.
- PASS 케이스:
- suite exit code `0`
- `parity_suite_report.json.status == "PASS"`
- FAIL 케이스:
- earliest failed stage와 주요 diff 지표가 리포트에 기록됨.
- 동일 명령 2회 실행 시 결과 일관성 확인.

## Troubleshooting

1. Stage command drift
- Symptom: 개별 검증은 되는데 suite에서만 실패.
- Check: suite가 개별 검증과 동일 CLI 인자를 전달하는지 확인.

2. fail_fast 오해
- Symptom: 첫 실패 후 나머지 stage 리포트가 없음.
- Check: `--fail_fast 0`로 실행해 전체 stage 수집.

3. Artifact path mismatch
- Symptom: FV/injection stage에서 파일 없음.
- Check: `results/<dataset>/<fixed_trials_id>/` canonical 경로 사용 여부 확인.

4. Reference normalization 누락
- Symptom: 특정 모델명에서 src 경로만 실패.
- Check: runner-only reference config normalization 정책 적용 여부 확인.

5. Non-deterministic runtime
- Symptom: 재실행 시 간헐적 mismatch.
- Check: CPU/fp32/eval/no-grad/seed 고정 강제 여부 확인.

## Expected Files Tree (M4)

```text
results_fv/
  antonym/
    fixed_trials_antonym_t10_s10_seed0/
      parity_suite_report.json
      parity_suite_stages.csv
      parity_suite.log
```
