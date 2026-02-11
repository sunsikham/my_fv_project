# Relation Q-wise FV 런타임 디버깅 플랜

## 목표
- `relation_qwise` 파이프라인( `StepD -> StepE -> Step6` )을 한 번에 실행했을 때 발생하는 오류를 재현하고, `src` 기준 FV 생성/주입 로직과 어디서 달라지는지 단계별로 식별한다.
- 최종적으로 "어느 단계에서, 어떤 입력/아티팩트/토큰 규칙이 달라졌는지"를 증거 파일 기준으로 정리한다.

## 범위
- 오케스트레이터: `scripts/run_relation_qwise_pipeline.py`
- StepD: `scripts/run_stepD_aie_head_sweep.py`
- StepE: `scripts/run_stepE_topk_fv_and_eval.py`
- Step6: `scripts/run_step6_fv_injection_eval.py`
- 비교 기준(src): `src/utils/extract_utils.py`, `src/utils/intervention_utils.py`, `src/utils/eval_utils.py`

## 핵심 차이 가설(우선 점검)
- 프롬프트/토크나이저 경계: `tok_add_special`, BOS, `target_first_token_id` 재계산 규칙
- mean activation 슬롯 정렬: `dummy_labels`, `idx_map`, `slot_index_map[QUERY_PRED]`
- FV 합성 방식: `fv_by_layer`(head slot 합) vs `fv_global_resid`(out_proj 통과 합)
- 주입 위치: block output residual hook(레이어 출력) 기준 일치 여부
- 평가 샘플링/스코프: q-wise 범위, `eval_scope`, `n_eval`, seed
- dtype/quant/device_map 영향: fp16/4bit 경로에서 수치/shape mismatch

## 사전 고정값(재현성)
- 동일 모델/스펙/seed 고정
- `--q_list`로 단일 qid부터 재현
- 최초 재현은 `--stop_on_error` 사용
- 실패 로그/아티팩트가 남도록 기존 결과 디렉토리 보존

## 단계별 실행 플랜

### 1) 실패 케이스 고정 및 1차 재현
1. 단일 qid, 단일 실행으로 실패를 재현한다.
2. 오케스트레이터 상태 파일(`qid_status.json`)과 레이어별 로그를 확보한다.

권장 실행 템플릿:
```bash
PY=${PY:-python}
$PY scripts/run_relation_qwise_pipeline.py \
  --model <MODEL> \
  --model_spec <MODEL_SPEC> \
  --relation_csv_path <RELATION_CSV> \
  --relation_name <RELATION_NAME> \
  --q_list <QID> \
  --out_root results_fv/relation_qwise \
  --seed <SEED> \
  --stop_on_error
```

수집 파일:
- `results_fv/relation_qwise/<RELATION_NAME>/<QID>/artifacts/qid_status.json`
- `results_fv/relation_qwise/<RELATION_NAME>/<QID>/logs/stepD_runner.log`
- `results_fv/relation_qwise/<RELATION_NAME>/<QID>/logs/stepE_runner.log`
- `results_fv/relation_qwise/<RELATION_NAME>/<QID>/logs/step6_layer_<LAYER>.log`

### 2) StepD 입력/슬롯/mean_activations 검증
1. `sampled_trials.json`에서 `target_first_token_id`, `corrupted_prompt_str`, `target_str`가 일관적인지 확인한다.
2. `mean_activations_meta.json`의 `slot_query_pred`, `dummy_labels`, `slot_index_map`를 확인한다.
3. 필요한 경우 StepD 단독 재실행으로 mean만 생성(`--mean_only 1`)해 shape와 슬롯을 고정한다.

권장 점검 명령:
```bash
PY=${PY:-python}
BASE=results_fv/relation_qwise/<RELATION_NAME>/<QID>

$PY scripts/run_stepD_aie_head_sweep.py \
  --model <MODEL> --model_spec <MODEL_SPEC> \
  --relation_csv_path <RELATION_CSV> \
  --relation_q_list <QID> \
  --relation_n_trials_per_q <N_TRIALS> \
  --relation_n_demos <N_DEMOS> \
  --seed <SEED> \
  --layers all --heads all \
  --out_base_dir "$BASE" \
  --mean_only 1
```

확인 포인트:
- `artifacts/mean_activations.pt` shape
- `artifacts/mean_activations_meta.json`의 `slot_query_pred` 존재
- `artifacts/sampled_trials.json` trial 수/필드 누락 여부

### 3) StepE FV 합성 검증
1. `aie_scores.csv` 기준 top-k head 선택이 의도한 `score_key`로 정렬되는지 확인한다.
2. `fv_by_layer.pt`와 `fv_global_resid.pt`의 dim 및 norm 확인.
3. StepD mean dim과 모델 dim 불일치로 실패하는지 확인한다.

권장 점검 명령:
```bash
PY=${PY:-python}
BASE=results_fv/relation_qwise/<RELATION_NAME>/<QID>

$PY scripts/run_stepE_topk_fv_and_eval.py \
  --stepd_artifacts_dir "$BASE/artifacts" \
  --sampled_trials_path "$BASE/artifacts/sampled_trials.json" \
  --model <MODEL> --model_spec <MODEL_SPEC> \
  --k <TOPK> --score_key <SCORE_KEY> \
  --alpha <ALPHA> \
  --out_dir "$BASE/artifacts"
```

확인 파일:
- `artifacts/top_heads.json`
- `artifacts/fv_by_layer.pt`
- `artifacts/fv_global_resid.pt`
- `artifacts/fv_global_resid_meta.json`
- `artifacts/stepE_eval.json`

### 4) Step6 주입/타겟 토큰 경계 검증
1. Step6 단독 실행으로 `target_id` 재계산 mismatch가 나는지 확인한다.
2. hook 호출 횟수(`residual hook calls`)와 `edit_layer` 범위 오류를 확인한다.
3. `eval_scope`/`eval_qids`가 의도와 다르지 않은지 확인한다.

권장 점검 명령:
```bash
PY=${PY:-python}
BASE=results_fv/relation_qwise/<RELATION_NAME>/<QID>

$PY scripts/run_step6_fv_injection_eval.py \
  --model <MODEL> --model_spec <MODEL_SPEC> \
  --fv_global_path "$BASE/artifacts/fv_global_resid.pt" \
  --fv_global_meta_path "$BASE/artifacts/fv_global_resid_meta.json" \
  --sampled_trials_path "$BASE/artifacts/sampled_trials_zeroshot.json" \
  --edit_layer <LAYER> \
  --alpha <ALPHA> --n_eval <N_EVAL> \
  --score_key <SCORE_KEY> \
  --out_dir "$BASE/artifacts/step6/layer_<LAYER>"
```

실패 시 우선 확인:
- `step6_target_id_debug_*.json` 생성 여부
- `token_rule`, `tok_add_special`, `prefix_ids/full_ids` 경계

### 5) src 대비 패리티 보조 검증(원인 분리)
- relation_qwise 자체 오류가 아니라 코어 로직 편차인지 분리하기 위해 parity 스크립트를 병행 실행한다.

권장 명령:
```bash
PY=${PY:-python}
$PY scripts/run_parity_suite.py \
  --dataset_name antonym \
  --fixed_trials_path datasets/fixed_trials/fixed_trials_antonym_t10_s10_seed0.json \
  --model_name gpt2 \
  --device cpu
```

목적:
- 코어 parity가 이미 깨져 있으면 relation_qwise 이전에 src/fv 차이부터 수정
- 코어 parity가 통과하면 relation_qwise 데이터/토큰 경계 문제에 집중

### 6) 최종 차이 리포트 작성
- 아래 형식으로 차이 요약 파일 작성:
  - 실패 단계: StepD/StepE/Step6
  - 첫 실패 시점: 파일/로그 라인
  - 원인 분류: token boundary / slot mapping / fv synthesis / injection / config
  - 재현 명령: 1줄
  - 수정 후보: 1~3개

## 완료 기준(Definition of Done)
- 단일 qid 기준으로 실패를 100% 재현 가능한 명령 확보
- 첫 divergence 지점이 아티팩트/로그로 특정됨
- "src 대비 무엇이 다른지"가 최소 1개 이상 증거와 함께 문서화됨
- 수정 전/후 검증 커맨드가 준비됨

## 빠른 실행 순서(요약)
1. 오케스트레이터 단일 qid 재현
2. StepD mean/slot 메타 확인
3. StepE FV 아티팩트 dim/norm 확인
4. Step6 target_id 경계/주입 로그 확인
5. parity suite로 코어 편차 여부 분리
6. 차이 리포트 작성
