# TECH SPEC M6
0. Scope (M6 only)
In

relation 데이터에서 fv-only 파이프라인을 end-to-end 실행:

Relation trials 준비 → StepD: CIE( qid×head ) 계산 → (옵션) AIE(qid subset 평균) 생성 → StepE: FV 생성 → Step6: FV injection eval(0-shot)

Canonical output 저장 규칙을 results_fv/ 기준으로 표준화

재현 가능한 실행(고정 seed, cpu/fp32, eval/no_grad, 메타/로그) 계약 확정

Out

src parity(M0~M4)

모델 스펙 확장/리팩터(M5)

대규모 성능 최적화

1. Goal

src import 없이 fv/ 경로만으로 relation 실험을 끝까지 실행합니다.

qid별 top head ranking(CIE 기반) 을 항상 보존합니다.

AIE는 “기능만 제공” 하며, 사용자가 지정한 qid subset에 대해서만 생성/집계할 수 있게 합니다.

FV 생성 및 injection eval 결과를 일관된 구조로 저장하여 재현 가능하게 합니다.

2. Definitions (핵심 개념/계약)
2.1 Relation / qid

q_id는 relation의 “과제/관계 타입” 인덱스입니다.

M6에서는 모든 qid에 대해 동일한 trial 수 N_TRIALS_PER_Q 를 목표로 합니다(기본 fail-fast).

2.2 CIE (qid별 head 점수; primary metric = mean_delta_logprob)

CIE는 q_id별로, 각 head (layer, head)에 대해 activation patching 기반 효과를 집계한 값입니다.

M6의 primary score는 mean_delta_logprob 로 고정합니다.

해석: corrupted 대비 patch(또는 intervention) 시 target first-token logprob 개선량의 평균.

2.3 AIE (선택된 qid subset에 대한 CIE 평균)

AIE는 사용자가 선택한 qid 집합 Q* 에 대해 다음과 같이 정의합니다.

AIE_Q*(layer, head) = mean_{q in Q*} CIE(q, layer, head)

기본은 qid-unweighted mean(qid를 동일 가중치로 평균)입니다.

Q*는 런마다 달라질 수 있으므로, AIE 산출물에는 반드시 selected_qids를 기록합니다.

중요: 순식님 정의대로 **AIE는 “qid별 CIE를 head-wise로 평균낸 뒤 그 평균 점수로 top-k를 다시 집계”**하는 용도입니다.

2.4 FV (Function Vector) 생성 정의

FV는 선택된 top-k heads의 “clean mean head output(또는 그에 준하는 평균 활성)”을 residual space로 합성한 벡터입니다.

FV는 residual stream 공간(d_model) 의 벡터이며, injection 시 해당 공간에 단순 덧셈(addition) 으로 주입합니다.

3. Inputs
3.1 relation 입력(기본 + override)

A) relation CSV

필수: q_id, 예시 컬럼(예: exA, exB)

실제 컬럼명은 스크립트 인터페이스로 고정(스펙에서는 “필수 의미”만 보장)

B) fixed_trials JSON (debug/replay override 전용; 기본 경로 아님)

StepD의 기본/정식 샘플링 입력은 relation CSV입니다.

fixed_trials_path는 디버깅/재현 확인을 위한 override 용도로만 허용합니다.

운영 기본 흐름에서 StepD 입력을 fixed_trials로 대체하지 않습니다.

fixed_trials 사용 시에도 trial 스키마는 StepD가 저장하는 sampled_trials.json과 동일 계약을 따릅니다.

최소 포함 정보(개념적):

q_id

clean_prompt_str, corrupted_prompt_str

target_str

target_first_token_id (또는 이를 재계산 가능한 정보)

(선택) demos 구조

3.2 공통 실행 파라미터(고정 계약)

device=cpu, dtype=fp32

seed 고정

tokenizer 정책 고정(특히 add_special_tokens / answer tokenization 규칙)

N_TRIALS_PER_Q 고정(기본 fail-fast)

4. Canonical Output Layout (코드 현실 동기화)

모든 산출물은 results_fv/relation/<run_id>/ 아래로 저장합니다.

results_fv/

relation/

<run_id>/

artifacts/

sampled_trials.json                    # StepD가 생성한 canonical trial snapshot (필수)

cie_scores.csv                         # StepD CIE table (qid x layer x head)

aie_scores.csv                         # StepD AIE table (layer x head)

trial_metrics.jsonl                    # StepD trial-level metrics

stepD_mean_acts/

global_clean_mean.pt                   # dict: clean_mean, slot_q, n_heads, head_dim, resid_dim

qid_<QID>_clean_mean.pt                # dict: clean_mean, slot_q, n_heads, head_dim, resid_dim

mean_activations.pt                    # legacy compat (tensor)

top_heads.json                         # StepE top-k

fv_by_layer.pt                         # StepE artifact

fv_global_resid.pt                     # StepE global FV

fv_global_resid_meta.json              # StepE FV metadata

stepE_eval.json                        # StepE quick eval (snapshot 기반)

step6/

step6.log

step6_results_<model>_layer<L>_n<N>.json

metadata_step6_<model>_layer<L>_n<N>.json

eval_summary.json                      # canonical

eval_trials.jsonl                      # canonical

eval_meta.json                         # canonical

logs/

stepD_aie.log

stepE.log                              # 현재 run 로그는 runs/<auto_run_id>/logs에 기록될 수 있음

4.1 AIE subset 옵션 산출물 패키지(고정 계약)

옵션 AIE(qid subset) 경로는 아래 최소 패키지로 고정합니다.

artifacts/aie/<aie_id>/

selected_qids.json

aie_scores.csv

top_heads.json

fv_meta.json

(옵션) fv_global_resid.pt

aie_id는 selected_qids + 핵심 하이퍼파라미터(score_key, k 등) 해시로 생성합니다.

5. Step Responsibilities (논문 개념 기준 역할 분리)
StepD — CIE 원재료 생성(필수)

역할

relation trials에서 qid×(layer,head) 단위의 CIE를 계산하고 저장합니다.

StepD는 relation CSV에서 N_TRIALS_PER_Q, N_DEMOS, seed로 trial을 샘플링하고, 해당 샘플을 run 스냅샷으로 고정 저장합니다.

FV 생성을 위해 qid별 clean mean activations(또는 clean mean head outputs에 준하는 통계량) 을 저장합니다.

필수 산출물

cie_scores.csv (qid별 점수표; 스키마 고정)

sampled_trials.json (필수; StepE/Step6 재사용용 canonical snapshot)

stepD_mean_acts/qid_<QID>_clean_mean.pt (qid별 mean acts)

CIE CSV 스키마(고정)

q_id : string

layer : int

head : int

n_trials : int

mean_delta_logprob : float (primary)

mean_delta_p : float (optional; 있으면 저장)

정렬/동률 규칙(결정적 타이브레이크)

primary: mean_delta_logprob 내림차순

tie-break: (layer 오름차순, head 오름차순) 고정

StepE — FV 생성(필수) + quick eval(옵션)

현재 코드 기준 StepE는 아래를 수행합니다.

입력: sampled_trials.json + StepD score/mean artifacts

동작: score_key 기반 top-k 선택 후 fv_by_layer.pt / fv_global_resid.pt 생성

옵션: sampled_trials 기반 quick eval(stepE_eval.json)

Step6 — FV Injection Eval (필수; 0-shot 주입)

핵심 전제

순식님 말씀대로 FV injection은 0-shot(데모 없는 query-only)에서 주입합니다.

입력

FV: (a) fv_qid.pt 또는 (b) fv_global_resid.pt

sampled_trials.json (StepD에서 저장한 동일 trial snapshot; 재샘플링 금지)

0-shot evaluation prompts: query-only 형태로 구성(예: “Q: … A:”)

injection policy: layer(s), alpha, token position

eval_scope: {in_domain, all_qids, qid_list}

FV 입력 모드별 기본 eval_scope(고정)

FV=fv_qid.pt 인 경우 기본값은 in_domain

FV=fv_global_resid.pt 인 경우 기본값은 all_qids

qid_list 모드에서는 사용자가 평가할 qid 집합을 명시해야 함

Injection 정의(고정)

모델의 residual stream (transformer block output)

query prompt의 마지막 토큰 위치(예: answer를 예측하는 직전 위치)

지정한 layer ℓ에서:

h_ℓ[pos] := h_ℓ[pos] + alpha * FV

layer sweep이 켜져 있으면, 지정된 layer 집합에 대해 동일 규칙을 적용해 각각 평가합니다.

평가지표(필수)

mean_delta_logprob (primary): target first-token logprob 변화량

accuracy_delta (optional but 권장)

(선택) mean_delta_p

산출물

artifacts/step6/eval_summary.json : layer/alpha별 요약

artifacts/step6/eval_trials.jsonl : trial-level (qid, layer, alpha, baseline/injected logprob 등)

artifacts/step6/eval_meta.json : 설정/seed/데이터 스냅샷

eval_scope=all_qids 일 때 요약은 최소한 (fv_qid, eval_qid) 축을 구분해야 함

(권장) eval_summary는 matrix[fv_qid][eval_qid] 또는 long-format 동등 정보로 저장

eval_meta.json 필수 키: eval_scope, eval_qids, (fv_qid 입력 시) fv_source_qid

StepE/Step6 공통 trial 계약(필수)

StepE와 Step6는 반드시 StepD가 생성한 sampled_trials.json을 입력으로 사용합니다.

relation CSV를 StepE/Step6에서 다시 읽어 trial을 재샘플링하는 동작은 금지합니다.

5.1 CLI Contracts (필수)

StepD:

--relation_csv_path --relation_n_trials_per_q --relation_n_demos --seed --out_base_dir

(현행 추가) --layers --heads --score_key --model --model_spec --device --dtype

(legacy 허용) --out_dir, --fixed_out_dir

StepE:

--sampled_trials_path <BASE>/artifacts/sampled_trials.json

(또는 --fixed_trials_path를 sampled_trials.json 입력 용도로 재사용)

--stepd_base_dir <BASE> (권장) 또는 --stepd_artifacts_dir <BASE>/artifacts

(legacy 허용) --run_id_stepD

--score_key (legacy alias: --metric)

Step6:

--sampled_trials_path <BASE>/artifacts/sampled_trials.json

(또는 --fixed_trials_path를 sampled_trials.json 입력 용도로 재사용)

--eval_scope {auto,in_domain,all_qids,qid_list}

--eval_qids <comma-separated> (eval_scope=qid_list일 때)

--fv_source_qid <QID> (qid-specific FV + in_domain에서 필요)

6. Defaults (M6 baseline)

compute:

cpu, fp32, eval, no_grad

seed 고정

data:

N_TRIALS_PER_Q 고정

qid list 고정(또는 fixed_trials로 고정)

StepE/Step6는 StepD sampled_trials.json 재사용(재샘플링 금지)

score:

primary score = mean_delta_logprob 고정

FV:

기본 워크플로: CIE 기반 qid별 FV 생성

AIE 기반 FV는 옵션 워크플로(qid subset을 명시할 때만 수행)

injection:

0-shot만 수행

layer sweep 기본 = all (비용 문제 시 selected layers 옵션 허용)

alpha 기본 = 1.0 (추가 sweep은 옵션)

7. Determinism & Drop Policy
7.1 Determinism rules (필수)

cpu/fp32 강제

model.eval() + no_grad 강제

seed 고정(파이썬/넘파이/토치 포함)

head ranking tie-break 규칙 고정(§5 StepD)

7.2 Drop policy (필수; 기본 fail-fast)

기본: 어떤 qid에서든 trial 생성/토크나이즈 실패로 목표 N_TRIALS_PER_Q를 못 채우면 즉시 실패

옵션(allow_drop)을 지원할 수 있으나, M6 baseline에서는 비권장

사용 시 sampled_trials.json meta 및 step6/eval_meta.json에 qid별 실제 사용 집합을 기록

AIE는 여전히 qid-unweighted mean으로 고정(단, “남은 qid 집합” 기준)

8. DoD (M6)

fv-only로 relation에서 end-to-end 완료:

StepD(CIE+mean acts) → StepE(FV global + optional quick eval) → Step6(0-shot injection eval)

필수 산출물이 모두 생성:

cie_scores.csv (스키마 준수)

qid_<QID>_clean_mean.pt(필요 qid 모두)

StepE FV 산출물(fv_by_layer.pt, fv_global_resid.pt, fv_global_resid_meta.json)

Step6 summary + trial-level jsonl + meta

logs(stepD_aie.log, stepE.log, step6/step6.log)

StepD가 생성한 sampled_trials.json을 StepE/Step6가 재사용하여 동일 trials 기준으로 E2E 완료

재현성:

동일 설정 2회 실행 시 mean_delta_logprob의 abs diff ≤ 1e-6 (요약 지표 기준)

top-k head 리스트 동일(동률 tie-break 규칙 때문에 결정적으로 같아야 함)

9. Re-Run Templates (확정 커맨드)

아래 3개를 그대로 사용하면 StepD→StepE→Step6가 같은 snapshot을 재사용합니다.

공통:

RUN_ID=<원하는_run_id>

BASE=results_fv/relation/${RUN_ID}

StepD:

python scripts/run_stepD_aie_head_sweep.py \
  --model gpt2 \
  --model_spec gpt2 \
  --device cpu \
  --dtype fp32 \
  --quant none \
  --layers 0 \
  --heads 0 \
  --relation_csv_path datasets/relation/relationA_ex.csv \
  --relation_n_trials_per_q 1 \
  --relation_n_demos 1 \
  --seed 42 \
  --out_base_dir "${BASE}" \
  --score_key mean_delta_logprob \
  --n_mean_trials 1 \
  --n_trials 1

StepE:

python scripts/run_stepE_topk_fv_and_eval.py \
  --stepd_base_dir "${BASE}" \
  --sampled_trials_path "${BASE}/artifacts/sampled_trials.json" \
  --k 1 \
  --model gpt2 \
  --model_spec gpt2 \
  --device cpu \
  --dtype fp32 \
  --quant none \
  --layers 0 \
  --alpha 1.0 \
  --n_eval_trials 4 \
  --seed 42 \
  --out_dir "${BASE}/artifacts"

Step6:

python scripts/run_step6_fv_injection_eval.py \
  --model gpt2 \
  --model_spec gpt2 \
  --device cpu \
  --dtype fp32 \
  --quant none \
  --edit_layer 0 \
  --fv_global_path "${BASE}/artifacts/fv_global_resid.pt" \
  --fv_global_meta_path "${BASE}/artifacts/fv_global_resid_meta.json" \
  --sampled_trials_path "${BASE}/artifacts/sampled_trials.json" \
  --eval_scope all_qids \
  --n_eval 4 \
  --seed 42 \
  --out_dir "${BASE}/artifacts/step6"

10. Known Gaps / TODO (코드 현실 기준)

StepE run 로그(`stepE.log`)는 현재 `runs/<auto_run_id>/logs`로 기록될 수 있습니다.

`results_fv/relation/<run_id>/logs` 단일화는 후속 정리 항목입니다.

AIE qid-subset 전용 패키지(`artifacts/aie/<aie_id>/...`)는 계약은 고정했지만, 현재 기본 파이프라인 구현은 root artifacts 중심입니다.

필요 시 AIE subset 전용 경로 생성기를 별도 스크립트/옵션으로 추가합니다.

11. Troubleshooting (대표 실패 모드)

relation schema mismatch: q_id/예시 컬럼 누락

target boundary mismatch: target first-token id 계산 정책 불일치

hook mismatch: model_spec에 없는 layer hook name

artifact 혼재: run_id/aie_id/eval_id 충돌 또는 덮어쓰기

비결정성: eval/no_grad/seed/cpu-fp32 강제 누락
