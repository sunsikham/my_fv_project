# PT ABC/ABD 9-Shot Workflow

As of `2026-04-01`.

## Goal

Compute shot-wise `PT_ABC` and `PT_ABD` for the 12 analysis questions using:

- shots `1,3,5,7,9`
- all five edges `AB, AC, AD, BC, BD`
- question-wise 5-edge normalization before product-test computation

The final target is a shot-wise table for each `q_id` with:

- `PT_ABC = PT(AB, AC, BC)`
- `PT_ABD = PT(AB, AD, BD)`
- `delta = PT_ABD - PT_ABC`

## Current Inputs

### Existing 11-Q 5-edge run

Canonical root:

- `/scratch/sunsik/my_fv_project/pt_analysis/llama31_70b_pt5edge_q12_topk_srun_20260331_235612_srun`

Properties:

- q_ids: `Q1,Q3,Q4,Q5,Q7,Q8,Q9,Q10,Q11,Q16,Q18`
- shots: `1,3,5,7,10`
- edges: `AB,AC,AD,BC,BD`

Use:

- full 5-edge source for most questions
- not canonical for the final `9-shot` comparison because it uses `10` instead of `9`

### Q6 9-shot recovery run

Canonical root:

- `/scratch/sunsik/my_fv_project/pt_analysis/llama31_70b_pt_q6_9shot_recovery_20260401_014648_srun`

Properties:

- q_ids: `Q6`
- shots: `1,3,5,7,9`
- edges: `AB,AC,AD,BC,BD`

Use:

- canonical 9-shot source for `Q6`

### Q11/Q18 override rerun

Canonical root:

- `/scratch/sunsik/my_fv_project/pt_analysis/llama31_70b_pt_q11_q18_9shot_override_20260401_032904_srun`

Properties:

- q_ids: `Q11,Q18`
- shots: `1,3,5,7,9`
- edges: `AB,AC,AD,BC,BD`
- uses current dataset C targets

Use:

- canonical 9-shot source for `Q11` and `Q18`

## Dataset Overrides

Current C-query source targets in:

- `datasets/relation/icl_C_data.csv`

Relevant overrides:

- `Q11`: `horn -> rhino`
- `Q18`: `squirrel -> nut`

These overrides must be used in all future PT recomputations. Old runs using:

- `Q11`: `horn -> deer`
- `Q18`: `squirrel -> acorn`

should be treated as historical only.

## Important Rule

Do not directly reuse old `target_s_norm` values when shot semantics or target definitions change.

Instead:

1. collect the relevant raw rows
2. use `target_logprob_raw`
3. rebuild question-wise normalization over the intended shot pool and all five edges
4. recompute `PT_ABC` and `PT_ABD` from the rebuilt normalized scores

## Provisional Workflow

Use this first to get an interpretable shot-wise PT table before every question has a true 9-shot rerun.

### Shot policy

- treat `shot=10` from the 11-Q 5-edge run as provisional `shot=9`
- use actual `shot=9` rows for:
  - `Q6`
  - `Q11`
  - `Q18`

### Question sources

- actual 9-shot:
  - `Q6` from the Q6 recovery run
  - `Q11,Q18` from the override rerun
- provisional 9-shot via `10 -> 9` relabel:
  - `Q1,Q3,Q4,Q5,Q7,Q8,Q9,Q10,Q16`

### Normalization rule

For each `q_id`, rebuild normalization from the combined row set using:

- shots `1,3,5,7,9`
- edges `AB,AC,AD,BC,BD`
- `target_logprob_raw`

For provisional questions, the relabeled `10 -> 9` rows are included in the `shot=9` pool.

### PT computation

After normalization:

- compute shot-wise `PT_ABC`
- compute shot-wise `PT_ABD`
- compute shot-wise `delta = PT_ABD - PT_ABC`

### Interpretation rule

This output is useful for:

- figure drafting
- qualitative comparison
- checking whether the overall `ABD > ABC` pattern holds

This output is not canonical final analysis because some questions still use relabeled `10 -> 9`.

## Canonical Workflow

This is the final target.

### Required condition

All 12 questions must have actual 5-edge rows at shots:

- `1,3,5,7,9`

with the intended targets already baked into scoring.

### Canonical recomputation

For all 12 questions:

1. use only true `1,3,5,7,9` rows
2. rebuild question-wise 5-edge normalization from raw logprobs
3. compute shot-wise `PT_ABC`, `PT_ABD`, and `delta`

This replaces the provisional table.

## Why The Extra Rebuild Is Necessary

The old runs already contain all five edges, but that alone is not sufficient.

The remaining issues are:

- the main 11-Q run uses `shot=10` instead of `shot=9`
- `Q11` and `Q18` required C-target changes
- normalization must match the exact shot pool and target definitions used in the final PT comparison

Therefore the safe unit of recomputation is:

- rebuild normalized edge scores from raw edge logprobs
- then compute PT

## Practical Summary

Short version:

1. build a provisional shot-wise PT table now
   - use actual 9-shot rows for `Q6,Q11,Q18`
   - use provisional `10 -> 9` for `Q1,Q3,Q4,Q5,Q7,Q8,Q9,Q10,Q16`
2. later replace provisional rows with real 9-shot reruns
3. rerun normalization and PT once more
4. treat that second table as canonical

