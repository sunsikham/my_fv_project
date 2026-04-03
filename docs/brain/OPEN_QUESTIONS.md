# Open Questions

## Documentation Gaps

### 1. README Scope

`README.md` now points to `docs/brain/`, but it still mostly reflects early-project usage examples.

Needed action:
- decide whether README should stay minimal
- or be rewritten as a light quickstart + link hub

### 2. Canonical vs Mirror Coverage

The current rules for home vs scratch are documented, and the main runners now record storage metadata.
However, some historical runs and lower-level stage outputs still require manual judgment.

Needed action:
- audit older runs and any remaining stage-level producers that still omit storage fields

### 3. Multi-Feature Source Compression

`docs/multi_feature_reweighting/` is still a source corpus, not yet a fully condensed brain layer.

Needed action:
- produce one stable theory summary
- produce one stable methods summary
- leave detailed notes as source/archive

## Project Questions

### 4. Super-FV Status

`super FV` has both a live tech spec and archived planning material.

Needed action:
- decide whether super-FV is still an active first-class track
- or whether it should be documented as a secondary branch

### 5. Repo-Local PT Completeness

Repo-local `pt_analysis/` appears partial compared with scratch.

Needed action:
- decide whether repo-local PT should remain partial by design
- or whether only scratch should be treated as the real PT root

### 6. Analysis Root Conventions

Condition-qwise analysis roots under scratch are useful but not yet described by a single naming convention doc.

Needed action:
- document `_analysis_stepwise_*`, `_analysis_multi_root`, `_analysis_state_intervention`, and `_analysis_local_tangent_curvature` as a formal family

## Suggested Next Pass

If continuing second-brain work, the best next steps are:
- condense multi-feature notes into one stable narrative doc
- audit older artifacts and remaining stage-level metadata coverage
