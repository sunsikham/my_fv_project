"""Head-output vector extraction for condition-wise PCA/FV analysis."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Set, Tuple


Head = Tuple[int, int]


@dataclass
class TopHeadRow:
    layer: int
    head: int
    score: float
    score_key: str


def _to_head_tuple(entry: object) -> Head:
    if isinstance(entry, dict):
        return (int(entry["layer"]), int(entry["head"]))
    if isinstance(entry, (tuple, list)) and len(entry) >= 2:
        return (int(entry[0]), int(entry[1]))
    raise ValueError(f"Unsupported head entry: {entry!r}")


def unique_heads(entries: Iterable[object]) -> List[Head]:
    seen: Set[Head] = set()
    out: List[Head] = []
    for entry in entries:
        head = _to_head_tuple(entry)
        if head in seen:
            continue
        seen.add(head)
        out.append(head)
    out.sort(key=lambda x: (x[0], x[1]))
    return out


def serialize_heads(heads: Sequence[Head]) -> List[Dict[str, int]]:
    return [{"layer": int(layer), "head": int(head)} for layer, head in heads]


def load_stepd_scores_csv(path: str, score_key: str) -> List[TopHeadRow]:
    rows: List[TopHeadRow] = []
    with open(path, "r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"CSV missing header row: {path}")
        if "layer" not in reader.fieldnames or "head" not in reader.fieldnames:
            raise ValueError(f"CSV must include layer/head columns: {path}")
        if score_key not in reader.fieldnames:
            raise ValueError(
                f"CSV missing score_key='{score_key}' "
                f"(available={reader.fieldnames}) path={path}"
            )
        for row in reader:
            raw_score = row.get(score_key)
            if raw_score is None:
                raise ValueError(
                    f"Missing score value score_key='{score_key}' "
                    f"layer={row.get('layer')} head={row.get('head')} path={path}"
                )
            score = float(raw_score)
            if score != score:  # NaN check
                raise ValueError(
                    f"NaN score value score_key='{score_key}' "
                    f"layer={row.get('layer')} head={row.get('head')} path={path}"
                )
            rows.append(
                TopHeadRow(
                    layer=int(row["layer"]),
                    head=int(row["head"]),
                    score=score,
                    score_key=score_key,
                )
            )
    if not rows:
        raise ValueError(f"No rows loaded from StepD score CSV: {path}")
    return rows


def select_topk(rows: Sequence[TopHeadRow], k: int) -> List[TopHeadRow]:
    if k < 1:
        raise ValueError("top-k must be >= 1")
    sorted_rows = sorted(rows, key=lambda r: (-r.score, r.layer, r.head))
    return sorted_rows[:k]


def build_rank_map(rows: Sequence[TopHeadRow]) -> Dict[Head, int]:
    sorted_rows = sorted(rows, key=lambda r: (-r.score, r.layer, r.head))
    rank_map: Dict[Head, int] = {}
    for idx, row in enumerate(sorted_rows, start=1):
        key = (row.layer, row.head)
        if key not in rank_map:
            rank_map[key] = idx
    return rank_map


def jaccard(a: Set[Head], b: Set[Head]) -> float:
    union = a | b
    if not union:
        return 1.0
    return float(len(a & b) / len(union))


def spearman_rank_correlation(rank_a: Dict[Head, int], rank_b: Dict[Head, int]) -> float:
    common = sorted(set(rank_a.keys()) & set(rank_b.keys()))
    n = len(common)
    if n < 2:
        return 0.0
    vals_a = [float(rank_a[key]) for key in common]
    vals_b = [float(rank_b[key]) for key in common]
    mean_a = sum(vals_a) / n
    mean_b = sum(vals_b) / n
    num = sum((a - mean_a) * (b - mean_b) for a, b in zip(vals_a, vals_b))
    den_a = sum((a - mean_a) ** 2 for a in vals_a) ** 0.5
    den_b = sum((b - mean_b) ** 2 for b in vals_b) ** 0.5
    if den_a == 0.0 or den_b == 0.0:
        return 0.0
    return float(num / (den_a * den_b))


def prepare_weight_slices(
    *,
    layer_modules: Dict[int, object],
    capture_heads: Sequence[Head],
    head_dim: int,
    logger: Optional[Callable[[str], None]] = None,
) -> Dict[Head, object]:
    import torch

    heads_by_layer: Dict[int, List[int]] = {}
    for layer, head in capture_heads:
        heads_by_layer.setdefault(int(layer), []).append(int(head))

    weights: Dict[Head, object] = {}
    for layer, head_list in sorted(heads_by_layer.items()):
        module = layer_modules[layer]
        weight = getattr(module, "weight", None)
        if weight is None:
            raise ValueError(f"Layer {layer} out_proj module has no weight")
        if logger is not None:
            logger(f"[vector] preparing weight slices layer={layer} n_heads={len(head_list)}")
        if hasattr(weight, "quant_state"):
            try:
                import bitsandbytes as bnb  # type: ignore
            except Exception as exc:  # pragma: no cover
                raise RuntimeError(
                    "bitsandbytes is required to dequantize 4bit out_proj weights"
                ) from exc
            with torch.no_grad():
                w_full = bnb.functional.dequantize_4bit(weight.data, weight.quant_state)
        else:
            with torch.no_grad():
                w_full = weight.detach()
        w_full = w_full.to(dtype=torch.float32, device="cpu")
        for head in sorted(set(head_list)):
            start = int(head) * head_dim
            end = start + head_dim
            if end > w_full.shape[1]:
                raise ValueError(
                    f"Head slice out of range layer={layer} head={head} "
                    f"head_dim={head_dim} in_dim={w_full.shape[1]}"
                )
            weights[(layer, int(head))] = w_full[:, start:end].contiguous().clone()
        del w_full
    return weights


def extract_condition_trial_vectors(
    *,
    model,
    tokenizer,
    tok_add_special: bool,
    layer_modules: Dict[int, object],
    n_heads: int,
    head_dim: int,
    resid_dim: int,
    trials: Sequence[Dict[str, object]],
    capture_heads: Sequence[Head],
    aaa_ref_heads: Set[Head],
    union_ref_heads: Optional[Set[Head]],
    cond_topk_heads: Set[Head],
    logger: Optional[Callable[[str], None]] = None,
) -> Dict[str, object]:
    import numpy as np
    import torch

    capture_heads = unique_heads(capture_heads)
    capture_pos = {head: idx for idx, head in enumerate(capture_heads)}
    heads_by_layer: Dict[int, List[int]] = {}
    for layer, head in capture_heads:
        heads_by_layer.setdefault(layer, []).append(head)

    for layer, head_list in heads_by_layer.items():
        if layer not in layer_modules:
            raise ValueError(f"Layer {layer} missing from layer_modules")
        for head in head_list:
            if head < 0 or head >= n_heads:
                raise ValueError(f"Invalid head index layer={layer} head={head} n_heads={n_heads}")

    weight_slices = prepare_weight_slices(
        layer_modules=layer_modules,
        capture_heads=capture_heads,
        head_dim=head_dim,
        logger=logger,
    )

    capture_state: Dict[int, object] = {}

    def _make_hook(layer_idx: int):
        def _hook(_module, inputs):
            if not inputs:
                return
            x = inputs[0]
            if x is None:
                return
            capture_state[layer_idx] = x.detach()

        return _hook

    handles = []
    for layer in sorted(heads_by_layer.keys()):
        handles.append(layer_modules[layer].register_forward_pre_hook(_make_hook(layer)))

    try:
        aaa_vecs: List[np.ndarray] = []
        union_vecs: List[np.ndarray] = []
        cond_vecs: List[np.ndarray] = []
        capture_headwise_vecs: List[np.ndarray] = []
        trial_ids: List[str] = []
        seq_token_indices: List[int] = []

        try:
            model_device = next(model.parameters()).device
        except StopIteration:
            model_device = torch.device("cpu")

        for idx, trial in enumerate(trials):
            prefix = trial.get("corrupted_prompt_str") or trial.get("corrupted_prefix_str")
            if not isinstance(prefix, str):
                raise ValueError(f"trial[{idx}] missing corrupted prompt string")
            trial_id = str(trial.get("trial_id", f"t{idx:06d}"))
            inputs = tokenizer(
                prefix,
                return_tensors="pt",
                add_special_tokens=tok_add_special,
            )
            inputs = {key: value.to(model_device) for key, value in inputs.items()}
            seq_token_idx = int(inputs["input_ids"].shape[1] - 1)
            capture_state.clear()
            with torch.inference_mode():
                _ = model(**inputs)

            sum_aaa = torch.zeros((resid_dim,), dtype=torch.float32)
            sum_union = torch.zeros((resid_dim,), dtype=torch.float32)
            sum_cond = torch.zeros((resid_dim,), dtype=torch.float32)
            capture_headwise = torch.zeros((len(capture_heads), resid_dim), dtype=torch.float32)

            for layer in sorted(heads_by_layer.keys()):
                x = capture_state.get(layer)
                if x is None:
                    raise ValueError(f"Missing captured activation for layer={layer}")
                if x.dim() != 3 or x.shape[0] != 1 or x.shape[-1] != resid_dim:
                    raise ValueError(
                        f"Unexpected activation shape layer={layer} shape={tuple(x.shape)}"
                    )
                if seq_token_idx < 0 or seq_token_idx >= x.shape[1]:
                    raise ValueError(
                        f"seq_token_idx out of range layer={layer} "
                        f"seq_token_idx={seq_token_idx} seq_len={x.shape[1]}"
                    )
                token_vec = x[0, seq_token_idx, :].reshape(n_heads, head_dim).to(
                    dtype=torch.float32,
                    device="cpu",
                )
                for head in heads_by_layer[layer]:
                    key = (layer, head)
                    w_sub = weight_slices[key]
                    contrib = torch.mv(w_sub, token_vec[head])
                    capture_headwise[capture_pos[key], :] = contrib
                    if key in aaa_ref_heads:
                        sum_aaa += contrib
                    if union_ref_heads is not None and key in union_ref_heads:
                        sum_union += contrib
                    if key in cond_topk_heads:
                        sum_cond += contrib

            aaa_vecs.append(sum_aaa.numpy().astype(np.float16, copy=False))
            if union_ref_heads is not None:
                union_vecs.append(sum_union.numpy().astype(np.float16, copy=False))
            cond_vecs.append(sum_cond.numpy().astype(np.float16, copy=False))
            capture_headwise_vecs.append(capture_headwise.numpy().astype(np.float16, copy=False))
            trial_ids.append(trial_id)
            seq_token_indices.append(seq_token_idx)

            if logger is not None and ((idx + 1) % 10 == 0 or (idx + 1) == len(trials)):
                logger(f"[vector] processed {idx + 1}/{len(trials)} trials")

        result = {
            "aaa_ref": np.stack(aaa_vecs, axis=0) if aaa_vecs else np.zeros((0, resid_dim), dtype=np.float16),
            "cond_topk": np.stack(cond_vecs, axis=0)
            if cond_vecs
            else np.zeros((0, resid_dim), dtype=np.float16),
            "capture_headwise": np.stack(capture_headwise_vecs, axis=0)
            if capture_headwise_vecs
            else np.zeros((0, len(capture_heads), resid_dim), dtype=np.float16),
            "capture_heads": capture_heads,
            "trial_ids": trial_ids,
            "seq_token_indices": seq_token_indices,
        }
        if union_ref_heads is not None:
            result["union_ref"] = (
                np.stack(union_vecs, axis=0)
                if union_vecs
                else np.zeros((0, resid_dim), dtype=np.float16)
            )
        else:
            result["union_ref"] = None
        return result
    finally:
        for handle in handles:
            handle.remove()
