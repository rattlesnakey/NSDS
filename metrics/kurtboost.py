import os
import json
import numpy as np
from scipy.stats import kurtosis
from tqdm import tqdm
from transformers import AutoModelForCausalLM
import torch


def _matrix_kurtosis(weight_matrix):
    w = weight_matrix.detach().cpu().float().numpy().flatten()
    return float(kurtosis(w, axis=None, fisher=True, bias=True, nan_policy="omit"))


def _get_layer_kurtosis(layer):
    proj_names = [
        ("self_attn", "q_proj"), ("self_attn", "k_proj"),
        ("self_attn", "v_proj"), ("self_attn", "o_proj"),
        ("mlp", "gate_proj"), ("mlp", "up_proj"), ("mlp", "down_proj"),
    ]
    gpt2_names = [
        ("attn", "c_attn"), ("attn", "c_proj"),
        ("mlp", "c_fc"), ("mlp", "c_proj"),
    ]

    kurt_vals = []
    for mod_name, proj_name in proj_names:
        mod = getattr(layer, mod_name, None)
        if mod is not None:
            proj = getattr(mod, proj_name, None)
            if proj is not None and hasattr(proj, "weight"):
                kurt_vals.append(_matrix_kurtosis(proj.weight))

    if not kurt_vals:
        for mod_name, proj_name in gpt2_names:
            mod = getattr(layer, mod_name, None)
            if mod is not None:
                proj = getattr(mod, proj_name, None)
                if proj is not None and hasattr(proj, "weight"):
                    kurt_vals.append(_matrix_kurtosis(proj.weight))

    return float(np.sum(kurt_vals) / len(kurt_vals)) if kurt_vals else 0.0


def _compute_difference_sequence(scores):
    diffs = [scores[l + 1] - scores[l] for l in range(len(scores) - 1)]
    if not diffs:
        return [0.0], [False]
    mu = float(np.mean(diffs))
    sigma = float(np.std(diffs)) + 1e-12
    z_diffs = [abs(d - mu) / sigma for d in diffs]
    outliers = [z > 2 for z in z_diffs]
    outliers = [False] + outliers
    return z_diffs, outliers


def compute_kurtboost(model_path, **kwargs):
    model = AutoModelForCausalLM.from_pretrained(
        model_path, device_map="auto", torch_dtype=torch.float16, trust_remote_code=True
    )

    cfg = model.config
    num_layers = getattr(cfg, "num_hidden_layers", getattr(cfg, "n_layer", 0))
    layers = model.model.layers if hasattr(model, "model") else model.transformer.h

    raw_scores = []
    for l_idx in tqdm(range(num_layers), desc="KurtBoost"):
        raw_scores.append(_get_layer_kurtosis(layers[l_idx]))

    _, outlier_flags = _compute_difference_sequence(raw_scores)

    layer_scores = []
    for l_idx in range(num_layers):
        boost = 2.0 if l_idx < len(outlier_flags) and outlier_flags[l_idx] else 1.0
        layer_scores.append(raw_scores[l_idx] * boost)

    output_dir = kwargs.get("output_dir", "results")
    model_name = os.path.basename(model_path.rstrip("/"))
    os.makedirs(output_dir, exist_ok=True)
    out_file = os.path.join(output_dir, f"{model_name}_kurtboost.json")
    result = {"scores": layer_scores, "raw_kurtosis": raw_scores}
    with open(out_file, "w") as f:
        json.dump(result, f, indent=2)

    return result
