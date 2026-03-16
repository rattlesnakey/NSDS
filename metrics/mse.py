import os
import json
import torch
import numpy as np
from tqdm import tqdm
from transformers import AutoModelForCausalLM


def _quantize_and_mse(W, num_bits=2):
    q_max = (2 ** (num_bits - 1)) - 1
    max_val = torch.max(torch.abs(W))
    if max_val == 0:
        return 0.0
    scale = max_val / (q_max + 1)
    W_quant_int = torch.round(W / scale)
    W_quant_int = torch.clamp(W_quant_int, -q_max, q_max)
    Q_W = W_quant_int * scale
    sensitivity = torch.sum((W - Q_W) ** 2).item()
    return sensitivity


def _get_layer_weight_matrices(layer):
    proj_names = [
        ("self_attn", "q_proj"), ("self_attn", "k_proj"),
        ("self_attn", "v_proj"), ("self_attn", "o_proj"),
        ("mlp", "gate_proj"), ("mlp", "up_proj"), ("mlp", "down_proj"),
    ]
    gpt2_names = [
        ("attn", "c_attn"), ("attn", "c_proj"),
        ("mlp", "c_fc"), ("mlp", "c_proj"),
    ]

    matrices = []
    for mod_name, proj_name in proj_names:
        mod = getattr(layer, mod_name, None)
        if mod is not None:
            proj = getattr(mod, proj_name, None)
            if proj is not None and hasattr(proj, "weight"):
                matrices.append(proj.weight.detach().float())

    if not matrices:
        for mod_name, proj_name in gpt2_names:
            mod = getattr(layer, mod_name, None)
            if mod is not None:
                proj = getattr(mod, proj_name, None)
                if proj is not None and hasattr(proj, "weight"):
                    matrices.append(proj.weight.detach().float())

    return matrices


def compute_mse(model_path, num_bits=2, **kwargs):
    model = AutoModelForCausalLM.from_pretrained(
        model_path, device_map="auto", torch_dtype=torch.float16, trust_remote_code=True
    )

    cfg = model.config
    num_layers = getattr(cfg, "num_hidden_layers", getattr(cfg, "n_layer", 0))
    layers = model.model.layers if hasattr(model, "model") else model.transformer.h

    layer_scores = []

    for l_idx in tqdm(range(num_layers), desc="MSE"):
        layer = layers[l_idx]
        weight_matrices = _get_layer_weight_matrices(layer)

        if not weight_matrices:
            layer_scores.append(0.0)
            continue

        total_mse = sum(_quantize_and_mse(W, num_bits) for W in weight_matrices)
        layer_score = total_mse / len(weight_matrices)
        layer_scores.append(layer_score)

    output_dir = kwargs.get("output_dir", "results")
    model_name = os.path.basename(model_path.rstrip("/"))
    os.makedirs(output_dir, exist_ok=True)
    out_file = os.path.join(output_dir, f"{model_name}_mse.json")
    result = {"scores": layer_scores}
    with open(out_file, "w") as f:
        json.dump(result, f, indent=2)

    return result
