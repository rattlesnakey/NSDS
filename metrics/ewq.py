import os
import json
import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from transformers import AutoModelForCausalLM


def _weight_entropy(weight_matrix, epsilon=0.01):
    w_flat = weight_matrix.view(-1).float()
    size_w = w_flat.numel()
    p = F.softmax(w_flat, dim=0)
    entropy = -torch.sum(p * torch.log(p) + epsilon).item()
    return entropy, size_w


def _block_ewq_score(weight_list, epsilon=0.01):
    total_weighted = 0.0
    total_params = 0

    for W in weight_list:
        if isinstance(W, torch.nn.Parameter):
            W = W.data
        h_wi, size_wi = _weight_entropy(W, epsilon)
        total_weighted += h_wi / size_wi
        total_params += size_wi

    if total_params > 0:
        return total_weighted / total_params
    return 0.0


def _get_layer_weights(layer):
    proj_names = [
        ("self_attn", "q_proj"), ("self_attn", "k_proj"),
        ("self_attn", "v_proj"), ("self_attn", "o_proj"),
        ("mlp", "gate_proj"), ("mlp", "up_proj"), ("mlp", "down_proj"),
    ]
    gpt2_names = [
        ("attn", "c_attn"), ("attn", "c_proj"),
        ("mlp", "c_fc"), ("mlp", "c_proj"),
    ]

    weights = []
    for mod_name, proj_name in proj_names:
        mod = getattr(layer, mod_name, None)
        if mod is not None:
            proj = getattr(mod, proj_name, None)
            if proj is not None and hasattr(proj, "weight"):
                weights.append(proj.weight.detach())

    if not weights:
        for mod_name, proj_name in gpt2_names:
            mod = getattr(layer, mod_name, None)
            if mod is not None:
                proj = getattr(mod, proj_name, None)
                if proj is not None and hasattr(proj, "weight"):
                    weights.append(proj.weight.detach())

    return weights


def compute_ewq(model_path, **kwargs):
    model = AutoModelForCausalLM.from_pretrained(
        model_path, device_map="auto", torch_dtype=torch.float16, trust_remote_code=True
    )

    cfg = model.config
    num_layers = getattr(cfg, "num_hidden_layers", getattr(cfg, "n_layer", 0))
    layers = model.model.layers if hasattr(model, "model") else model.transformer.h

    layer_scores = []

    for l_idx in tqdm(range(num_layers), desc="EWQ"):
        layer = layers[l_idx]
        weights = _get_layer_weights(layer)
        score = _block_ewq_score(weights)
        layer_scores.append(score)

    output_dir = kwargs.get("output_dir", "results")
    model_name = os.path.basename(model_path.rstrip("/"))
    os.makedirs(output_dir, exist_ok=True)
    out_file = os.path.join(output_dir, f"{model_name}_ewq.json")
    result = {"scores": layer_scores}
    with open(out_file, "w") as f:
        json.dump(result, f, indent=2)

    return result
