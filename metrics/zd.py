import os
import json
import torch
import numpy as np
from tqdm import tqdm
from transformers import AutoModelForCausalLM


def _zd_score_for_matrix(weight_matrix, threshold=1.0):
    weights = weight_matrix.view(-1).float()
    mu = weights.mean()
    sigma = weights.std(unbiased=True)
    if sigma.item() == 0:
        return 0.0
    z_scores = (weights - mu) / sigma
    ratio = (torch.abs(z_scores) > threshold).float().mean()
    return ratio.item()


def compute_zd(model_path, threshold=1.0, **kwargs):
    model = AutoModelForCausalLM.from_pretrained(
        model_path, device_map="auto", torch_dtype=torch.float16, trust_remote_code=True
    )

    cfg = model.config
    num_layers = getattr(cfg, "num_hidden_layers", getattr(cfg, "n_layer", 0))

    layer_scores = []

    proj_names_llama = [
        "self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj",
        "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj",
    ]
    proj_names_gpt2 = [
        "attn.c_attn", "attn.c_proj", "mlp.c_fc", "mlp.c_proj",
    ]

    layers = model.model.layers if hasattr(model, "model") else model.transformer.h

    for l_idx in tqdm(range(num_layers), desc="ZD"):
        layer = layers[l_idx]
        scores = []
        total_params = 0

        for pname in proj_names_llama:
            parts = pname.split(".")
            obj = layer
            valid = True
            for p in parts:
                if hasattr(obj, p):
                    obj = getattr(obj, p)
                else:
                    valid = False
                    break
            if valid and hasattr(obj, "weight"):
                W = obj.weight.detach().float()
                scores.append((_zd_score_for_matrix(W, threshold), W.numel()))
                total_params += W.numel()

        if not scores:
            for pname in proj_names_gpt2:
                parts = pname.split(".")
                obj = layer
                valid = True
                for p in parts:
                    if hasattr(obj, p):
                        obj = getattr(obj, p)
                    else:
                        valid = False
                        break
                if valid and hasattr(obj, "weight"):
                    W = obj.weight.detach().float()
                    scores.append((_zd_score_for_matrix(W, threshold), W.numel()))
                    total_params += W.numel()

        if scores:
            layer_score = sum(s for s, _ in scores) / len(scores)
        else:
            layer_score = 0.0

        layer_scores.append(layer_score)

    output_dir = kwargs.get("output_dir", "results")
    model_name = os.path.basename(model_path.rstrip("/"))
    os.makedirs(output_dir, exist_ok=True)
    out_file = os.path.join(output_dir, f"{model_name}_zd.json")
    result = {"scores": layer_scores}
    with open(out_file, "w") as f:
        json.dump(result, f, indent=2)

    return result
