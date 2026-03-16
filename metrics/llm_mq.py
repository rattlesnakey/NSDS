import os
import re
import json
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from utils.data_utils import load_calibration_data

try:
    import pulp
    _PULP_AVAILABLE = True
except ImportError:
    _PULP_AVAILABLE = False


def _fake_quantize(weight, bit_width, group_size=128, outlier_ratio=0.005):
    pad_len = (group_size - weight.numel() % group_size) % group_size
    if pad_len > 0:
        w_padded = torch.nn.functional.pad(weight.flatten(), (0, pad_len))
    else:
        w_padded = weight.flatten()

    k = max(1, int(w_padded.numel() * outlier_ratio))
    threshold = torch.kthvalue(torch.abs(w_padded), w_padded.numel() - k + 1).values
    outlier_mask = torch.abs(w_padded) >= threshold

    reshaped_w = w_padded.view(-1, group_size)
    reshaped_mask = outlier_mask.view(-1, group_size)

    normal_w = reshaped_w.clone()
    normal_w[reshaped_mask] = 0

    w_min = normal_w.min(dim=1, keepdim=True).values
    w_max = normal_w.max(dim=1, keepdim=True).values
    q_max = 2 ** bit_width - 1
    scale = torch.clamp((w_max - w_min) / q_max, min=1e-8)
    zero_point = torch.round(-w_min / scale)

    q_w = torch.clamp(torch.round(reshaped_w / scale + zero_point), 0, q_max)
    dq_w = (q_w - zero_point) * scale

    output = dq_w.flatten()
    output[outlier_mask] = w_padded[outlier_mask]
    if pad_len > 0:
        output = output[:-pad_len]
    return output.view_as(weight)


def _compute_sensitivity(weight, grad, bit_width):
    dq_weight = _fake_quantize(weight, bit_width)
    error = weight - dq_weight
    sensitivity = torch.abs(torch.mean(grad * error)).item()
    return sensitivity


def _allocate_bits(sensitivities, avg_bit_budget):
    if not _PULP_AVAILABLE:
        raise RuntimeError("pulp is required for LLM-MQ. Install with: pip install pulp")

    layer_names = list(sensitivities.keys())
    num_layers = len(layer_names)
    bit_widths = [2, 4]

    prob = pulp.LpProblem("LLM_MQ", pulp.LpMinimize)
    c = pulp.LpVariable.dicts("C", ((i, b) for i in layer_names for b in bit_widths), cat="Binary")

    prob += pulp.lpSum([c[i, b] * sensitivities[i][b] for i in layer_names for b in bit_widths])
    prob += pulp.lpSum([c[i, b] * b for i in layer_names for b in bit_widths]) <= num_layers * avg_bit_budget
    for i in layer_names:
        prob += pulp.lpSum([c[i, b] for b in bit_widths]) == 1

    prob.solve(pulp.PULP_CBC_CMD(msg=False))

    allocation = {}
    for i in layer_names:
        for b in bit_widths:
            if pulp.value(c[i, b]) == 1:
                allocation[i] = b
    return allocation


def compute_llm_mq(model_path, avg_bit_budget=3.0, num_samples=128, max_length=2048, **kwargs):
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.float16, device_map="auto", trust_remote_code=True
    )

    cfg = model.config
    num_layers = getattr(cfg, "num_hidden_layers", getattr(cfg, "n_layer", 0))

    linear_layers = {
        name: mod
        for name, mod in model.named_modules()
        if isinstance(mod, nn.Linear) and "model.layers" in name
    }

    dataset = load_calibration_data(num_samples=num_samples)
    texts = [item["text"] for item in dataset if item.get("text", "").strip()]

    model.eval()
    model.zero_grad()

    for text in tqdm(texts, desc="LLM-MQ calibration"):
        if not text.strip():
            continue
        inputs = tokenizer(text, return_tensors="pt", max_length=max_length, truncation=True)
        inputs = {k: v.to(next(model.parameters()).device) for k, v in inputs.items()}
        if inputs["input_ids"].shape[1] < 10:
            continue
        outputs = model(**inputs, labels=inputs["input_ids"])
        loss = outputs.loss / len(texts)
        loss.backward()

    module_sens = {}
    with torch.no_grad():
        for name, layer in tqdm(linear_layers.items(), desc="Sensitivity"):
            W = layer.weight.data
            grad = layer.weight.grad
            if grad is None:
                continue
            module_sens[name] = {b: _compute_sensitivity(W, grad, b) for b in [2, 4]}

    model.zero_grad()

    layer_sens_lists = {}
    for mod_name, sens_dict in module_sens.items():
        parts = mod_name.split(".")
        layer_key = ".".join(parts[:2])
        if layer_key not in layer_sens_lists:
            layer_sens_lists[layer_key] = {2: [], 4: []}
        for b in [2, 4]:
            layer_sens_lists[layer_key][b].append(sens_dict[b])

    layer_sens = {
        k: {b: float(np.mean(v[b])) for b in [2, 4]}
        for k, v in layer_sens_lists.items()
    }

    allocation = _allocate_bits(layer_sens, avg_bit_budget)

    layer_scores = [0.0] * num_layers
    for key, val in layer_sens.items():
        m = re.search(r"\d+", key)
        if m:
            idx = int(m.group())
            if idx < num_layers:
                layer_scores[idx] = val[2]

    output_dir = kwargs.get("output_dir", "results")
    model_name = os.path.basename(model_path.rstrip("/"))
    os.makedirs(output_dir, exist_ok=True)
    out_file = os.path.join(output_dir, f"{model_name}_llm_mq.json")
    result = {"scores": layer_scores, "allocation": {k: int(v) for k, v in allocation.items()}}
    with open(out_file, "w") as f:
        json.dump(result, f, indent=2)

    return result
