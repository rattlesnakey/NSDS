import os
import json
import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from utils.data_utils import load_calibration_data


def _compute_bi(x_in, x_out):
    x_in = x_in.view(-1, x_in.size(-1)).float()
    x_out = x_out.view(-1, x_out.size(-1)).float()
    cos_sim = F.cosine_similarity(x_in, x_out, dim=0)
    return (1 - cos_sim.mean()).item()


def compute_lim(model_path, num_samples=128, batch_size=8, max_length=2048, **kwargs):
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_path, device_map="auto", torch_dtype=torch.float16, trust_remote_code=True
    )
    model.eval()

    cfg = model.config
    num_layers = getattr(cfg, "num_hidden_layers", getattr(cfg, "n_layer", 0))
    layers = model.model.layers if hasattr(model, "model") else model.transformer.h

    dataset = load_calibration_data(num_samples=num_samples)
    texts = [item["text"] for item in dataset if item.get("text", "").strip()]

    bi_sums = torch.zeros(num_layers)
    count = 0

    for i in tqdm(range(0, len(texts), batch_size), desc="LIM"):
        batch_texts = texts[i: i + batch_size]
        inputs = tokenizer(
            batch_texts, return_tensors="pt", padding=True,
            truncation=True, max_length=max_length,
        ).to(next(model.parameters()).device)

        hidden_in, hidden_out = {}, {}

        def _make_hook(lid):
            def _hook(module, inp, out):
                hidden_in[lid] = inp[0].detach().cpu()
                hidden_out[lid] = out[0].detach().cpu()
            return _hook

        hooks = [
            block.register_forward_hook(_make_hook(i))
            for i, block in enumerate(layers, start=1)
        ]

        with torch.no_grad():
            model(**inputs)

        for h in hooks:
            h.remove()

        for l in range(num_layers):
            if l in hidden_in and l in hidden_out:
                bi_sums[l] += _compute_bi(hidden_in[l], hidden_out[l])

        count += 1

    layer_scores = (bi_sums / count).tolist()

    output_dir = kwargs.get("output_dir", "results")
    model_name = os.path.basename(model_path.rstrip("/"))
    os.makedirs(output_dir, exist_ok=True)
    out_file = os.path.join(output_dir, f"{model_name}_lim.json")
    result = {"scores": layer_scores}
    with open(out_file, "w") as f:
        json.dump(result, f, indent=2)

    return result
