import os
import json
import math
import torch
import numpy as np
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer, AutoConfig


def _normalize_rep(R):
    with torch.no_grad():
        if R.dim() == 3:
            R = R.reshape(-1, R.shape[-1])
        R = R.to(torch.float32)
        mean = R.mean(dim=1, keepdim=True)
        R = R - mean
        norms = torch.norm(R, p=2, dim=1, keepdim=True).clamp(min=1e-6)
        R = R / norms
    return R


def _compute_cov(R):
    with torch.no_grad():
        if R.dim() == 3:
            R = R.reshape(-1, R.shape[-1])
        Z = torch.nn.functional.normalize(R, dim=1, eps=1e-6)
        A = torch.matmul(Z.T, Z) / Z.shape[1]
    return A.to(torch.float32)


def _representational_compactness(A, eps=1e-8):
    with torch.no_grad():
        if A.dtype in (torch.bfloat16, torch.float16):
            A = A.to(torch.float32)
        eig_vals = torch.linalg.svdvals(A)
        prob = eig_vals / (eig_vals.sum() + eps)
        entropy = -(prob * torch.log(prob + eps)).nansum().item()
    rc = math.exp(-entropy)
    return rc


def _get_layers(m):
    return getattr(getattr(m, "model", m), "layers")


def compute_lieq(model_path, device="cuda:0", **kwargs):
    device_torch = torch.device(device if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model_tr = AutoModel.from_pretrained(
        model_path,
        torch_dtype=torch.float16 if device_torch.type == "cuda" else torch.float32,
        device_map=str(device_torch),
        trust_remote_code=True,
    )

    cfg = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    model_un = AutoModel.from_config(cfg).to(device_torch)
    if device_torch.type == "cuda":
        model_un = model_un.half()

    layers_tr = _get_layers(model_tr)
    layers_un = _get_layers(model_un)
    num_layers = len(layers_tr)

    sample_text = "The representational structure of neural networks reflects the learned semantics."
    input_ids = tokenizer(sample_text, return_tensors="pt").input_ids.to(device_torch)

    with torch.inference_mode():
        out_tr = model_tr(input_ids, output_hidden_states=True, use_cache=False, return_dict=True)
        out_un = model_un(input_ids, output_hidden_states=True, use_cache=False, return_dict=True)

    hs_tr = out_tr.hidden_states
    hs_un = out_un.hidden_states

    rc_diffs = []

    for l in tqdm(range(num_layers), desc="LieQ"):
        h_t = hs_tr[l + 1] if l + 1 < len(hs_tr) else hs_tr[l]
        h_u = hs_un[l + 1] if l + 1 < len(hs_un) else hs_un[l]

        proj_pairs = [
            (layers_tr[l].self_attn.q_proj, layers_un[l].self_attn.q_proj),
            (layers_tr[l].self_attn.k_proj, layers_un[l].self_attn.k_proj),
            (layers_tr[l].self_attn.v_proj, layers_un[l].self_attn.v_proj),
        ]

        layer_diffs = []
        for proj_t, proj_u in proj_pairs:
            Z_t = proj_t(h_t.to(proj_t.weight.device))
            Z_u = proj_u(h_u.to(proj_u.weight.device))
            rc_t = _representational_compactness(_compute_cov(_normalize_rep(Z_t)))
            rc_u = _representational_compactness(_compute_cov(_normalize_rep(Z_u)))
            layer_diffs.append(rc_t - rc_u)

        rc_diffs.append(float(np.mean(layer_diffs)))

    del model_tr, model_un
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    layer_scores = rc_diffs

    output_dir = kwargs.get("output_dir", "results")
    model_name = os.path.basename(model_path.rstrip("/"))
    os.makedirs(output_dir, exist_ok=True)
    out_file = os.path.join(output_dir, f"{model_name}_lieq.json")
    result = {"scores": layer_scores}
    with open(out_file, "w") as f:
        json.dump(result, f, indent=2)

    return result
