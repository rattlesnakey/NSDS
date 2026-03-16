import os
import json
import torch
import numpy as np
from scipy.stats import kurtosis
from transformers import AutoModelForCausalLM
from tqdm import tqdm


def _mad_normalize(arr):
    arr = np.array(arr, dtype=np.float64)
    med = np.median(arr)
    mad = np.median(np.abs(arr - med))
    epsilon = 1e-12
    z = (arr - med) / (1.0 * mad + epsilon)
    return z


def _mad_sigmoid(arr, T=1.0):
    z = _mad_normalize(arr)
    return 1.0 / (1.0 + np.exp(-z / T))


def _soft_or_aggregate(score_list):
    if not score_list:
        return None
    stacked = np.array(score_list, dtype=np.float64)
    epsilon = 1e-12
    log_mean = np.mean(np.log(1.0 - stacked + epsilon))
    return 1.0 - np.exp(log_mean)


def _compute_matrix_kurtosis(matrix):
    mat_np = matrix.cpu().numpy().flatten()
    return float(kurtosis(mat_np, fisher=True))


def _compute_vector_kurtosis(matrix):
    mat_np = matrix.cpu().numpy()
    kurt_vals = kurtosis(mat_np, axis=0, fisher=True)
    return torch.tensor(kurt_vals, dtype=torch.float32, device=matrix.device)


def _get_top90_threshold_index(S):
    energy = S ** 2
    total_energy = torch.sum(energy)
    cumulative_energy = torch.cumsum(energy, dim=0)
    cutoff_idx = torch.searchsorted(cumulative_energy, 0.99 * total_energy).item() + 1
    cutoff_idx = max(cutoff_idx, 2)
    return min(cutoff_idx, len(S))


def _compute_capacity_score(S):
    if len(S) == 0:
        return 0.0
    energy_sum = torch.sum(S).item()
    S_sum = torch.sum(S)
    if S_sum == 0:
        return 0.0
    p = S / S_sum
    entropy = -torch.sum(p * torch.log(p + 1e-12))
    effective_rank = torch.exp(entropy).item()
    return energy_sum / effective_rank


def compute_nsds(model_path, output_dir="results", **kwargs):
    model = AutoModelForCausalLM.from_pretrained(
        model_path, device_map="auto", torch_dtype=torch.bfloat16, trust_remote_code=True
    )

    cfg = model.config
    num_layers = getattr(cfg, "num_hidden_layers", getattr(cfg, "n_layer", 0))
    d_model = getattr(cfg, "hidden_size", getattr(cfg, "n_embd", 0))
    n_heads = getattr(cfg, "num_attention_heads", getattr(cfg, "n_head", 0))
    n_kv_heads = getattr(cfg, "num_key_value_heads", n_heads)
    head_dim = d_model // n_heads

    W_U = model.get_output_embeddings().weight.detach().to(torch.float32)
    _, S_U_emb, Vh_U_emb = torch.linalg.svd(W_U, full_matrices=False)
    P_W_U = Vh_U_emb.T[:, :_get_top90_threshold_index(S_U_emb)]

    modules = ["up_proj", "gate_proj", "QK_Circuit", "down_proj", "OV_Circuit"]
    metrics = {mod: {"NV": [], "SE": []} for mod in modules}

    for l_idx in tqdm(range(num_layers), desc="Processing Layers"):
        layer = model.model.layers[l_idx] if hasattr(model, "model") else model.transformer.h[l_idx]

        for proj_type in ["up_proj", "down_proj", "gate_proj"]:
            W = None
            if hasattr(layer, "mlp"):
                if proj_type == "up_proj" and hasattr(layer.mlp, "up_proj"):
                    W = layer.mlp.up_proj.weight.detach().to(torch.float32)
                elif proj_type == "down_proj" and hasattr(layer.mlp, "down_proj"):
                    W = layer.mlp.down_proj.weight.detach().to(torch.float32)
                elif proj_type == "gate_proj" and hasattr(layer.mlp, "gate_proj"):
                    W = layer.mlp.gate_proj.weight.detach().to(torch.float32)
                elif not hasattr(layer.mlp, "up_proj"):
                    if proj_type == "up_proj" and hasattr(layer.mlp, "c_fc"):
                        W = layer.mlp.c_fc.weight.detach().to(torch.float32).T
                    elif proj_type == "down_proj" and hasattr(layer.mlp, "c_proj"):
                        W = layer.mlp.c_proj.weight.detach().to(torch.float32).T

            if W is None:
                metrics[proj_type]["NV"].append(0.0)
                metrics[proj_type]["SE"].append(0.0)
                continue

            metrics[proj_type]["NV"].append(_compute_matrix_kurtosis(W))

            U, S, Vh = torch.linalg.svd(W, full_matrices=False)
            V = Vh.T
            k_90 = _get_top90_threshold_index(S)
            S_90, U_90, V_90 = S[:k_90], U[:, :k_90], V[:, :k_90]

            if proj_type in ["up_proj", "gate_proj"]:
                kurt_vec = _compute_vector_kurtosis(V_90)
                alpha_kwse = torch.log(1 + torch.relu(kurt_vec))
                metrics[proj_type]["SE"].append(_compute_capacity_score(S_90 * alpha_kwse))
            else:
                alpha_swic = torch.norm(torch.matmul(U_90.T, P_W_U), dim=0)
                metrics[proj_type]["SE"].append(_compute_capacity_score(S_90 * alpha_swic))

        W_q, W_k, W_v, W_o = None, None, None, None
        if hasattr(layer, "self_attn"):
            W_q = layer.self_attn.q_proj.weight.detach().to(torch.float32)
            W_k = layer.self_attn.k_proj.weight.detach().to(torch.float32)
            W_v = layer.self_attn.v_proj.weight.detach().to(torch.float32)
            W_o = layer.self_attn.o_proj.weight.detach().to(torch.float32)
        elif hasattr(layer, "attn"):
            W_attn = layer.attn.c_attn.weight.detach().to(torch.float32)
            W_q_raw, W_k_raw, W_v_raw = W_attn.split(d_model, dim=1)
            W_q, W_k, W_v = W_q_raw.T, W_k_raw.T, W_v_raw.T
            W_o = layer.attn.c_proj.weight.detach().to(torch.float32).T

        if W_q is not None:
            temp_ov = {"NV": [], "SE": []}
            temp_qk = {"NV": [], "SE": []}

            for h in range(n_heads):
                kv_h = h // (n_heads // n_kv_heads)
                W_Q_h = W_q[h * head_dim: (h + 1) * head_dim, :]
                W_K_h = W_k[kv_h * head_dim: (kv_h + 1) * head_dim, :]
                W_V_h = W_v[kv_h * head_dim: (kv_h + 1) * head_dim, :]
                W_O_h = W_o[:, h * head_dim: (h + 1) * head_dim]

                W_OV_h = torch.matmul(W_O_h, W_V_h)
                temp_ov["NV"].append(_compute_matrix_kurtosis(W_OV_h))

                U_ov, S_ov, _ = torch.linalg.svd(W_OV_h, full_matrices=False)
                k_90_ov = _get_top90_threshold_index(S_ov)
                S_ov_90, U_ov_90 = S_ov[:k_90_ov], U_ov[:, :k_90_ov]

                alpha_swic_ov = torch.norm(torch.matmul(U_ov_90.T, P_W_U), dim=1)
                temp_ov["SE"].append(_compute_capacity_score(S_ov_90 * alpha_swic_ov))

                W_QK_h = torch.matmul(W_Q_h.T, W_K_h)
                temp_qk["NV"].append(_compute_matrix_kurtosis(W_QK_h))

                U_qk, S_qk, Vh_qk = torch.linalg.svd(W_QK_h, full_matrices=False)
                V_qk = Vh_qk.T
                k_90_qk = _get_top90_threshold_index(S_qk)
                S_qk_90 = S_qk[:k_90_qk]
                U_qk_90 = U_qk[:, :k_90_qk]
                V_qk_90 = V_qk[:, :k_90_qk]

                kurt_u_qk = _compute_vector_kurtosis(U_qk_90)
                kurt_v_qk = _compute_vector_kurtosis(V_qk_90)
                alpha_kwse_qk = torch.log(1 + torch.relu(kurt_u_qk) * torch.relu(kurt_v_qk))
                temp_qk["SE"].append(_compute_capacity_score(S_qk_90 * alpha_kwse_qk))

            metrics["OV_Circuit"]["NV"].append(np.mean(temp_ov["NV"]))
            metrics["OV_Circuit"]["SE"].append(np.mean(temp_ov["SE"]))
            metrics["QK_Circuit"]["NV"].append(np.mean(temp_qk["NV"]))
            metrics["QK_Circuit"]["SE"].append(np.mean(temp_qk["SE"]))
        else:
            for circ in ["OV_Circuit", "QK_Circuit"]:
                metrics[circ]["NV"].append(0.0)
                metrics[circ]["SE"].append(0.0)

    layer_nv_lists = []
    layer_se_lists = []

    for mod in modules:
        if any(v != 0.0 for v in metrics[mod]["NV"]):
            layer_nv_lists.append(_mad_sigmoid(metrics[mod]["NV"]).tolist())
        if any(v != 0.0 for v in metrics[mod]["SE"]):
            layer_se_lists.append(_mad_sigmoid(metrics[mod]["SE"]).tolist())

    final_nv = _soft_or_aggregate(layer_nv_lists) if layer_nv_lists else np.zeros(num_layers)
    final_se = _soft_or_aggregate(layer_se_lists) if layer_se_lists else np.zeros(num_layers)
    final_nsds = _soft_or_aggregate([final_nv.tolist(), final_se.tolist()])

    result = {
        "scores": final_nsds.tolist() if final_nsds is not None else [],
        "final_nv": final_nv.tolist() if final_nv is not None else [],
        "final_se": final_se.tolist() if final_se is not None else [],
        "final_nsds": final_nsds.tolist() if final_nsds is not None else [],
    }

    os.makedirs(output_dir, exist_ok=True)
    out_file = os.path.join(output_dir, "nsds_scores.json")
    with open(out_file, "w") as f:
        json.dump(result, f, indent=2)

    return result
