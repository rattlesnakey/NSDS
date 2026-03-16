import os
import sys
import json
import argparse
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from metrics import METRIC_REGISTRY


CALIBRATION_METRICS = {"lim", "llm_mq", "lsaq"}

DATA_FREE_METRICS = {"nsds", "zd", "mse", "ewq", "kurtboost", "lieq"}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Layer-wise sensitivity metric evaluation for mixed-precision quantization."
    )
    parser.add_argument(
        "--model_path", type=str, required=True,
        help="Path or HuggingFace Hub ID of the target model.",
    )
    parser.add_argument(
        "--metric", type=str, required=True,
        choices=sorted(METRIC_REGISTRY.keys()),
        help="Sensitivity metric to compute.",
    )
    parser.add_argument(
        "--output_dir", type=str, default="results",
        help="Directory to save output JSON files.",
    )
    parser.add_argument("--num_bits", type=int, default=2)
    parser.add_argument("--num_samples", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--max_length", type=int, default=2048)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--avg_bit_budget", type=float, default=3.0)
    parser.add_argument(
        "--normalize", action="store_true",
        help="Apply min-max normalization to layer scores before saving.",
    )
    return parser.parse_args()


def _normalize_scores(scores):
    arr = np.array(scores, dtype=np.float64)
    min_val = arr.min()
    max_val = arr.max()
    normalized = (arr - min_val) / (max_val + 1e-8)
    return normalized.tolist()


def _rank_layers(scores):
    arr = np.array(scores)
    ranked = np.argsort(arr).tolist()
    return ranked


def _build_output(metric_name, model_path, scores, extra=None):
    model_name = os.path.basename(model_path.rstrip("/\\"))
    num_layers = len(scores)
    layer_labels = [f"layer_{i}" for i in range(1, num_layers + 1)]
    layer_map = {label: float(s) for label, s in zip(layer_labels, scores)}
    out = {
        "metric": metric_name,
        "model": model_name,
        "num_layers": num_layers,
        "layer_scores": layer_map,
        "ranked_layers": _rank_layers(scores),
        "scores": scores,
    }
    if extra:
        out.update(extra)
    return out


def _save_output(output_dict, output_dir, metric_name, model_path):
    model_name = os.path.basename(model_path.rstrip("/\\"))
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f"{model_name}_{metric_name}.json")
    with open(out_path, "w") as f:
        json.dump(output_dict, f, indent=2)
    return out_path


def main():
    args = parse_args()

    if args.metric not in METRIC_REGISTRY:
        print(f"Unknown metric: {args.metric}. Available: {sorted(METRIC_REGISTRY.keys())}")
        sys.exit(1)

    print(f"[*] Computing metric: {args.metric}")
    print(f"[*] Model: {args.model_path}")
    print(f"[*] Output directory: {args.output_dir}")

    metric_fn = METRIC_REGISTRY[args.metric]

    shared_kwargs = {
        "model_path": args.model_path,
        "output_dir": args.output_dir,
        "num_bits": args.num_bits,
        "num_samples": args.num_samples,
        "batch_size": args.batch_size,
        "max_length": args.max_length,
        "device": args.device,
        "avg_bit_budget": args.avg_bit_budget,
    }

    result = metric_fn(**shared_kwargs)

    raw_scores = result.get("scores", [])

    if not raw_scores:
        print("[!] Warning: metric returned empty scores.")
        sys.exit(1)

    if args.normalize:
        raw_scores = _normalize_scores(raw_scores)

    extra_keys = {k: v for k, v in result.items() if k != "scores"}
    output_dict = _build_output(args.metric, args.model_path, raw_scores, extra=extra_keys)

    out_path = _save_output(output_dict, args.output_dir, args.metric, args.model_path)
    print(f"[+] Results saved to: {out_path}")

    print(f"\n[*] Top-5 most sensitive layers (0-indexed):")
    ranked = _rank_layers(raw_scores)
    for rank, layer_idx in enumerate(ranked[:5]):
        print(f"    Rank {rank + 1}: Layer {layer_idx}  (score={raw_scores[layer_idx]:.6f})")

    return output_dict


if __name__ == "__main__":
    main()
