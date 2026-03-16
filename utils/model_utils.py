import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig


def load_model(model_path, dtype=torch.float16, device_map="auto"):
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=dtype,
        device_map=device_map,
        trust_remote_code=True,
    )
    model.eval()
    return model, tokenizer


def get_model_arch(model):
    cfg = model.config
    num_layers = getattr(cfg, "num_hidden_layers", getattr(cfg, "n_layer", None))
    d_model = getattr(cfg, "hidden_size", getattr(cfg, "n_embd", None))
    n_heads = getattr(cfg, "num_attention_heads", getattr(cfg, "n_head", None))
    n_kv_heads = getattr(cfg, "num_key_value_heads", n_heads)
    if any(v is None for v in [num_layers, d_model, n_heads]):
        raise ValueError("Cannot resolve model architecture from config.")
    return {
        "num_layers": num_layers,
        "d_model": d_model,
        "n_heads": n_heads,
        "n_kv_heads": n_kv_heads,
        "head_dim": d_model // n_heads,
    }


def get_transformer_layers(model):
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return model.transformer.h
    raise ValueError("Unsupported model architecture: cannot locate transformer layers.")
