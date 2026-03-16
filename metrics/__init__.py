from .nsds import compute_nsds
from .zd import compute_zd
from .mse import compute_mse
from .ewq import compute_ewq
from .lim import compute_lim
from .llm_mq import compute_llm_mq
from .lsaq import compute_lsaq
from .lieq import compute_lieq
from .kurtboost import compute_kurtboost

METRIC_REGISTRY = {
    "nsds": compute_nsds,
    "zd": compute_zd,
    "mse": compute_mse,
    "ewq": compute_ewq,
    "lim": compute_lim,
    "llm_mq": compute_llm_mq,
    "lsaq": compute_lsaq,
    "lieq": compute_lieq,
    "kurtboost": compute_kurtboost,
}
