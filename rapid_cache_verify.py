# rapid_cache_verify.py
import numpy as np
from typing import Tuple, Dict

# ---------- numeric utilities ----------

def _unit_rows(X: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(X, axis=1, keepdims=True) + 1e-12
    return X / n

def cosine_top1_sim(Q: np.ndarray, K: np.ndarray, chunk: int = 0) -> np.ndarray:
    """
    Top-1 cosine similarity of each row in Q against rows in K.
    For large inputs, set chunk>0 to process Q in batches.
    """
    Qn = _unit_rows(Q)
    Kn = _unit_rows(K)
    N = Qn.shape[0]
    if chunk and chunk < N:
        out = np.empty(N, dtype=np.float32)
        for i in range(0, N, chunk):
            j = min(i + chunk, N)
            sims = Qn[i:j] @ Kn.T
            out[i:j] = sims.max(axis=1)
        return out
    sims = Qn @ Kn.T
    return sims.max(axis=1)

# ---------- confidence bounds ----------

def wilson_interval(p_hat: float, n: int, alpha: float = 0.05) -> Tuple[float, float]:
    """Two-sided Wilson score interval."""
    if n <= 0:
        return 0.0, 1.0
    z = 1.959963984540054 if alpha == 0.05 else _z_from_alpha(alpha)
    denom = 1 + z**2 / n
    center = (p_hat + z**2/(2*n)) / denom
    half = (z/denom) * np.sqrt((p_hat*(1-p_hat)/n) + (z**2/(4*n**2)))
    return max(0.0, center - half), min(1.0, center + half)

def hoeffding_lower(p_hat: float, n: int, alpha: float = 0.05) -> float:
    """One-sided Hoeffding lower bound for Bernoulli mean."""
    if n <= 0:
        return 0.0
    eps = np.sqrt(np.log(1/alpha) / (2*n))
    return max(0.0, p_hat - eps)

def _z_from_alpha(alpha: float) -> float:
    # fallback numeric approximation for z; alpha is two-sided tail
    from math import erfcinv, sqrt
    return sqrt(2) * erfcinv(alpha)

# ---------- metrics ----------

def estimate_hit_rate(Q: np.ndarray, K: np.ndarray, tau: float, alpha: float = 0.05,
                      chunk: int = 0) -> Dict[str, float]:
    top1 = cosine_top1_sim(Q, K, chunk=chunk)
    hits = (top1 >= tau).astype(np.int32)
    n = hits.size
    p_hat = float(hits.mean()) if n > 0 else 0.0
    lo, hi = wilson_interval(p_hat, n, alpha)
    return {"n": n, "p_hat": p_hat, "ci_lo": float(lo), "ci_hi": float(hi)}

def estimate_reuse_precision(agree_flags: np.ndarray, alpha: float = 0.05,
                             method: str = "hoeffding") -> Dict[str, float]:
    n = int(agree_flags.size)
    p_hat = float(agree_flags.mean() if n > 0 else 0.0)
    if method == "wilson":
        lo, _ = wilson_interval(p_hat, n, alpha*2)  # conservative one-sided
        lb = lo
    else:
        lb = hoeffding_lower(p_hat, n, alpha)
    return {"n": n, "p_hat": p_hat, "lb": float(lb), "method": method}

def net_benefit_lower_bound(hit_rate_lo: float, precision_lo: float,
                            C_comp: float, C_cache: float, C_penalty: float) -> Dict[str, float]:
    delta_cost = C_comp - C_cache
    benefit_term = delta_cost - (1.0 - precision_lo) * C_penalty
    B_lo = hit_rate_lo * benefit_term
    return {"benefit_per_query_lo": float(B_lo), "benefit_term": float(benefit_term)}

# ---------- decision wrapper ----------

def rapid_verify_caching(Q: np.ndarray, K: np.ndarray, tau: float,
                         agree_flags: np.ndarray,
                         C_comp: float, C_cache: float, C_penalty: float,
                         alpha: float = 0.05,
                         req_precision_lb: float = 0.99,
                         req_hit_rate_lb: float = 0.20,
                         chunk: int = 0) -> Dict[str, float]:
    """
    Conservative Go/No-Go:
      - hit_rate lower bound >= req_hit_rate_lb
      - precision lower bound >= req_precision_lb
      - net benefit lower bound > 0
    """
    hit = estimate_hit_rate(Q, K, tau, alpha=alpha, chunk=chunk)
    prec = estimate_reuse_precision(agree_flags, alpha=alpha, method="hoeffding")
    nb = net_benefit_lower_bound(hit_rate_lo=hit["ci_lo"], precision_lo=prec["lb"],
                                 C_comp=C_comp, C_cache=C_cache, C_penalty=C_penalty)
    go = (prec["lb"] >= req_precision_lb) and (hit["ci_lo"] >= req_hit_rate_lb) and (nb["benefit_per_query_lo"] > 0.0)
    return {
        "decision_go": bool(go),
        "tau": float(tau), "alpha": float(alpha),
        "hit_rate_hat": hit["p_hat"], "hit_rate_lo": hit["ci_lo"], "hit_rate_hi": hit["ci_hi"], "n_hit": hit["n"],
        "precision_hat": prec["p_hat"], "precision_lo": prec["lb"], "n_precision": prec["n"], "precision_method": prec["method"],
        "benefit_per_query_lo": nb["benefit_per_query_lo"], "benefit_term": nb["benefit_term"],
        "req_precision_lb": float(req_precision_lb), "req_hit_rate_lb": float(req_hit_rate_lb),
        "C_comp": float(C_comp), "C_cache": float(C_cache), "C_penalty": float(C_penalty)
    }
