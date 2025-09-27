# demo_simulation.py
import numpy as np
from rapid_cache_verify import rapid_verify_caching

rng = np.random.default_rng(42)

# --- simulate embeddings ---
N, k, d = 4000, 800, 128      # queries, cache size, dimension
Q = rng.normal(size=(N, d))
K = rng.normal(size=(k, d))

# --- construct a "ground truth" reuse correctness model for demo ---
# If a query is close to some cache item (high cosine), reuse is very likely correct.
# We simulate agreement probability as a sigmoid of top-1 cosine.
def top1_cosine(Q, K, chunk=0):
    from rapid_cache_verify import cosine_top1_sim
    return cosine_top1_sim(Q, K, chunk=chunk)

top1 = top1_cosine(Q, K, chunk=500)
# Choose a threshold tau for candidate hits
tau = 0.75

# Probed set for precision estimation (simulate cached vs fresh agreement flags)
M = 800
idx = rng.choice(N, size=M, replace=False)
top1_probe = top1[idx]

# agreement probability rises sharply above tau; add a little noise
def agree_prob(c):
    # Map cosine in [-1, 1] to probability ~ [0.95, 0.999] when high
    return 0.95 + 0.049 * (1 / (1 + np.exp(-25*(c - tau))))  # smooth step around tau

p_agree = agree_prob(top1_probe)
agree_flags = (rng.random(M) < p_agree).astype(np.int32)

# --- costs ---
C_comp = 1.00    # cost to recompute
C_cache = 0.10   # cost to reuse
C_penalty = 8.0  # cost of a wrong reuse (domain-dependent)

# --- decision thresholds ---
alpha = 0.05
req_precision_lb = 0.99
req_hit_rate_lb = 0.20

# --- run rapid verification ---
res = rapid_verify_caching(
    Q=Q, K=K, tau=tau, agree_flags=agree_flags,
    C_comp=C_comp, C_cache=C_cache, C_penalty=C_penalty,
    alpha=alpha, req_precision_lb=req_precision_lb, req_hit_rate_lb=req_hit_rate_lb,
    chunk=500
)

print("=== Rapid Verification of Caching ===")
for k in ["decision_go", "hit_rate_hat", "hit_rate_lo", "precision_hat", "precision_lo",
          "benefit_per_query_lo", "benefit_term", "n_hit", "n_precision"]:
    print(f"{k}: {res[k]}")
print("\nParams:",
      {kk: res[kk] for kk in ["tau","alpha","req_precision_lb","req_hit_rate_lb","C_comp","C_cache","C_penalty"]})
