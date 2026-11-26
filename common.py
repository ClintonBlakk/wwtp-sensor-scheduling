"""common.py - data loading, sensor configurations, estimator features and the frozen estimator."""
import itertools

import numpy as np
import pandas as pd

import config as C

OFFS = np.arange(-(C.EST_STEPS - 1), 1)       # [-4, -3, -2, -1, 0]


# ---------------------------------------------------------------- data
def load_scenario(name):
    df = pd.read_csv(C.RAW_DIR / f"bsm1_{name}.csv")
    X = df[C.POOL].to_numpy(float)                     # (N, 12) sensor series
    Y = df[C.TARGETS].to_numpy(float)                  # (N, 2) effluent series
    dY = np.full_like(Y, np.nan)
    dY[:-C.HORIZON] = Y[C.HORIZON:] - Y[:-C.HORIZON]   # change over the next 30 min
    return {"name": name, "X": X, "Y": Y, "dY": dY}


def t_range(split):
    a, b = C.SPLITS[split]
    return np.arange(a, b + 1)


# ---------------------------------------------------------------- configurations
def all_configs():
    """All 495 four-of-twelve configurations: list of index tuples and a (495, 12) 0/1 matrix."""
    combos = list(itertools.combinations(range(C.N_SENSORS), C.BUDGET))
    CM = np.zeros((len(combos), C.N_SENSORS))
    for i, c in enumerate(combos):
        CM[i, list(c)] = 1.0
    return combos, CM


def config_name(combo):
    return "+".join(C.POOL[i] for i in combo)


# ---------------------------------------------------------------- estimator features
def build_features(Z, t_idx, M):
    """Z: (N,12) standardised sensors. t_idx: (B,) decision times. M: (B,5,12) 0/1 acquisition masks
    for steps t-4..t.  Returns (B,120): masked values (60) followed by the masks (60)."""
    B = len(t_idx)
    zz = Z[t_idx[:, None] + OFFS[None, :]]             # (B,5,12)
    return np.concatenate([(zz * M).reshape(B, -1), M.reshape(B, -1)], axis=1)


def sample_masks(rng, n, CM, p_full=0.10, p_fixed=0.45):
    """Random acquisition histories used to train the estimator:
    10% all 12 sensors, 45% one fixed configuration held for 5 steps, 45% a new random configuration each step."""
    u = rng.random(n)
    idx = rng.integers(0, len(CM), size=(n, C.EST_STEPS))
    M = CM[idx]                                         # switching histories (n,5,12)
    fixed = (u >= p_full) & (u < p_full + p_fixed)
    fidx = rng.integers(0, len(CM), size=n)
    M[fixed] = np.repeat(CM[fidx[fixed]][:, None, :], C.EST_STEPS, axis=1)
    M[u < p_full] = 1.0
    return M


# ---------------------------------------------------------------- frozen estimator
class MaskedEstimator:
    """Two MLPs (one per target) predicting the 30-min effluent change from masked sensor history."""

    def __init__(self, mu, sd, dy_sd, models):
        self.mu, self.sd, self.dy_sd, self.models = mu, sd, dy_sd, models

    def standardise(self, X):
        return (X - self.mu) / self.sd

    def predict(self, Z, t_idx, M, chunk=20000):
        out = []
        for s in range(0, len(t_idx), chunk):
            F = build_features(Z, t_idx[s:s + chunk], M[s:s + chunk])
            out.append(np.column_stack([m.predict(F) * k for m, k in zip(self.models, self.dy_sd)]))
        return np.vstack(out)


def est_tag(base):
    """Estimator file tag. Budget 4 keeps the plain name (e.g. 'dry'); other budgets get a suffix ('dry_b2').
    Estimators named 'full_...' (all-12 information) do not depend on the budget."""
    if base.startswith("full_") or C.BUDGET == 4:
        return base
    return f"{base}_b{C.BUDGET}"


def rmse(pred, truth):
    return np.sqrt(np.mean((pred - truth) ** 2, axis=0))