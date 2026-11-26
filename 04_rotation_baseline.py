"""04_rotation_baseline.py - strongest NON-learned dynamic baseline: a fixed cyclic rotation chosen on validation."""
import argparse
import itertools
import math

import joblib
import numpy as np
import pandas as pd

import config as C
from common import OFFS, all_configs, config_name, est_tag, load_scenario, rmse, t_range


def candidates(n, m, rng, max_c):
    if m == 1:
        return [(i,) for i in range(n)], False
    total = math.comb(n, m) * math.factorial(m - 1)
    if total <= max_c:
        out = []
        for comb in itertools.combinations(range(n), m):
            for perm in itertools.permutations(comb[1:]):
                out.append((comb[0],) + perm)
        return out, False
    return [tuple(rng.choice(n, m, replace=False)) for _ in range(max_c)], True


def eval_cycles(est, Z, dY, t, CM, cyc, chunk=40):
    cyc = np.asarray(cyc)
    c, m = cyc.shape
    idx = t[:, None] + OFFS[None, :]
    res = np.empty((c, 2))
    for s in range(0, c, chunk):
        cc = cyc[s:s + chunk]
        k = len(cc)
        M = CM[cc[:, idx % m]]
        pred = est.predict(Z, np.tile(t, k), M.reshape(-1, C.EST_STEPS, C.N_SENSORS))
        err = pred.reshape(k, len(t), 2) - dY[t][None]
        res[s:s + k] = np.sqrt((err ** 2).mean(axis=1))
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--estimator", default="dry")
    ap.add_argument("--max-period", type=int, default=4)
    ap.add_argument("--max-candidates", type=int, default=4000)
    args = ap.parse_args()

    tag = est_tag(args.estimator)
    est = joblib.load(C.MODELS_DIR / f"estimator_{tag}.joblib")["estimator"]
    combos, CM = all_configs()
    names = [config_name(c) for c in combos]
    s = load_scenario("dry")
    Z, dY = est.standardise(s["X"]), s["dY"]
    tv, tt = t_range("val"), t_range("test")
    zc_v = rmse(np.zeros((len(tv), 2)), dY[tv])
    rng = np.random.default_rng(0)
    rows, best = [], None
    print(f"budget {C.BUDGET}, estimator {tag}: {len(CM)} configurations")
    print(f"{'period':>6}{'cycles':>9}  {'val dNH4':>9}{'val dSNO':>9}{'TEST dNH4':>10}{'TEST dSNO':>10}   best rotation on validation")
    for m in range(1, args.max_period + 1):
        cyc, sampled = candidates(len(CM), m, rng, args.max_candidates)
        rv = eval_cycles(est, Z, dY, tv, CM, cyc)
        k = int(np.argmin((rv / zc_v).sum(axis=1)))
        rt = eval_cycles(est, Z, dY, tt, CM, [cyc[k]])[0]
        label = " -> ".join(names[i] for i in cyc[k])
        print(f"{m:>6}{len(cyc):>8}{'*' if sampled else ' '}  {rv[k, 0]:>9.4f}{rv[k, 1]:>9.4f}{rt[0]:>10.4f}{rt[1]:>10.4f}   {label}")
        rows.append({"estimator": tag, "period": m, "n_cycles": len(cyc), "sampled": sampled, "cycle": label,
                     "val_dNH4": rv[k, 0], "val_dSNO": rv[k, 1], "test_dNH4": rt[0], "test_dSNO": rt[1]})
        crit = float((rv[k] / zc_v).sum())
        if best is None or crit < best[0]:
            best = (crit, rows[-1])
    b = best[1]
    print(f"\nperiod chosen on validation: {b['period']}  ->  TEST dNH4 {b['test_dNH4']:.4f}  dSNO {b['test_dSNO']:.4f}")
    print("(* = random sample of the possible rotations, not all of them)")
    C.RESULTS_DIR.mkdir(exist_ok=True)
    out = pd.DataFrame(rows)
    out["chosen_on_val"] = [r["period"] == b["period"] for r in rows]
    out.to_csv(C.RESULTS_DIR / f"rotation_{tag}.csv", index=False)
    print(f"saved results/rotation_{tag}.csv")


if __name__ == "__main__":
    main()