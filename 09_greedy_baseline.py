"""09_greedy_baseline.py - an uncertainty-driven greedy scheduler (no learning, no look at the targets).

At every 15-min step it tries each of the 12 sensors, asks the five frozen estimators (dry, dry_e1 ... dry_e4) for their
30-min predictions with that sensor added to the readings already collected, and picks the sensor for which the five
estimators disagree least (smallest spread of predictions, scaled by each target's standard deviation). Nothing from the
effluent targets is used; the choice depends only on past readings and the estimators.
The schedule is then scored with each estimator in turn (same scoring as the other baselines), on validation and test.
The first 4 steps before each split are a warm-up (nothing observed yet) and are not scored.

    $env:WWTP_BUDGET="1"; python 09_greedy_baseline.py
    $env:WWTP_BUDGET="1"; python 09_greedy_baseline.py --estimators dry dry_e1 dry_e2 dry_e3 dry_e4

Writes results/greedy_unc.csv (one row per estimator and split) and results/greedy_sched_<split>.npy (the schedule).
"""
import argparse

import joblib
import numpy as np
import pandas as pd

import config as C
from common import OFFS, all_configs, est_tag, load_scenario, rmse, t_range


def run_greedy(ests, Z, CM, t_all):
    """Sequential greedy rollout over decision times t_all (consecutive). Returns chosen config index per step."""
    n_c = len(CM)
    dy_sd = np.mean([e.dy_sd for e in ests], axis=0)
    first = t_all[0] - (C.EST_STEPS - 1)
    chosen = {}                                   # step -> config index
    for t in range(first, t_all[-1] + 1):
        hist = np.zeros((C.EST_STEPS, C.N_SENSORS))          # masks for steps t-4 .. t-1, then candidate at t
        for k, s in enumerate(range(t - C.EST_STEPS + 1, t)):
            if s in chosen:
                hist[k] = CM[chosen[s]]
        M = np.tile(hist, (n_c, 1, 1))
        M[:, -1, :] = CM                                       # candidate configuration at step t
        tt = np.full(n_c, t)
        P = np.stack([e.predict(Z, tt, M) for e in ests])      # (n_est, n_c, 2)
        spread = (P.std(axis=0) / dy_sd).sum(axis=1)           # (n_c,)
        chosen[t] = int(np.argmin(spread))
    return chosen


def score(est, Z, dY, CM, chosen, t):
    idx = t[:, None] + OFFS[None, :]
    M = np.stack([[CM[chosen[s]] for s in row] for row in idx])
    pred = est.predict(Z, t, M)
    return rmse(pred, dY[t])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--estimators", nargs="+", default=["dry", "dry_e1", "dry_e2", "dry_e3", "dry_e4"])
    args = ap.parse_args()
    tags = [est_tag(b) for b in args.estimators]
    ests = [joblib.load(C.MODELS_DIR / f"estimator_{tg}.joblib")["estimator"] for tg in tags]
    _, CM = all_configs()
    s = load_scenario("dry")
    Z, dY = ests[0].standardise(s["X"]), s["dY"]
    rows = []
    C.RESULTS_DIR.mkdir(exist_ok=True)
    for split in ("val", "test"):
        t = t_range(split)
        chosen = run_greedy(ests, Z, CM, t)
        np.save(C.RESULTS_DIR / f"greedy_sched_{split}.npy", np.array([chosen[x] for x in sorted(chosen)]))
        zc = rmse(np.zeros((len(t), 2)), dY[t])
        counts = np.bincount([chosen[x] for x in t], minlength=len(CM))
        print(f"\n[{split}] sensors picked (config index: share): " +
              ", ".join(f"{C.POOL[i]} {c / len(t):.0%}" for i, c in enumerate(counts) if c))
        for tg, e in zip(tags, ests):
            r = score(e, Z, dY, CM, chosen, t)
            comb = float((r / zc).sum())
            rows.append({"estimator": tg, "split": split, "dNH4": r[0], "dSNO": r[1], "combined": comb})
            print(f"  {tg:<10} dNH4 {r[0]:.4f}  dSNO {r[1]:.4f}  combined {comb:.3f}")
    out = pd.DataFrame(rows)
    out.to_csv(C.RESULTS_DIR / "greedy_unc.csv", index=False)
    m = out[out.split == "test"]["combined"]
    print(f"\nTEST combined score, mean over estimators: {m.mean():.3f}  (zero-change = 2.0, lower is better)")
    print("saved results/greedy_unc.csv")


if __name__ == "__main__":
    main()
