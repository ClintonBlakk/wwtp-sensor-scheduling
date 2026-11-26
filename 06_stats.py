"""06_stats.py - uncertainty for the clean-data comparison: block bootstrap over the test period.

The test period is split into blocks (default 12 h = 48 steps) which are resampled with replacement (same blocks for
every method and estimator, so differences are paired).  Combined score = dNH4/zero-change + dSNO/zero-change,
averaged over the estimator seeds.  Agents: mean squared error over their training seeds.

    $env:WWTP_BUDGET="1"; python 06_stats.py                      # agents 'main' and 'robust'
"""
import argparse
import glob

import joblib
import numpy as np
import pandas as pd

import config as C
from common import OFFS, all_configs, config_name, est_tag, load_scenario, rmse, t_range


def sq_err(est, Z, dY, t, Meff):
    M = Meff[t[:, None] + OFFS[None, :]]
    return (est.predict(Z, t, M) - dY[t]) ** 2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--estimators", nargs="+", default=["dry", "dry_e1", "dry_e2", "dry_e3", "dry_e4"])
    ap.add_argument("--agents", nargs="+", default=["main", "robust"])
    ap.add_argument("--block", type=int, default=48)
    ap.add_argument("--boot", type=int, default=5000)
    args = ap.parse_args()

    combos, CM = all_configs()
    names = [config_name(c) for c in combos]
    n2i = {n: i for i, n in enumerate(names)}
    ng = C.N_SENSORS // C.BUDGET
    s = load_scenario("dry")
    dY = s["dY"]
    N = len(dY)
    tv, tt = t_range("val"), t_range("test")
    zc_v = rmse(dY[tv], np.zeros((len(tv), 2)))
    per_est = []
    for ename in args.estimators:
        tag = est_tag(ename)
        try:
            est = joblib.load(C.MODELS_DIR / f"estimator_{tag}.joblib")["estimator"]
            rot = pd.read_csv(C.RESULTS_DIR / f"rotation_{tag}.csv")
        except FileNotFoundError as e:
            print(f"skip {tag}: {e}")
            continue
        Z = est.standardise(s["X"])
        sv = []
        for c in range(len(CM)):
            M = np.broadcast_to(CM[c], (len(tv), C.EST_STEPS, C.N_SENSORS))
            sv.append(rmse(est.predict(Z, tv, M) - dY[tv], np.zeros((len(tv), 2))) / zc_v)
        sidx = int(np.argmin(np.array(sv).sum(axis=1)))
        sq = {"static (val)": sq_err(est, Z, dY, tt, CM[np.full(N, sidx)])}
        for m in (3, 4):
            r = rot[rot.period == m]
            if len(r):
                cyc = [n2i[x] for x in r.iloc[0].cycle.split(" -> ")]
                sq[f"rotation p{m} (val)"] = sq_err(est, Z, dY, tt, CM[np.array(cyc)[np.arange(N) % m]])
        rr = []
        for k in range(10):
            perm = np.random.default_rng(2000 + k).permutation(C.N_SENSORS).reshape(ng, C.BUDGET)
            G = np.zeros((ng, C.N_SENSORS))
            for g in range(ng):
                G[g, perm[g]] = 1.0
            rr.append(sq_err(est, Z, dY, tt, G[np.arange(N) % ng]))
        sq["round-robin"] = np.mean(rr, axis=0)
        for nm in args.agents:
            files = sorted(glob.glob(str(C.RESULTS_DIR / f"agent_sched_{nm}_{tag}_dry_s*.npy")))
            if files:
                sq["AGENT " + nm] = np.mean([sq_err(est, Z, dY, tt, CM[np.load(f)]) for f in files], axis=0)
        per_est.append(sq)
        print(f"{tag}: methods {list(sq)}", flush=True)
    if not per_est:
        raise SystemExit("nothing evaluated")

    T = len(tt)
    nb = max(2, T // args.block)
    blocks = np.array_split(np.arange(T), nb)
    cnt = np.array([len(b) for b in blocks], float)
    zsq = dY[tt] ** 2
    bz = np.array([zsq[b].sum(axis=0) for b in blocks])
    rng = np.random.default_rng(0)
    idx_all = [np.arange(nb)] + [rng.integers(0, nb, size=nb) for _ in range(args.boot)]
    idx = np.array(idx_all)                                           # row 0 = the actual data

    def comb(sqm):
        bs = np.array([sqm[b].sum(axis=0) for b in blocks])
        num = bs[idx].sum(axis=1)
        n = cnt[idx].sum(axis=1)[:, None]
        z = bz[idx].sum(axis=1)
        return (np.sqrt(num / n) / np.sqrt(z / n)).sum(axis=1)       # (boot+1,)

    methods = [m for m in ["AGENT " + a for a in args.agents] + ["static (val)", "rotation p3 (val)", "rotation p4 (val)", "round-robin"]
               if all(m in e for e in per_est)]
    score = {m: np.mean([comb(e[m]) for e in per_est], axis=0) for m in methods}
    print(f"\n##### budget {C.BUDGET}, dry TEST, combined score; {len(per_est)} estimators; {nb} blocks of ~{args.block} steps; "
          f"{args.boot} resamples #####")
    print(f"{'method':<22}{'score':>8}   95% interval")
    rows = []
    for m in methods:
        lo, hi = np.percentile(score[m][1:], [2.5, 97.5])
        print(f"{m:<22}{score[m][0]:>8.4f}   [{lo:.4f}, {hi:.4f}]")
        rows.append({"kind": "score", "a": m, "b": "", "point": score[m][0], "lo": lo, "hi": hi, "p_a_better": np.nan})
    print(f"\n{'paired difference A - B (negative = A better)':<56}{'diff':>8}   95% interval      P(A better)")
    pairs = [(a, b) for a in methods if a.startswith("AGENT") for b in methods if b != a and not (b.startswith("AGENT") and b < a)]
    for a, b in pairs:
        d = score[a] - score[b]
        lo, hi = np.percentile(d[1:], [2.5, 97.5])
        p = float((d[1:] < 0).mean())
        print(f"{a + '  vs  ' + b:<56}{d[0]:>+8.4f}   [{lo:+.4f}, {hi:+.4f}]   {p:>6.2f}")
        rows.append({"kind": "diff", "a": a, "b": b, "point": d[0], "lo": lo, "hi": hi, "p_a_better": p})
    C.RESULTS_DIR.mkdir(exist_ok=True)
    pd.DataFrame(rows).to_csv(C.RESULTS_DIR / "stats_clean.csv", index=False)
    print("\nsaved results/stats_clean.csv")
    print("note: the intervals cover test-period variation only (about %.1f days); they do not include training randomness." % (T / C.STEPS_PER_DAY))


if __name__ == "__main__":
    main()
