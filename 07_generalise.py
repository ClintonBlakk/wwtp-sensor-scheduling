"""07_generalise.py - do the schedules (chosen / trained on the dry scenario only) hold up on influents they never saw?

Scenarios: the perturbed dry influents var1..varN (made by export_bsm1.py --variants N) are pooled; the rain event
is reported separately (the estimators were trained on dry weather only).  Nothing is re-tuned: static set and
rotations are the ones chosen on the DRY validation period, agents are the trained ones.  Test period of each scenario.
Block bootstrap (12 h blocks) over all test blocks of the group; paired differences as in 06_stats.py.

    $env:WWTP_BUDGET="1"; python 07_generalise.py
"""
import argparse
import glob
import importlib
import os

import joblib
import numpy as np
import pandas as pd

import config as C
from common import OFFS, all_configs, config_name, est_tag, load_scenario, rmse, t_range

R5 = importlib.import_module("05_robustness")      # load_agents, agent_masks


def sq_err(est, Z, dY, t, Meff):
    M = Meff[t[:, None] + OFFS[None, :]]
    return (est.predict(Z, t, M) - dY[t]) ** 2


def report(label, group, data, dtest, methods, tags, block, nboot, outfile):
    """data[scn][tag][method] -> (T,2) squared errors; dtest[scn] -> (T,2) true changes."""
    units_cnt, units_z, units_sq = [], [], {t: {m: [] for m in methods} for t in tags}
    for scn in group:
        T = len(dtest[scn])
        for b in np.array_split(np.arange(T), max(2, T // block)):
            units_cnt.append(len(b))
            units_z.append((dtest[scn][b] ** 2).sum(axis=0))
            for t in tags:
                for m in methods:
                    units_sq[t][m].append(data[scn][t][m][b].sum(axis=0))
    cnt, bz = np.array(units_cnt, float), np.array(units_z)
    U = len(cnt)
    rng = np.random.default_rng(0)
    idx = np.array([np.arange(U)] + [rng.integers(0, U, size=U) for _ in range(nboot)])

    def comb(bs):
        num = np.array(bs)[idx].sum(axis=1)
        n = cnt[idx].sum(axis=1)[:, None]
        z = bz[idx].sum(axis=1)
        return (np.sqrt(num / n) / np.sqrt(z / n)).sum(axis=1)

    score = {m: np.mean([comb(units_sq[t][m]) for t in tags], axis=0) for m in methods}
    print(f"\n##### {label}: {len(group)} scenario(s), {U} blocks, {len(tags)} estimators, combined score (zero-change = 2.0) #####")
    print(f"{'method':<22}{'score':>8}   95% interval")
    rows = []
    for m in methods:
        lo, hi = np.percentile(score[m][1:], [2.5, 97.5])
        print(f"{m:<22}{score[m][0]:>8.4f}   [{lo:.4f}, {hi:.4f}]")
        rows.append({"group": label, "kind": "score", "a": m, "b": "", "point": score[m][0], "lo": lo, "hi": hi, "p_a_better": np.nan})
    print(f"\n{'paired difference A - B (negative = A better)':<56}{'diff':>8}   95% interval      P(A better)")
    for a in [m for m in methods if m.startswith("AGENT")]:
        for b in methods:
            if b == a or (b.startswith("AGENT") and b < a):
                continue
            d = score[a] - score[b]
            lo, hi = np.percentile(d[1:], [2.5, 97.5])
            p = float((d[1:] < 0).mean())
            print(f"{a + '  vs  ' + b:<56}{d[0]:>+8.4f}   [{lo:+.4f}, {hi:+.4f}]   {p:>6.2f}")
            rows.append({"group": label, "kind": "diff", "a": a, "b": b, "point": d[0], "lo": lo, "hi": hi, "p_a_better": p})
    # per scenario (no bootstrap)
    print(f"\n{label}: score per scenario (mean over estimators)")
    print(f"{'method':<22}" + "".join(f"{s:>9}" for s in group))
    for m in methods:
        line = f"{m:<22}"
        for scn in group:
            v = []
            for t in tags:
                r = np.sqrt(data[scn][t][m].mean(axis=0)) / np.sqrt((dtest[scn] ** 2).mean(axis=0))
                v.append(r.sum())
            line += f"{np.mean(v):>9.3f}"
        print(line)
    pd.DataFrame(rows).to_csv(C.RESULTS_DIR / outfile, index=False)
    print(f"saved results/{outfile}")


def select_on_pool(est, scn_names, CM, max_c=4000):
    """Static sensor and period-3/4 rotations chosen on the validation periods of several influents (summed score)."""
    R4 = importlib.import_module("04_rotation_baseline")
    tv = t_range("val")
    S = [load_scenario(n) for n in scn_names]
    Zs = [est.standardise(s["X"]) for s in S]
    zc = [rmse(np.zeros((len(tv), 2)), s["dY"][tv]) for s in S]
    crit = np.zeros(len(CM))
    for s, Z, z in zip(S, Zs, zc):
        for c in range(len(CM)):
            M = np.broadcast_to(CM[c], (len(tv), C.EST_STEPS, C.N_SENSORS))
            crit[c] += float((rmse(est.predict(Z, tv, M), s["dY"][tv]) / z).sum())
    sidx = int(np.argmin(crit))
    rng = np.random.default_rng(0)
    cycles = {}
    for m in (3, 4):
        cyc, _ = R4.candidates(len(CM), m, rng, max_c)
        tot = np.zeros(len(cyc))
        for s, Z, z in zip(S, Zs, zc):
            tot += (R4.eval_cycles(est, Z, s["dY"], tv, CM, cyc) / z).sum(axis=1)
        cycles[m] = [int(i) for i in cyc[int(np.argmin(tot))]]
    return sidx, cycles


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--select-on", nargs="+", default=None,
                    help="choose static sensor and rotations on the validation periods of these influents (default: dry only)")
    ap.add_argument("--estimators", nargs="+", default=["dry", "dry_e1", "dry_e2", "dry_e3", "dry_e4"])
    ap.add_argument("--agents", nargs="+", default=["main", "robust"])
    ap.add_argument("--variants", nargs="+", default=None, help="default: every data/raw/bsm1_var*.csv")
    ap.add_argument("--stress", nargs="+", default=["rain"], help="scenarios reported separately")
    ap.add_argument("--block", type=int, default=48)
    ap.add_argument("--boot", type=int, default=5000)
    ap.add_argument("--out-suffix", default="", help="added to the result file names, e.g. _retrained")
    args = ap.parse_args()

    variants = args.variants or sorted(os.path.basename(f)[5:-4] for f in glob.glob(str(C.RAW_DIR / "bsm1_var*.csv"))
                                       if not f.endswith("_meta.json"))
    stress = [s for s in args.stress if (C.RAW_DIR / f"bsm1_{s}.csv").exists()]
    scns = variants + stress
    if not scns:
        raise SystemExit("no scenarios found - run export_bsm1.py --only-variants --variants 8 first")
    print("scenarios:", scns)

    combos, CM = all_configs()
    names = [config_name(c) for c in combos]
    n2i = {n: i for i, n in enumerate(names)}
    ng = C.N_SENSORS // C.BUDGET
    dry = load_scenario("dry")
    tv, tt = t_range("val"), t_range("test")
    zc_v = rmse(dry["dY"][tv], np.zeros((len(tv), 2)))
    S = {n: load_scenario(n) for n in scns}
    dtest = {n: S[n]["dY"][tt] for n in scns}
    data = {n: {} for n in scns}
    tags = []
    methods = None
    for ename in args.estimators:
        tag = est_tag(ename)
        try:
            est = joblib.load(C.MODELS_DIR / f"estimator_{tag}.joblib")["estimator"]
            rot = pd.read_csv(C.RESULTS_DIR / f"rotation_{tag}.csv")
        except FileNotFoundError as e:
            print(f"skip {tag}: {e}")
            continue
        Zd = est.standardise(dry["X"])
        sv = []
        for c in range(len(CM)):
            M = np.broadcast_to(CM[c], (len(tv), C.EST_STEPS, C.N_SENSORS))
            sv.append(rmse(est.predict(Zd, tv, M) - dry["dY"][tv], np.zeros((len(tv), 2))) / zc_v)
        sidx = int(np.argmin(np.array(sv).sum(axis=1)))                 # chosen on DRY validation
        cycles = {m: [n2i[x] for x in rot[rot.period == m].iloc[0].cycle.split(" -> ")] for m in (3, 4) if (rot.period == m).any()}
        if args.select_on:
            sidx, cycles = select_on_pool(est, args.select_on, CM)
        agent_sets = {nm: R5.load_agents(tag, nm) for nm in args.agents}
        print(f"{tag}: agents " + ", ".join(f"{nm}={len(v)}" for nm, v in agent_sets.items()) + f"; static = {names[sidx]}", flush=True)
        tags.append(tag)
        for n in scns:
            Z, dY = est.standardise(S[n]["X"]), S[n]["dY"]
            N = len(dY)
            sq = {"static (val)": sq_err(est, Z, dY, tt, CM[np.full(N, sidx)])}
            for m, cyc in cycles.items():
                sq[f"rotation p{m} (val)"] = sq_err(est, Z, dY, tt, CM[np.array(cyc)[np.arange(N) % m]])
            rr = []
            for k in range(10):
                perm = np.random.default_rng(2000 + k).permutation(C.N_SENSORS).reshape(ng, C.BUDGET)
                G = np.zeros((ng, C.N_SENSORS))
                for g in range(ng):
                    G[g, perm[g]] = 1.0
                rr.append(sq_err(est, Z, dY, tt, G[np.arange(N) % ng]))
            sq["round-robin"] = np.mean(rr, axis=0)
            for nm, ags in agent_sets.items():
                if ags:
                    ones = np.ones((N, C.N_SENSORS))
                    sq["AGENT " + nm] = np.mean([sq_err(est, Z, dY, tt, R5.agent_masks(ag, Z, ones, CM)) for ag in ags], axis=0)
            data[n][tag] = sq
        methods = [m for m in ["AGENT " + a for a in args.agents] + ["static (val)", "rotation p3 (val)", "rotation p4 (val)", "round-robin"]
                   if m in data[scns[0]][tag]]
    if not tags:
        raise SystemExit("nothing evaluated")
    C.RESULTS_DIR.mkdir(exist_ok=True)
    if variants:
        report("perturbed influents", variants, data, dtest, methods, tags, args.block, args.boot, f"generalise_variants{args.out_suffix}.csv")
    for s in stress:
        report(f"stress: {s}", [s], data, dtest, methods, tags, args.block, args.boot, f"generalise_{s}{args.out_suffix}.csv")


if __name__ == "__main__":
    main()
# RETRAIN-PATCH
# FAIR-PATCH
