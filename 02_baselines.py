"""02_baselines.py - reference results and simple (non-learned) schedulers, all under the same
causal masking and the same frozen estimator.

    python 02_baselines.py                        # estimator 'dry'; evaluates dry and rain
    python 02_baselines.py --scenarios dry        # dry only
    $env:WWTP_BUDGET="2"; python 02_baselines.py  # a different sensor budget (needs: 01_train_estimator.py first)

Methods reported (RMSE of the 30-min change in effluent NH4-N and SNO):
  zero-change reference
  all 12 sensors, mixed-trained estimator  and  all 12 sensors, full-information estimator (if trained)
  exhaustive fixed configurations: best / mean / median / worst   (best and worst are chosen ex post)
  static configuration chosen on the validation period (deployable)
  random scheduler (new random configuration every step), mean over seeds
  round-robin scheduler over 12/budget groups, mean over random partitions
  recent-change heuristic scheduler
  temporal oracle (best fixed configuration per window, chosen ex post)
  persistence test: the configuration that was best in the PREVIOUS window, used in the next window
"""
import argparse

import joblib
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

import config as C
from common import OFFS, all_configs, config_name, est_tag, load_scenario, rmse, t_range


def err_for_masks(est, Z, dY, t, Mseq):
    """Prediction error when the acquisition schedule is Mseq (N,12) (one row per time step)."""
    M = Mseq[t[:, None] + OFFS[None, :]]               # (T,5,12)
    return est.predict(Z, t, M) - dY[t]


def fixed_errors(est, Z, dY, t, CM, chunk=40):
    """Errors (n_configs, T, 2) for every fixed configuration held for the whole period."""
    out = np.empty((len(CM), len(t), 2))
    for s in range(0, len(CM), chunk):
        c = CM[s:s + chunk]
        k = len(c)
        M = np.broadcast_to(c[:, None, None, :], (k, len(t), C.EST_STEPS, C.N_SENSORS)).reshape(-1, C.EST_STEPS, C.N_SENSORS)
        tt = np.tile(t, k)
        out[s:s + k] = (est.predict(Z, tt, M) - np.tile(dY[t], (k, 1))).reshape(k, len(t), 2)
    return out


def rms(e):
    return np.sqrt(np.mean(e ** 2, axis=-2))


def heuristic_schedule(Z):
    """Recent-change heuristic: activate the sensors whose last two acquired readings differed most;
    never-observed sensors first. Uses only information acquired before the current step."""
    N = len(Z)
    last, prev = np.zeros(C.N_SENSORS), np.zeros(C.N_SENSORS)
    n_obs, age = np.zeros(C.N_SENSORS, int), np.zeros(C.N_SENSORS)
    Ms = np.zeros((N, C.N_SENSORS))
    for t in range(N):
        score = np.where(n_obs == 0, 1e6, np.where(n_obs >= 2, np.abs(last - prev), 0.0)) + 1e-3 * age
        chosen = np.argsort(-score, kind="stable")[:C.BUDGET]
        Ms[t, chosen] = 1.0
        age += 1
        for i in chosen:
            prev[i], last[i] = last[i], Z[t, i]
            n_obs[i] += 1
            age[i] = 0
    return Ms


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--estimator", default="dry")
    ap.add_argument("--full-estimator", default="full_dry")
    ap.add_argument("--scenarios", nargs="+", default=["dry", "rain"])
    ap.add_argument("--n-random", type=int, default=20)
    args = ap.parse_args()

    B = C.BUDGET
    ng = C.N_SENSORS // B
    if C.N_SENSORS % B != 0:
        raise SystemExit("Budget must divide 12 (1, 2, 3, 4 or 6) so round-robin groups are complete.")
    C.RESULTS_DIR.mkdir(exist_ok=True)
    tag = est_tag(args.estimator)
    est = joblib.load(C.MODELS_DIR / f"estimator_{tag}.joblib")["estimator"]
    full_path = C.MODELS_DIR / f"estimator_{args.full_estimator}.joblib"
    full_est = joblib.load(full_path)["estimator"] if full_path.exists() else None
    print(f"budget = {B}; estimator = {tag}; full-information estimator: {'yes' if full_est else 'not found (skipped)'}")
    combos, CM = all_configs()
    names = [config_name(c) for c in combos]
    rows = []

    def add(scn, split, method, r, sd=None, zc=None):
        row = {"scenario": scn, "split": split, "method": method, "rmse_dNH4": r[0], "rmse_dSNO": r[1]}
        if sd is not None:
            row["sd_dNH4"], row["sd_dSNO"] = sd
        if zc is not None:
            row["red_dNH4_%"], row["red_dSNO_%"] = 100 * (1 - r[0] / zc[0]), 100 * (1 - r[1] / zc[1])
        rows.append(row)

    # configuration chosen on the dry validation period (used for cross-scenario tests)
    dry = load_scenario("dry")
    Zd = est.standardise(dry["X"])
    tv0 = t_range("val")
    ev = rms(fixed_errors(est, Zd, dry["dY"], tv0, CM))
    zc_dry_val = rmse(np.zeros((len(tv0), 2)), dry["dY"][tv0])
    best_dry_val = int(np.argmin((ev / zc_dry_val).sum(axis=1)))
    print(f"static configuration chosen on dry validation: {names[best_dry_val]}")

    for scn in args.scenarios:
        s = load_scenario(scn)
        Z = est.standardise(s["X"])
        N = len(Z)
        tv = t_range("val")
        ev_same = rms(fixed_errors(est, Z, s["dY"], tv, CM))
        zc_v = rmse(np.zeros((len(tv), 2)), s["dY"][tv])
        best_same_val = int(np.argmin((ev_same / zc_v).sum(axis=1)))
        fr_val = None

        for split in ["val", "test"]:
            t = t_range(split)
            truth = s["dY"][t]
            zc = rmse(np.zeros_like(truth), truth)
            add(scn, split, "zero-change", zc, zc=zc)
            add(scn, split, "all 12 sensors (mixed-trained estimator)",
                rmse(est.predict(Z, t, np.ones((len(t), C.EST_STEPS, C.N_SENSORS))), truth), zc=zc)
            if full_est is not None:
                Zf = full_est.standardise(s["X"])
                add(scn, split, "all 12 sensors (full-information estimator)",
                    rmse(full_est.predict(Zf, t, np.ones((len(t), C.EST_STEPS, C.N_SENSORS))), truth), zc=zc)

            fe = fixed_errors(est, Z, s["dY"], t, CM)          # (n_configs, T, 2)
            fr = rms(fe)
            comb = (fr / zc).sum(axis=1)
            add(scn, split, f"fixed {B}: best (ex post)", fr[int(np.argmin(comb))], zc=zc)
            add(scn, split, f"fixed {B}: mean of {len(CM)}", fr.mean(axis=0), sd=fr.std(axis=0), zc=zc)
            add(scn, split, f"fixed {B}: median of {len(CM)}", np.median(fr, axis=0), zc=zc)
            add(scn, split, f"fixed {B}: worst (ex post)", fr[int(np.argmax(comb))], zc=zc)
            add(scn, split, "static: best on same-scenario val", fr[best_same_val], zc=zc)
            add(scn, split, "static: best on dry val", fr[best_dry_val], zc=zc)

            rr = []
            for k in range(args.n_random):
                rng = np.random.default_rng(1000 + k)
                Ms = CM[rng.integers(0, len(CM), size=N)]
                rr.append(rms(err_for_masks(est, Z, s["dY"], t, Ms)))
            rr = np.array(rr)
            add(scn, split, f"random scheduler (mean of {args.n_random})", rr.mean(0), sd=rr.std(0), zc=zc)

            rb = []
            for k in range(args.n_random):
                rng = np.random.default_rng(2000 + k)
                perm = rng.permutation(C.N_SENSORS).reshape(ng, B)
                G = np.zeros((ng, C.N_SENSORS))
                for g in range(ng):
                    G[g, perm[g]] = 1.0
                Ms = G[np.arange(N) % ng]
                rb.append(rms(err_for_masks(est, Z, s["dY"], t, Ms)))
            rb = np.array(rb)
            add(scn, split, f"round-robin, {ng} groups (mean of {args.n_random})", rb.mean(0), sd=rb.std(0), zc=zc)

            add(scn, split, "recent-change heuristic",
                rms(err_for_masks(est, Z, s["dY"], t, heuristic_schedule(Z))), zc=zc)

            if split == "val":
                fr_val = fr
            if split == "test":
                # does a configuration's rank carry over from validation to test?
                rho = [spearmanr(fr_val[:, j], fr[:, j])[0] for j in range(2)]
                print(f"[{scn}] rank correlation of the {len(CM)} fixed configurations, validation vs test: "
                      f"dNH4 {rho[0]:.3f}  dSNO {rho[1]:.3f}")

                wins = np.array_split(np.arange(len(t)), 5)
                best_idx, parts = [[], []], [0.0, 0.0]
                pers_sq, pers_n = np.zeros(2), 0
                later = np.concatenate(wins[1:])
                for j in range(2):
                    for w in wins:
                        wr = np.sqrt((fe[:, w, j] ** 2).mean(axis=1))
                        b = int(np.argmin(wr))
                        best_idx[j].append(b)
                        parts[j] += (fe[b, w, j] ** 2).sum()
                    # persistence: configuration best in window k, applied in window k+1
                    for k in range(4):
                        pers_sq[j] += (fe[best_idx[j][k], wins[k + 1], j] ** 2).sum()
                pers_n = len(later)
                orc = np.sqrt(np.array(parts) / len(t))
                add(scn, split, "temporal oracle, 5 windows (ex post)", orc, zc=zc)
                zc_later = rmse(np.zeros((len(later), 2)), truth[later])
                add(scn, split, "windows 2-5: previous-window best config", np.sqrt(pers_sq / pers_n), zc=zc_later)
                add(scn, split, f"windows 2-5: mean of {len(CM)} fixed",
                    np.sqrt((fe[:, later, :] ** 2).mean(axis=1)).mean(axis=0), zc=zc_later)
                add(scn, split, "windows 2-5: static (best on same-scenario val)",
                    np.sqrt((fe[best_same_val][later] ** 2).mean(axis=0)), zc=zc_later)
                for j, lab in enumerate(C.TARGET_LABELS):
                    print(f"[{scn}] oracle windows {lab}: {len(set(best_idx[j]))} distinct configurations")
                    for wi, b in enumerate(best_idx[j], 1):
                        print(f"     window {wi}: {names[b]}")

            pd.DataFrame({"config": names, "rmse_dNH4": fr[:, 0], "rmse_dSNO": fr[:, 1]}).to_csv(
                C.RESULTS_DIR / f"fixed_configs_{tag}_{scn}_{split}.csv", index=False)

    df = pd.DataFrame(rows)
    out = C.RESULTS_DIR / f"baselines_{tag}.csv"
    df.to_csv(out, index=False)
    pd.set_option("display.width", 220)
    pd.set_option("display.max_columns", 20)
    for scn in args.scenarios:
        for split in ["val", "test"]:
            d = df[(df.scenario == scn) & (df.split == split)].drop(columns=["scenario", "split"])
            print(f"\n=== {scn} / {split}  (budget {B}) ===")
            print(d.round(4).to_string(index=False))
    print("\nsaved:", out)


if __name__ == "__main__":
    main()