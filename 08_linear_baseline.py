"""08_linear_baseline.py - is the learned estimator actually better than a simple linear model?

A reviewer will ask whether "70 % better than zero-change" means much.  This script fits RIDGE REGRESSIONS on the same
5-step sensor history and the same targets (30-min change of effluent NH4-N and nitrate) and reports them on the test
period of every scenario, next to the zero-change reference:

  linear, all 12 sensors      ridge on all 12 sensors (information upper bound for a linear model)
  linear, static sensor       ridge on ONE sensor; the sensor is chosen on the dry validation period
                              (the linear counterpart of the static schedule with the budget-1 estimator)

Ridge strength is chosen on the dry validation period.  Nothing is fitted on test data.  Use --train-scenarios to fit
on several influents (e.g. dry var1 var2 var3 var4), exactly as for the pooled neural estimators.

    python 08_linear_baseline.py
    python 08_linear_baseline.py --train-scenarios dry var1 var2 var3 var4 --out linear_baseline_pooled.csv
Writes results/linear_baseline.csv (one row per scenario and method) and prints a table.
"""
import argparse
import glob
import os

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

import config as C
from common import OFFS, load_scenario, rmse, t_range


def feats(Z, t, cols):
    """(B, 5*len(cols)) standardised values at steps t-4..t for the chosen sensors."""
    zz = Z[t[:, None] + OFFS[None, :]][:, :, cols]
    return zz.reshape(len(t), -1)


def fit_ridge(X, Y, alphas, Xv, Yv):
    best = None
    for a in alphas:
        m = Ridge(alpha=a).fit(X, Y)
        e = rmse(m.predict(Xv), Yv)
        crit = float((e / Yv.std(axis=0)).sum())
        if best is None or crit < best[0]:
            best = (crit, m)
    return best[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-scenarios", nargs="+", default=["dry"])
    ap.add_argument("--scenarios", nargs="+", default=None, help="default: dry, rain and every var* found")
    ap.add_argument("--out", default="linear_baseline.csv")
    args = ap.parse_args()

    found = sorted(os.path.basename(f)[5:-4] for f in glob.glob(str(C.RAW_DIR / "bsm1_*.csv")))
    scns = args.scenarios or [s for s in found if s in ("dry", "rain") or s.startswith("var")]
    train = {s: load_scenario(s) for s in args.train_scenarios}
    tr, tv, tt = t_range("train"), t_range("val"), t_range("test")
    Xtr = np.vstack([s["X"][tr] for s in train.values()])
    mu, sd = Xtr.mean(axis=0), Xtr.std(axis=0)
    sd = np.where(sd < 1e-6, 1.0, sd)
    Zs = {n: (s["X"] - mu) / sd for n, s in train.items()}
    alphas = [0.1, 1, 10, 100, 1000, 10000]
    allc = list(range(C.N_SENSORS))

    def stack(cols, split):
        t = t_range(split)
        return (np.vstack([feats(Zs[n], t, cols) for n in train]), np.vstack([train[n]["dY"][t] for n in train]))

    X, Y = stack(allc, "train")
    Xv, Yv = stack(allc, "val")
    m_all = fit_ridge(X, Y, alphas, Xv, Yv)

    # best single sensor, chosen on the validation periods of the training scenarios
    crit, models = [], []
    for j in range(C.N_SENSORS):
        X, Y = stack([j], "train")
        Xv, Yv = stack([j], "val")
        m = fit_ridge(X, Y, alphas, Xv, Yv)
        crit.append(float((rmse(m.predict(Xv), Yv) / Yv.std(axis=0)).sum()))
        models.append(m)
    js = int(np.argmin(crit))
    print(f"static sensor chosen on validation: {C.POOL[js]}")

    rows = []
    for n in scns:
        s = load_scenario(n)
        Z = (s["X"] - mu) / sd
        truth = s["dY"][tt]
        zc = rmse(np.zeros_like(truth), truth)
        for label, pred in [("zero-change", np.zeros_like(truth)),
                            ("linear, all 12 sensors", m_all.predict(feats(Z, tt, allc))),
                            (f"linear, static sensor ({C.POOL[js]})", models[js].predict(feats(Z, tt, [js])))]:
            r = rmse(pred, truth)
            rows.append({"scenario": n, "method": label, "rmse_dNH4": r[0], "rmse_dSNO": r[1],
                         "score": float((r / zc).sum()), "trained_on": "+".join(args.train_scenarios)})
    df = pd.DataFrame(rows)
    C.RESULTS_DIR.mkdir(exist_ok=True)
    df.to_csv(C.RESULTS_DIR / args.out, index=False)
    piv = df.pivot_table(index="method", columns="scenario", values="score")
    print("\ncombined score on the test period (zero-change = 2.0, lower is better)")
    print(piv.round(3).to_string())
    vcols = [c for c in piv.columns if c.startswith("var")]
    if vcols:
        print(f"\nmean over {len(vcols)} perturbed influents: " + ", ".join(f"{m}: {piv.loc[m, vcols].mean():.3f}" for m in piv.index))
    print(f"\nsaved results/{args.out}")


if __name__ == "__main__":
    main()
