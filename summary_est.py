import numpy as np
import pandas as pd

import config as C
from common import est_tag

B = C.BUDGET
tags = [est_tag("dry")] + [est_tag(f"dry_e{s}") for s in range(1, 10)]
tags = [t for t in tags if (C.RESULTS_DIR / f"baselines_{t}.csv").exists()]
KEYS = [("zero-change", "zero-change"), ("all 12 sensors", "all 12 sensors (mixed"),
        (f"best fixed {B} (ex post, optimistic)", f"fixed {B}: best"), ("static, chosen on validation", "static: best on dry val"),
        ("round-robin", "round-robin"), ("random", "random scheduler"),
        ("temporal oracle (ex post, optimistic)", "temporal oracle"),
        ("windows 2-5: previous-window best", "windows 2-5: previous-window"),
        ("windows 2-5: static", "windows 2-5: static")]
bl = {}
for t in tags:
    d = pd.read_csv(C.RESULTS_DIR / f"baselines_{t}.csv")
    bl[t] = d[(d.scenario == "dry") & (d.split == "test")].set_index("method")


def get(t, prefix, col):
    m = [x for x in bl[t].index if x.startswith(prefix)]
    return float(bl[t].loc[m[0], col]) if m else np.nan


for col, lab in (("rmse_dNH4", "dNH4"), ("rmse_dSNO", "dSNO")):
    print(f"\n##### budget {B}, dry TEST, {lab} - one column per estimator seed #####")
    head = f"{'method':<42}" + "".join(f"{'est' + str(i):>8}" for i in range(len(tags))) + f"{'mean':>9}{'sd':>8}"
    print(head)
    for name, pre in KEYS:
        v = np.array([get(t, pre, col) for t in tags])
        print(f"{name:<42}" + "".join(f"{x:>8.4f}" for x in v) + f"{np.nanmean(v):>9.4f}{np.nanstd(v):>8.4f}")

p = C.RESULTS_DIR / "agent_runs.csv"
if p.exists():
    a = pd.read_csv(p)
    if "estimator" not in a.columns:
        a["estimator"] = np.nan
    a["estimator"] = a["estimator"].fillna(tags[0])
    a = a[(a.scenario == "dry") & (a.budget == B)]
    for name, g in a.groupby("agent"):
        print(f"\n##### agent {name}: per estimator, dry test (mean over its seeds) #####")
        print(f"{'estimator':<14}{'seeds':>6}{'agent NH4':>11}{'static':>9}{'vs static':>11}{'RR':>9}{'vs RR':>9}"
              f"{'agent SNO':>11}{'static':>9}{'vs static':>11}{'RR':>9}{'vs RR':>9}")
        res = []
        for t, h in g.groupby("estimator"):
            if t not in bl:
                continue
            r = [h.test_dNH4.mean(), get(t, "static: best on dry val", "rmse_dNH4"), get(t, "round-robin", "rmse_dNH4"),
                 h.test_dSNO.mean(), get(t, "static: best on dry val", "rmse_dSNO"), get(t, "round-robin", "rmse_dSNO")]
            res.append(r)
            print(f"{t:<14}{len(h):>6}{r[0]:>11.4f}{r[1]:>9.4f}{100 * (r[0] / r[1] - 1):>+10.1f}%{r[2]:>9.4f}{100 * (r[0] / r[2] - 1):>+8.1f}%"
                  f"{r[3]:>11.4f}{r[4]:>9.4f}{100 * (r[3] / r[4] - 1):>+10.1f}%{r[5]:>9.4f}{100 * (r[3] / r[5] - 1):>+8.1f}%")
        if len(res) > 1:
            R = np.array(res)
            pct = lambda i, j: 100 * (R[:, i] / R[:, j] - 1)
            print(f"{'MEAN':<14}{'':>6}{R[:, 0].mean():>11.4f}{R[:, 1].mean():>9.4f}{pct(0, 1).mean():>+10.1f}%{R[:, 2].mean():>9.4f}{pct(0, 2).mean():>+8.1f}%"
                  f"{R[:, 3].mean():>11.4f}{R[:, 4].mean():>9.4f}{pct(3, 4).mean():>+10.1f}%{R[:, 5].mean():>9.4f}{pct(3, 5).mean():>+8.1f}%")
            print(f"{'SD of % diff':<14}{'':>6}{'':>11}{'':>9}{pct(0, 1).std():>10.1f}%{'':>9}{pct(0, 2).std():>8.1f}%"
                  f"{'':>11}{'':>9}{pct(3, 4).std():>10.1f}%{'':>9}{pct(3, 5).std():>8.1f}%")