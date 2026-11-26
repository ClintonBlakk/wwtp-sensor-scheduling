import numpy as np
import pandas as pd

import config as C
from common import est_tag

B = C.BUDGET
R = C.RESULTS_DIR
tags = [est_tag("dry")] + [est_tag(f"dry_e{s}") for s in range(1, 10)]
tags = [t for t in tags if (R / f"rotation_{t}.csv").exists() and (R / f"baselines_{t}.csv").exists()]
a = pd.read_csv(R / "agent_runs.csv")
if "estimator" not in a.columns:
    a["estimator"] = np.nan
a = a[~(a["estimator"].isna() & (a["best_step"] == 9600))]
a["estimator"] = a["estimator"].fillna(tags[0])
a = a[(a.scenario == "dry") & (a.budget == B)]

rows, zc = {}, {}


def put(m, t, v):
    rows.setdefault(m, {})[t] = v


for t in tags:
    b = pd.read_csv(R / f"baselines_{t}.csv")
    b = b[(b.scenario == "dry") & (b.split == "test")].set_index("method")
    g = lambda p: [x for x in b.index if x.startswith(p)][0]
    zc[t] = (b.loc[g("zero-change"), "rmse_dNH4"], b.loc[g("zero-change"), "rmse_dSNO"])
    for lab, p in (("static (chosen on validation)", "static: best on dry val"), ("round-robin", "round-robin")):
        put(lab, t, (b.loc[g(p), "rmse_dNH4"], b.loc[g(p), "rmse_dSNO"]))
    for _, x in pd.read_csv(R / f"rotation_{t}.csv").iterrows():
        if int(x.period) >= 2:
            put(f"rotation, period {int(x.period)}", t, (x.test_dNH4, x.test_dSNO))
        if bool(x.chosen_on_val):
            put("rotation, period chosen on val", t, (x.test_dNH4, x.test_dSNO))
    for name, h in a[a.estimator == t].groupby("agent"):
        put(f"AGENT {name}", t, (h.test_dNH4.mean(), h.test_dSNO.mean()))

order = sorted(rows, key=lambda m: (not m.startswith("AGENT"), m.startswith("round"), m))
for k, title in ((0, "dNH4"), (1, "dSNO"), (2, "combined = dNH4/zero + dSNO/zero (lower is better)")):
    print(f"\n##### budget {B}, dry TEST, {title} #####")
    print(f"{'method':<36}" + "".join(f"{'est' + str(i):>8}" for i in range(len(tags))) + f"{'mean':>9}")
    for m in order:
        vals = []
        for t in tags:
            v = rows[m].get(t)
            if v is None:
                vals.append(np.nan)
            elif k < 2:
                vals.append(v[k])
            else:
                vals.append(v[0] / zc[t][0] + v[1] / zc[t][1])
        print(f"{m:<36}" + "".join(f"{x:>8.4f}" for x in vals) + f"{np.nanmean(vals):>9.4f}")