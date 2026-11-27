"""verify_bsm1_steady_state.py - check our BSM1 open-loop run against the published BSM1 benchmark.

Reference: Alex et al. (2008) BSM1 report (LTH-IEA-7229): constant influent (Table 5),
open-loop steady state (Tables 6 and 8). Two checks:
  A. the package's dry-weather influent file: its averages should equal the Table 5 values;
  B. BSM1OL run with the Table 5 constant influent: reactor and effluent steady state vs Tables 6/8.
Usage:  python verify_bsm1_steady_state.py            (about the time of one scenario run)
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import export_bsm1 as E

# 21-component vector order used by bsm2-python: SI SS XI XS XBH XBA XP SO SNO SNH SND XND SALK TSS Q TEMP SD1 SD2 SD3 XD4 XD5
TABLE5 = {0: ("SI", 30.0), 1: ("SS", 69.5), 2: ("XI", 51.2), 3: ("XS", 202.32), 4: ("XBH", 28.17),
          9: ("SNH", 31.56), 10: ("SND", 6.95), 11: ("XND", 10.59), 12: ("SALK", 7.0), 14: ("Q", 18446.0)}
REF = {  # Table 6, units 1-5
    "SO": [0.00430, 0.0000631, 1.72, 2.43, 0.491],
    "SNO": [5.37, 3.66, 6.54, 9.30, 10.4],
    "SNH": [7.92, 8.34, 5.55, 2.97, 1.73],
}
REF_EFF = {"SNH": 1.73, "SNO": 10.4}   # Table 8


def check_a(dry):
    print("\n=== A. Package dry-weather influent: averages vs BSM1 Table 5 ===")
    rows = []
    mean_row = dry[:, 1:].mean(axis=0)
    for j, (name, ref) in TABLE5.items():
        val = float(mean_row[j])
        rows.append({"check": "dry_influent_mean", "variable": name, "model": val, "reference": ref,
                     "rel_diff_pct": 100 * (val - ref) / ref})
        print(f"  {name:5s} file mean {val:10.3f}   reference {ref:10.3f}   diff {100 * (val - ref) / ref:+6.2f} %")
    return rows


def check_b(days):
    print(f"\n=== B. BSM1OL with the Table 5 constant influent, {days:g} days ===")
    v = np.zeros(21)
    for j, (name, ref) in TABLE5.items():
        v[j] = ref
    v[13] = 0.75 * (v[3] + v[2] + v[4])      # TSS from particulate COD (0.75 * (XS + XI + XBH + XBA + XP))
    v[15] = 15.0                               # temperature, deg C
    n = int(round(days * E.STEPS_PER_DAY))
    t = np.arange(n) / E.STEPS_PER_DAY
    data_in = np.column_stack([t, np.tile(v, (n, 1))])
    model = E.run_bsm1(data_in)
    outs = [model.y_out1_all, model.y_out2_all, model.y_out3_all, model.y_out4_all, model.y_out5_all]
    idx = {"SO": E.SO, "SNO": E.SNO, "SNH": E.SNH}
    rows = []
    for var in ["SO", "SNO", "SNH"]:
        for r, y in enumerate(outs):
            val, ref = float(y[-1, idx[var]]), REF[var][r]
            rows.append({"check": "steady_state", "variable": f"{var}_R{r + 1}", "model": val, "reference": ref,
                         "rel_diff_pct": 100 * (val - ref) / ref})
    for var in ["SNH", "SNO"]:
        val, ref = float(model.ys_eff_all[-1, idx[var]]), REF_EFF[var]
        rows.append({"check": "steady_state", "variable": f"effluent_{var}", "model": val, "reference": ref,
                     "rel_diff_pct": 100 * (val - ref) / ref})
    for r in rows:
        print(f"  {r['variable']:14s} model {r['model']:10.4f}   reference {r['reference']:10.4f}   diff {r['rel_diff_pct']:+7.2f} %")
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=float, default=100.0)
    ap.add_argument("--outdir", default="results")
    args = ap.parse_args()
    dry = E.regrid(E.load_package_influent(E.INFLUENT_FILES["dry"]))
    rows = check_a(dry) + check_b(args.days)
    os.makedirs(args.outdir, exist_ok=True)
    path = os.path.join(args.outdir, "validation_bsm1_steady_state.csv")
    pd.DataFrame(rows).to_csv(path, index=False)
    print("\nsaved:", path)
    ss = [r for r in rows if r["check"] == "steady_state" and not r["variable"].startswith("SO_")]
    print("largest steady-state difference (SNH, SNO; reactors and effluent): "
          f"{max(abs(r['rel_diff_pct']) for r in ss):.2f} %")


if __name__ == "__main__":
    main()
