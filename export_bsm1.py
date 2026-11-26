"""
export_bsm1.py  -  Step 1 of the project: generate BSM1 data and save it as CSV.

What it does
------------
1. Reads the dry and rain influent files that ship inside the bsm2-python package.
2. Puts them on an exact 15-minute grid (1344 rows = 14 days) and checks their units.
3. Runs the open-loop BSM1 plant (BSM1OL) with a constant-influent spin-up
   (default 100 days) followed by the 14-day dynamic influent.
4. Saves one CSV per scenario with all reactor DO / NH4-N / SNO series plus the
   effluent NH4-N and SNO targets.
5. Prints a short check report (NaNs, ranges, zero-change RMSE).

Usage (from the project folder, with the virtual environment active)
-------------------------------------------------------------------
    python export_bsm1.py --scenarios dry --spinup-days 1     # quick test (about a minute)
    python export_bsm1.py                                      # full run: dry + rain, 100-day spin-up
"""
import argparse
import json
import os
import re
import time

import numpy as np
import pandas as pd

STEPS_PER_DAY = 96            # 15-minute decisions
N_DYN = 14 * STEPS_PER_DAY    # 1344 rows
# column positions inside the 21-component ASM1 vector used by bsm2-python
SO, SNO, SNH, TSS, Q = 7, 8, 9, 13, 14
INFLUENT_FILES = {"dry": "dryinfluent.csv", "rain": "raininfluent.csv"}


# ----------------------------------------------------------------------------
# Influent handling
# ----------------------------------------------------------------------------
def read_numeric_file(path, ncols=22, report=None):
    """Read an influent file robustly (comma, semicolon, tab or space separated).

    One unreadable token per line is repaired, and every repair is reported with the
    neighbouring rows so it can be checked by eye:
      * line has the expected number of fields  -> the token has stray characters;
        keep its leading number ('30.044.50' -> 30.044)
      * line has one field too few              -> two numbers were glued together;
        split them ('30.044.50' -> 30.04 and 4.50)
    Anything else stops with the full offending line printed.
    """
    rows, repairs = [], []
    with open(path, encoding="utf-8-sig") as f:
        for ln, raw in enumerate(f, 1):
            raw = raw.strip()
            if not raw:
                continue
            fields = [t for t in re.split(r"[,;\s]+", raw) if t]
            vals, bad = [], []
            for k, tok in enumerate(fields):
                try:
                    vals.append(float(tok))
                except ValueError:
                    vals.append(None)
                    bad.append(k)
            if bad:
                if len(bad) != 1:
                    raise ValueError(f"{path}, line {ln}: several unreadable values.\nFull line: {raw}")
                k, tok = bad[0], fields[bad[0]]
                if len(fields) == ncols:
                    m = re.match(r"\d+(?:\.\d+)?", tok)
                    if not m:
                        raise ValueError(f"{path}, line {ln}: cannot read '{tok}'.\nFull line: {raw}")
                    vals[k] = float(m.group(0))
                    how = f"kept the leading number {m.group(0)} (dropped '{tok[len(m.group(0)):]}')"
                    width = 1
                elif len(fields) == ncols - 1:
                    m = re.fullmatch(r"(\d+\.\d{2})(\d+\.\d+)", tok)
                    if not m:
                        raise ValueError(f"{path}, line {ln}: cannot split '{tok}'.\nFull line: {raw}")
                    vals[k:k + 1] = [float(m.group(1)), float(m.group(2))]
                    how = f"split into {m.group(1)} and {m.group(2)}"
                    width = 2
                else:
                    raise ValueError(f"{path}, line {ln}: unexpected field count {len(fields)}.\nFull line: {raw}")
                repairs.append((len(rows), k, width, ln, tok, how))
            if len(vals) != ncols:
                raise ValueError(f"{path}, line {ln}: found {len(vals)} values, expected {ncols}.\nFull line: {raw}")
            rows.append(vals)
    arr = np.array(rows)
    for r, k, width, ln, tok, how in repairs:
        lo, hi = max(r - 1, 0), min(r + 2, len(arr))
        msg = (f"REPAIRED line {ln}: '{tok}' -> {how}.  "
               f"Column {k} of rows {lo}..{hi - 1}: {arr[lo:hi, k:k + width].round(3).tolist()}")
        if report is not None:
            report.append(msg)      # printed later by main()
        else:
            print("  " + msg)
    return arr


def load_package_influent(filename, path=None, report=None):
    if path is None:
        import bsm2_python
        root = os.path.dirname(bsm2_python.__file__)
        path = os.path.join(root, "data", filename)
    return read_numeric_file(path, report=report)


def regrid(arr):
    """Force an exact 15-minute time column (some package files store rounded times)."""
    out = arr.copy()
    out[:, 0] = np.arange(len(out)) / STEPS_PER_DAY
    return out


def fix_units_against_dry(rain, dry, report):
    """Rescale rain-file columns that are off by a factor 1000 relative to the dry file.

    The first two days of the rain and dry influents should be (nearly) identical, so
    a column ratio close to 0.001 means a unit mismatch rather than real data.
    """
    out = rain.copy()
    n = 2 * STEPS_PER_DAY
    for j in range(1, out.shape[1]):
        d, r = dry[:n, j], rain[:n, j]
        mask = d > 1e-9
        if mask.sum() < 10:
            continue
        ratio = float(np.median(r[mask] / d[mask]))
        if 0.0009 < ratio < 0.0011:
            out[:, j] *= 1000.0
            report.append(f"rain column {j}: ratio to dry = {ratio:.5f} -> multiplied by 1000")
        elif not (0.98 < ratio < 1.02):
            report.append(f"WARNING rain column {j}: ratio to dry = {ratio:.4f} (not 1 and not 0.001) - check this column")
    return out


def perturb(dry, k):
    """Influent variant k (seeded): the dry influent with a time shift, a slowly varying scale on the flow and
    another on the concentrations (+- 10-15 %), plus a little autocorrelated noise. Temperature is unchanged."""
    rng = np.random.default_rng(1000 + k)
    out = dry.copy()
    n = len(out)
    t = np.arange(n) / STEPS_PER_DAY
    out[:, 1:] = np.roll(out[:, 1:], int(rng.integers(0, 13)), axis=0)       # shift the daily pattern by 0-3 h

    def factor(amp):
        T, ph = rng.uniform(2.5, 6.0), rng.uniform(0, 2 * np.pi)
        e = np.zeros(n)
        for i in range(1, n):
            e[i] = 0.98 * e[i - 1] + rng.normal(0, 0.004)
        return rng.uniform(0.85, 1.15) * (1 + amp * np.sin(2 * np.pi * t / T + ph)) * (1 + e)

    out[:, 1 + Q] *= factor(0.10)                                              # flow
    out[:, 1:1 + TSS + 1] *= factor(0.12)[:, None]                             # concentrations SI ... TSS
    return out


def build_input(dyn, mean_row, spin_days):
    """[time | constant mean influent for the spin-up | the 14-day dynamic influent]."""
    n_spin = int(round(spin_days * STEPS_PER_DAY))
    const = np.tile(mean_row, (n_spin, 1))
    body = np.vstack([const, dyn[:, 1:]])
    t = np.arange(len(body)) / STEPS_PER_DAY
    return np.column_stack([t, body]), n_spin


# ----------------------------------------------------------------------------
# Simulation  (the only part that touches the bsm2-python API)
# ----------------------------------------------------------------------------
def run_bsm1(data_in):
    from bsm2_python import BSM1OL
    from tqdm import tqdm

    model = BSM1OL(data_in=data_in)          # timestep=None -> one step per influent row (15 min)
    n = len(model.simtime)
    if n != len(data_in):
        raise RuntimeError(f"Expected {len(data_in)} simulation steps, model has {n}")
    t0 = time.time()
    for i in tqdm(range(n), desc="BSM1 simulation"):
        model.step(i)
        if i == 199:
            est = (time.time() - t0) / 200 * n / 60
            print(f"\n  (about {est:.1f} minutes in total for this scenario)")
    return model


def extract(model, n_spin):
    sl = slice(n_spin, n_spin + N_DYN)
    outs = [model.y_out1_all, model.y_out2_all, model.y_out3_all,
            model.y_out4_all, model.y_out5_all]
    cols = {"step": np.arange(N_DYN), "time_d": np.arange(N_DYN) / STEPS_PER_DAY}
    for r, y in enumerate(outs, 1):
        cols[f"DO_R{r}"] = y[sl, SO]
    for r, y in enumerate(outs, 1):
        cols[f"NH_R{r}"] = y[sl, SNH]
    for r, y in enumerate(outs, 1):
        cols[f"SNO_R{r}"] = y[sl, SNO]
    cols["eff_SNH"] = model.ys_eff_all[sl, SNH]
    cols["eff_SNO"] = model.ys_eff_all[sl, SNO]
    cols["eff_TSS"] = model.ys_eff_all[sl, TSS]
    cols["eff_Q"] = model.ys_eff_all[sl, Q]
    cols["inf_Q"] = model.y_in_all[sl, Q]
    cols["inf_SNH"] = model.y_in_all[sl, SNH]
    return pd.DataFrame(cols)


# ----------------------------------------------------------------------------
# Checks
# ----------------------------------------------------------------------------
def check_report(df, name):
    print(f"\n=== CHECK REPORT: {name} ===")
    print("rows:", len(df), "(expected", N_DYN, ")")
    print("NaN values:", int(df.isna().sum().sum()))
    conc = [c for c in df.columns if c.startswith(("DO_", "NH_", "SNO_", "eff_S"))]
    print("most negative concentration:", float(df[conc].min().min()))
    pool = ["DO_R1", "DO_R2", "DO_R3", "DO_R4", "DO_R5",
            "NH_R1", "NH_R2", "NH_R3", "NH_R4", "NH_R5", "SNO_R3", "SNO_R5"]
    print("\nsensor statistics (mean / std / min / max):")
    for c in pool + ["eff_SNH", "eff_SNO"]:
        s = df[c]
        print(f"  {c:8s} {s.mean():9.4f} {s.std():9.4f} {s.min():9.4f} {s.max():9.4f}")
    print("\nzero-change RMSE of the 30-min change, decisions 863-1340 (478 values):")
    for tgt in ["eff_SNH", "eff_SNO"]:
        y = df[tgt].to_numpy()
        t = np.arange(863, 1341)
        d = y[t + 2] - y[t]
        print(f"  {tgt}: {np.sqrt(np.mean(d ** 2)):.4f}")
    print("  (first paper reported 0.5309 for NH4-N and 0.2764 for SNO - for orientation only)")


# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenarios", nargs="+", default=["dry", "rain"], choices=["dry", "rain"])
    ap.add_argument("--spinup-days", type=float, default=100.0)
    ap.add_argument("--outdir", default=os.path.join("data", "raw"))
    ap.add_argument("--rain-file", default=None, help="optional path to an official rain influent file")
    ap.add_argument("--variants", type=int, default=0, help="also make N perturbed dry influents: var1 ... varN")
    ap.add_argument("--only-variants", action="store_true", help="skip the dry and rain scenarios")
    ap.add_argument("--first-variant", type=int, default=1, help="first variant number to make (default 1)")
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    dry = regrid(load_package_influent(INFLUENT_FILES["dry"]))
    if dry.shape != (N_DYN, 22):
        raise RuntimeError(f"dry influent has shape {dry.shape}, expected ({N_DYN}, 22)")
    mean_row = dry[:, 1:].mean(axis=0)       # constant influent used for the spin-up

    names = [] if args.only_variants else list(args.scenarios)
    names += [f"var{k}" for k in range(args.first_variant, args.variants + 1)]
    for name in names:
        print(f"\n##### Scenario: {name} #####")
        notes = []
        if name == "dry":
            dyn = dry
        elif name.startswith("var"):
            dyn = perturb(dry, int(name[3:]))
        else:
            rain = regrid(load_package_influent(INFLUENT_FILES["rain"], path=args.rain_file, report=notes))
            if rain.shape != (N_DYN, 22):
                raise RuntimeError(f"rain influent has shape {rain.shape}, expected ({N_DYN}, 22)")
            dyn = fix_units_against_dry(rain, dry, notes)
        for line in notes:
            print("  ", line)
        print(f"  influent flow: mean {dyn[:, 1 + Q].mean():.0f}, max {dyn[:, 1 + Q].max():.0f} (m3/d)")

        data_in, n_spin = build_input(dyn, mean_row, args.spinup_days)
        model = run_bsm1(data_in)
        df = extract(model, n_spin)

        path = os.path.join(args.outdir, f"bsm1_{name}.csv")
        df.to_csv(path, index=False)
        meta = {"scenario": name, "spinup_days": args.spinup_days, "plant": "BSM1OL (open loop)",
                "step_minutes": 15, "rows": len(df), "notes": notes}
        with open(os.path.join(args.outdir, f"bsm1_{name}_meta.json"), "w") as f:
            json.dump(meta, f, indent=2)
        print("saved:", path)
        check_report(df, name)


if __name__ == "__main__":
    main()
# RETRAIN-PATCH
