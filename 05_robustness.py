"""05_robustness.py - how do the learned agent and the non-learned schedules cope with imperfect sensors?

Applied identically to every method on the TEST period of one scenario (frozen estimator, trained agents from
03_train_agent.py --name main):
  noise      : Gaussian noise on every reading (sd = 0.1 / 0.25 of the sensor's training sd)
  drift      : each sensor drifts linearly to +-0.5 sd by the end of the test period
  1 failed   : one sensor delivers nothing for the whole period (all 12 cases; mean and worst reported)
  dropout    : every acquisition fails at random with probability 0.2 / 0.4
Methods: agent (mean over its seeds), static set chosen on clean validation, rotations (period 3 and 4, chosen on clean
validation), round-robin (mean of 5 random partitions).  The agent was NOT trained with any of these faults.

    $env:WWTP_BUDGET="1"; python 05_robustness.py                      # 5 estimators, agents named 'main'
    $env:WWTP_BUDGET="1"; python 05_robustness.py --estimators dry dry_e1 --draws 2
"""
import argparse
import glob
import importlib

import joblib
import numpy as np
import pandas as pd

import config as C
from common import OFFS, all_configs, config_name, est_tag, load_scenario, rmse, t_range

A = importlib.import_module("03_train_agent")      # build_obs


def load_agents(tag, name):
    files = sorted(glob.glob(str(C.MODELS_DIR / f"agent_{name}_{tag}_s*.pt")))
    if not files:
        return []
    import torch
    import torch.nn as nn

    class QNet(nn.Module):
        def __init__(self, arch, H, d_in, nA):
            super().__init__()
            self.arch = arch
            if arch == "mlp":
                self.body = nn.Sequential(nn.Flatten(), nn.Linear(H * d_in, 256), nn.ReLU(), nn.Linear(256, 128), nn.ReLU())
            else:
                self.gru = nn.GRU(d_in, 128, batch_first=True)
                self.post = nn.Sequential(nn.Linear(128, 128), nn.ReLU())
            self.head = nn.Linear(128, nA)

        def forward(self, x):
            if self.arch == "mlp":
                h = self.body(x)
            else:
                out, _ = self.gru(x)
                h = self.post(out[:, -1])
            return self.head(h)

    agents = []
    for f in files:
        ck = torch.load(f, map_location="cpu", weights_only=True)
        q = QNet(ck["arch"], ck["hist"], ck["d_in"], ck["n_actions"])
        q.load_state_dict(ck["state"])
        q.eval()
        agents.append({"q": q, "ck": ck, "seed": ck["seed"], "torch": torch})
    return agents


def agent_masks(ag, Zm, Avail, CM):
    """Deploy the agent causally on perturbed readings; the effective mask = chosen subset x availability."""
    torch, q, ck = ag["torch"], ag["q"], ag["ck"]
    H, N = ck["hist"], len(Zm)
    MH = np.zeros((1, H, C.N_SENSORS))
    Meff = np.zeros((N, C.N_SENSORS))
    with torch.no_grad():
        for t in range(H, N):
            o = A.build_obs(Zm, np.array([t]), MH, ck["tod"], ck["mask"])
            a = int(q(torch.from_numpy(o)).argmax(1))
            m = CM[a] * Avail[t]
            Meff[t] = m
            MH = np.concatenate([MH[:, 1:], m[None, None, :]], axis=1)
    return Meff


def score(est, Zm, dY, t, Meff):
    M = Meff[t[:, None] + OFFS[None, :]]
    return rmse(est.predict(Zm, t, M) - dY[t], np.zeros((len(t), 2)))


def make_cases(Z, rng, t0, nt, draws):
    N, S = Z.shape
    ones = np.ones((N, S))
    cases = {"clean": [(Z, ones)]}
    for sg in (0.1, 0.25):
        cases[f"noise {sg}"] = [(Z + sg * rng.standard_normal(Z.shape), ones) for _ in range(draws)]
    ramp = np.clip((np.arange(N) - t0) / nt, 0, 1)[:, None]
    cases["drift 0.5"] = [(Z + 0.5 * ramp * rng.choice([-1.0, 1.0], size=S)[None, :], ones) for _ in range(draws)]
    fails = []
    for j in range(S):
        Av = ones.copy()
        Av[:, j] = 0.0
        fails.append((Z, Av))
    cases["1 failed (mean)"] = fails
    for p in (0.2, 0.4):
        cases[f"dropout {p}"] = [(Z, (rng.random((N, S)) >= p).astype(float)) for _ in range(draws)]
    return cases


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--estimators", nargs="+", default=["dry", "dry_e1", "dry_e2", "dry_e3", "dry_e4"])
    ap.add_argument("--agents", nargs="+", default=["main"], help="names given to 03_train_agent.py --name")
    ap.add_argument("--scenario", default="dry")
    ap.add_argument("--draws", type=int, default=3)
    args = ap.parse_args()

    combos, CM = all_configs()
    names = [config_name(c) for c in combos]
    n2i = {n: i for i, n in enumerate(names)}
    ng = C.N_SENSORS // C.BUDGET
    s = load_scenario(args.scenario)
    dY = s["dY"]
    tv, tt = t_range("val"), t_range("test")
    zc_t = rmse(dY[tt], np.zeros((len(tt), 2)))
    zc_v = rmse(dY[tv], np.zeros((len(tv), 2)))
    N = len(dY)
    per_est, rows = [], []

    for ename in args.estimators:
        tag = est_tag(ename)
        try:
            est = joblib.load(C.MODELS_DIR / f"estimator_{tag}.joblib")["estimator"]
            rot = pd.read_csv(C.RESULTS_DIR / f"rotation_{tag}.csv")
        except FileNotFoundError as e:
            print(f"skip {tag}: {e}")
            continue
        Z = est.standardise(s["X"])
        # schedules chosen on CLEAN validation
        sv = []
        for c in range(len(CM)):
            M = np.broadcast_to(CM[c], (len(tv), C.EST_STEPS, C.N_SENSORS))
            sv.append(rmse(est.predict(Z, tv, M) - dY[tv], np.zeros((len(tv), 2))) / zc_v)
        sidx = int(np.argmin(np.array(sv).sum(axis=1)))
        sched = {"static (val)": CM[np.full(N, sidx)]}
        for m in (3, 4):
            r = rot[rot.period == m]
            if len(r):
                cyc = [n2i[x] for x in r.iloc[0].cycle.split(" -> ")]
                sched[f"rotation p{m} (val)"] = CM[np.array(cyc)[np.arange(N) % m]]
        rrs = []
        for k in range(5):
            perm = np.random.default_rng(2000 + k).permutation(C.N_SENSORS).reshape(ng, C.BUDGET)
            G = np.zeros((ng, C.N_SENSORS))
            for g in range(ng):
                G[g, perm[g]] = 1.0
            rrs.append(G[np.arange(N) % ng])
        agent_sets = {nm: load_agents(tag, nm) for nm in args.agents}
        print(f"{tag}: agents " + ", ".join(f"{nm}={len(v)}" for nm, v in agent_sets.items()) + f"; static = {names[sidx]}", flush=True)

        rng = np.random.default_rng(123)
        cases = make_cases(Z, rng, tt[0], len(tt), args.draws)
        res = {}                                           # method -> case -> (dNH4, dSNO)
        for cname, draws in cases.items():
            acc = {}
            for Zm, Av in draws:
                for m, Ms in sched.items():
                    acc.setdefault(m, []).append(score(est, Zm, dY, tt, Ms * Av))
                acc.setdefault("round-robin", []).append(np.mean([score(est, Zm, dY, tt, R_ * Av) for R_ in rrs], axis=0))
                for nm, agents in agent_sets.items():
                    if agents:
                        acc.setdefault("AGENT " + nm, []).append(np.mean([score(est, Zm, dY, tt, agent_masks(ag, Zm, Av, CM)) for ag in agents], axis=0))
            for m, lst in acc.items():
                lst = np.array(lst)
                res.setdefault(m, {})[cname] = lst.mean(axis=0)
                if cname.startswith("1 failed"):
                    res[m]["1 failed (worst)"] = lst[int(np.argmax((lst / zc_t).sum(axis=1)))]
        for m, d in res.items():
            for cname, v in d.items():
                rows.append({"estimator": tag, "method": m, "case": cname, "dNH4": v[0], "dSNO": v[1]})
        per_est.append(res)

    if not per_est:
        raise SystemExit("nothing evaluated")
    C.RESULTS_DIR.mkdir(exist_ok=True)
    outname = "_".join(args.agents)
    pd.DataFrame(rows).to_csv(C.RESULTS_DIR / f"robustness_{outname}.csv", index=False)
    methods = [m for m in ["AGENT " + nm for nm in args.agents] + ["static (val)", "rotation p3 (val)", "rotation p4 (val)", "round-robin"] if m in per_est[0]]
    cases_all = list(per_est[0][methods[0]].keys())
    for k, title in ((2, "combined = dNH4/zero-change + dSNO/zero-change"), (0, "dNH4"), (1, "dSNO")):
        print(f"\n##### budget {C.BUDGET}, {args.scenario} TEST, {title}; mean over {len(per_est)} estimators #####")
        print(f"{'method':<20}" + "".join(f"{c[:16]:>18}" for c in cases_all))
        for m in methods:
            line = f"{m:<20}"
            for c in cases_all:
                vs = [(r[m][c][0] / zc_t[0] + r[m][c][1] / zc_t[1]) if k == 2 else r[m][c][k] for r in per_est if m in r]
                line += f"{np.mean(vs):>18.4f}"
            print(line)
    print(f"\nsaved results/robustness_{outname}.csv")


if __name__ == "__main__":
    main()
