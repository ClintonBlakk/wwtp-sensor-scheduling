"""03_train_agent.py - learned sensor scheduler (constraint-aware DQN, MLP or GRU) on top of the frozen estimator.

At every 15-minute step the agent picks ONE of the C(12,K) sensor subsets of size K (so the budget K is met by
construction).  It sees only what was acquired BEFORE the decision: the masked sensor history, the availability
mask and (optionally) the time of day.  Reward = minus the estimator's normalised squared error for that step
(optionally minus a penalty when the subset changes).  The estimator is frozen (trained by 01_train_estimator.py).

Examples (PowerShell, project root, venv active):
    $env:WWTP_BUDGET="1"; python 03_train_agent.py --seeds 11 --steps 30000        # quick test (about 1 min)
    $env:WWTP_BUDGET="1"; python 03_train_agent.py --seeds 11 22 33                # real run, 3 seeds
    $env:WWTP_BUDGET="2"; python 03_train_agent.py --seeds 11 22 33
Ablations:  --arch mlp | --no-tod | --no-mask | --hist 4 | --switch-penalty 0.1

Needs: models/estimator_dry[_b{K}].joblib  (01_train_estimator.py) and results/baselines_dry[_b{K}].csv (02_baselines.py).
Writes: results/agent_runs.csv (one row per seed and scenario), results/agent_sched_<name>_s<seed>.npy (the schedules).
"""
import argparse
import copy
import time

import joblib
import numpy as np
import pandas as pd

import config as C
from common import OFFS, all_configs, est_tag, load_scenario, rmse, t_range


# ------------------------------------------------------------------ observations
def build_obs(Z, t, MH, use_tod=True, use_mask=True):
    """Observation for decisions at times t (B,). MH (B,H,12) = masks of steps t-H..t-1 (what was acquired).
    Returns (B,H,d): masked standardised values [+ masks] [+ sin/cos time of day]."""
    H = MH.shape[1]
    idx = t[:, None] - H + np.arange(H)[None, :]          # steps t-H .. t-1
    return build_obs_vh(Z[idx] * MH, MH, t, use_tod, use_mask)


def build_obs_vh(VH, MH, t, use_tod=True, use_mask=True):
    """Same observation, from the values actually received (VH, already masked) instead of the clean series."""
    H = MH.shape[1]
    idx = t[:, None] - H + np.arange(H)[None, :]
    parts = [VH]
    if use_mask:
        parts.append(MH)
    if use_tod:                                            # phase of step s+1: the last row = the decision time
        ph = 2 * np.pi * ((idx + 1) % C.STEPS_PER_DAY) / C.STEPS_PER_DAY
        parts += [np.sin(ph)[..., None], np.cos(ph)[..., None]]
    return np.concatenate(parts, axis=2).astype(np.float32)


# ------------------------------------------------------------------ training environment (vectorised)
class VecEnv:
    def __init__(self, Z, dY, est, CM, n_env, H, t_lo, t_hi, rng, ref, lam, use_tod, use_mask, fault=False,
                 n_scn=1, n_len=0):
        self.Z, self.dY, self.est, self.CM = Z, dY, est, CM
        self.n, self.H, self.t_lo, self.t_hi = n_env, H, t_lo, t_hi
        self.n_scn, self.n_len = n_scn, n_len             # several influents stacked one after the other
        self.hi = np.zeros(n_env, int)                    # last training step of each environment
        self.rng, self.ref, self.lam = rng, ref, lam
        self.use_tod, self.use_mask = use_tod, use_mask
        self.dy_sd = np.asarray(est.dy_sd, float)
        self.t = np.zeros(n_env, int)
        self.MH = np.zeros((n_env, H, C.N_SENSORS))
        self.prev = np.zeros(n_env, int)
        self.fault = fault                               # fault-injection training (dead sensor, dropout, noise)
        self.A = np.ones((n_env, C.N_SENSORS))           # sensor availability
        self.pdrop = np.zeros(n_env)
        self.nsd = np.zeros(n_env)
        self.VH = np.zeros((n_env, H, C.N_SENSORS))      # values received over the last H steps
        self.reset(np.arange(n_env))

    def reset(self, idx):
        if len(idx) == 0:
            return
        base = (self.rng.integers(0, self.n_scn, size=len(idx)) if self.n_scn > 1 else np.zeros(len(idx), int)) * self.n_len
        self.t[idx] = base + self.rng.integers(self.t_lo, self.t_hi + 1, size=len(idx))
        self.hi[idx] = base + self.t_hi
        a = self.rng.integers(0, len(self.CM), size=(len(idx), self.H))   # random earlier history
        self.MH[idx] = self.CM[a]
        self.prev[idx] = a[:, -1]
        if self.fault:
            k = len(idx)
            self.A[idx] = 1.0
            rows = idx[self.rng.random(k) < 0.5]                    # half of the episodes: one sensor is dead
            self.A[rows, self.rng.integers(0, C.N_SENSORS, size=len(rows))] = 0.0
            self.pdrop[idx] = np.where(self.rng.random(k) < 0.5, self.rng.uniform(0, 0.3, size=k), 0.0)
            self.nsd[idx] = np.where(self.rng.random(k) < 0.5, self.rng.uniform(0, 0.25, size=k), 0.0)
            self.MH[idx] = self.MH[idx] * self.A[idx][:, None, :]
            hidx = self.t[idx][:, None] - self.H + np.arange(self.H)[None, :]
            noise = self.nsd[idx][:, None, None] * self.rng.standard_normal((k, self.H, C.N_SENSORS))
            self.VH[idx] = (self.Z[hidx] + noise) * self.MH[idx]

    def obs(self):
        if self.fault:
            return build_obs_vh(self.VH, self.MH, self.t, self.use_tod, self.use_mask)
        return build_obs(self.Z, self.t, self.MH, self.use_tod, self.use_mask)

    def step(self, act):
        m_now = self.CM[act]
        if self.fault:
            avail = self.A * (self.rng.random((self.n, C.N_SENSORS)) >= self.pdrop[:, None])
            m_now = m_now * avail                                    # what is actually received
            zread = (self.Z[self.t] + self.nsd[:, None] * self.rng.standard_normal((self.n, C.N_SENSORS))) * m_now
        M5 = np.concatenate([self.MH[:, -(C.EST_STEPS - 1):], m_now[:, None, :]], axis=1)   # steps t-4..t
        if self.fault:
            V5 = np.concatenate([self.VH[:, -(C.EST_STEPS - 1):], zread[:, None, :]], axis=1)
            Fe = np.concatenate([V5.reshape(self.n, -1), M5.reshape(self.n, -1)], axis=1)
            pred = np.column_stack([m.predict(Fe) * k for m, k in zip(self.est.models, self.est.dy_sd)])
        else:
            pred = self.est.predict(self.Z, self.t, M5)
        e2 = ((pred - self.dY[self.t]) / self.dy_sd) ** 2                # (n,2)
        r = -np.minimum((e2 / self.ref).mean(axis=1), 10.0)
        r = r - self.lam * (act != self.prev)
        self.MH = np.concatenate([self.MH[:, 1:], m_now[:, None, :]], axis=1)
        if self.fault:
            self.VH = np.concatenate([self.VH[:, 1:], zread[:, None, :]], axis=1)
        self.prev = act.copy()
        self.t = self.t + 1
        o2 = self.obs()                                                      # t may be t_hi+1 (valid index)
        done = self.t > self.hi
        self.reset(np.where(done)[0])
        return r.astype(np.float32), o2, done, e2


# ------------------------------------------------------------------ scoring a schedule
def pred_for_masks(est, Z, t, Mseq):
    M = Mseq[t[:, None] + OFFS[None, :]]
    return est.predict(Z, t, M)


def score(est, Z, dY, Ms, split):
    t = t_range(split)
    r = rmse(pred_for_masks(est, Z, t, Ms), dY[t])
    zc = rmse(np.zeros((len(t), 2)), dY[t])
    return r, zc


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--estimator", default="dry")
    ap.add_argument("--seeds", type=int, nargs="+", default=[11])
    ap.add_argument("--arch", choices=["gru", "mlp"], default="gru")
    ap.add_argument("--hist", type=int, default=8, help="steps of history the agent sees")
    ap.add_argument("--no-tod", action="store_true", help="hide the time of day")
    ap.add_argument("--no-mask", action="store_true", help="hide the availability mask (values only)")
    ap.add_argument("--switch-penalty", type=float, default=0.0)
    ap.add_argument("--steps", type=int, default=120000, help="total environment steps")
    ap.add_argument("--n-env", type=int, default=16)
    ap.add_argument("--updates", type=int, default=2, help="gradient updates per vector step")
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--gamma", type=float, default=0.9)
    ap.add_argument("--eval-every", type=int, default=300, help="validation every N vector steps")
    ap.add_argument("--eval-scenarios", nargs="+", default=["dry"])
    ap.add_argument("--train-scenarios", nargs="+", default=["dry"],
                    help="influents the agent trains on (and is validated on); default: dry only")
    ap.add_argument("--name", default=None)
    ap.add_argument("--fault-train", action="store_true",
                    help="train with random dead sensors, dropout and noise (robust agent)")
    args = ap.parse_args()

    import torch                      # imported here so the data-only parts can be tested without torch
    import torch.nn as nn
    import torch.nn.functional as F
    torch.set_num_threads(max(1, min(4, torch.get_num_threads())))

    B, H = C.BUDGET, args.hist
    use_tod, use_mask = not args.no_tod, not args.no_mask
    combos, CM = all_configs()
    nA = len(CM)
    tag = est_tag(args.estimator)
    est = joblib.load(C.MODELS_DIR / f"estimator_{tag}.joblib")["estimator"]
    name = args.name or (f"{args.arch}_b{B}_h{H}" + ("_notod" if args.no_tod else "") + ("_nomask" if args.no_mask else "")
                         + (f"_sw{args.switch_penalty:g}" if args.switch_penalty else "") + ("_fault" if args.fault_train else ""))
    C.RESULTS_DIR.mkdir(exist_ok=True)
    print(f"agent {name}: budget {B}, {nA} actions, estimator {tag}, history {H}, tod={use_tod}, mask={use_mask}, "
          f"switch penalty {args.switch_penalty}")

    scn = {s: load_scenario(s) for s in set(["dry"] + args.eval_scenarios + args.train_scenarios)}
    Zs = {s: est.standardise(scn[s]["X"]) for s in scn}
    Z, dY = Zs["dry"], scn["dry"]["dY"]
    N = len(Z)
    trs = args.train_scenarios
    if any(len(Zs[s]) != N for s in trs):
        raise SystemExit("training scenarios must all have the same length")
    Zt = np.concatenate([Zs[s] for s in trs]) if len(trs) > 1 else Z
    dYt = np.concatenate([scn[s]["dY"] for s in trs]) if len(trs) > 1 else dY
    t_lo, t_hi = max(C.SPLITS["train"][0], H), C.SPLITS["train"][1]
    d_in = build_obs(Z, np.array([H]), np.zeros((1, H, C.N_SENSORS)), use_tod, use_mask).shape[2]

    # reward scale: mean normalised squared error of a random scheduler on the training period
    rng0 = np.random.default_rng(0)
    tmp = VecEnv(Zt, dYt, est, CM, 32, H, t_lo, t_hi, rng0, np.ones(2), 0.0, use_tod, use_mask,
                 n_scn=len(trs), n_len=N)
    e2s = [tmp.step(rng0.integers(0, nA, size=32))[3] for _ in range(300)]
    ref = np.mean(np.concatenate(e2s), axis=0)
    print(f"reward scale (random scheduler, train): {np.round(ref, 4)}")

    class QNet(nn.Module):
        def __init__(self):
            super().__init__()
            if args.arch == "mlp":
                self.body = nn.Sequential(nn.Flatten(), nn.Linear(H * d_in, 256), nn.ReLU(),
                                          nn.Linear(256, 128), nn.ReLU())
            else:
                self.gru = nn.GRU(d_in, 128, batch_first=True)
                self.post = nn.Sequential(nn.Linear(128, 128), nn.ReLU())
            self.head = nn.Linear(128, nA)

        def forward(self, x):
            if args.arch == "mlp":
                h = self.body(x)
            else:
                out, _ = self.gru(x)
                h = self.post(out[:, -1])
            return self.head(h)

    @torch.no_grad()
    def rollout(q, Zx):
        """Continuous greedy deployment over the whole series (causal)."""
        q.eval()
        Nx = len(Zx)
        MH = np.zeros((1, H, C.N_SENSORS))
        Ms, acts = np.zeros((Nx, C.N_SENSORS)), np.zeros(Nx, int)
        for t in range(H, Nx):
            o = build_obs(Zx, np.array([t]), MH, use_tod, use_mask)
            a = int(q(torch.from_numpy(o)).argmax(1))
            Ms[t], acts[t] = CM[a], a
            MH = np.concatenate([MH[:, 1:], CM[a][None, None, :]], axis=1)
        q.train()
        return Ms, acts

    def val_metric(q):
        vs, rs = [], []
        for s in trs:                                   # validation period of every training influent
            Zv, dYv = Zs[s], scn[s]["dY"]
            Ms, _ = rollout(q, Zv)
            r, zc = score(est, Zv, dYv, Ms, "val")
            vs.append(float((r / zc).sum()))
            rs.append(r)
        return float(np.mean(vs)), np.mean(rs, axis=0)

    rows = []
    for seed in args.seeds:
        t0 = time.time()
        rng = np.random.default_rng(seed)
        torch.manual_seed(seed)
        q = QNet()
        qt = copy.deepcopy(q)
        opt = torch.optim.Adam(q.parameters(), lr=args.lr)
        env = VecEnv(Zt, dYt, est, CM, args.n_env, H, t_lo, t_hi, rng, ref, args.switch_penalty, use_tod, use_mask,
                     fault=args.fault_train, n_scn=len(trs), n_len=N)

        cap = 60000
        bo = np.zeros((cap, H, d_in), np.float32)
        bo2 = np.zeros((cap, H, d_in), np.float32)
        ba, br, bd = np.zeros(cap, np.int64), np.zeros(cap, np.float32), np.zeros(cap, np.float32)
        ptr, size = 0, 0

        n_vec = args.steps // args.n_env
        eps_end_it = int(0.5 * n_vec)
        best_val, best_state, best_it, hist = np.inf, None, 0, []
        for it in range(n_vec):
            eps = max(0.05, 1.0 - 0.95 * it / max(1, eps_end_it))
            o = env.obs()
            with torch.no_grad():
                greedy = q(torch.from_numpy(o)).argmax(1).numpy()
            a = np.where(rng.random(args.n_env) < eps, rng.integers(0, nA, size=args.n_env), greedy)
            r, o2, done, _ = env.step(a)
            for k in range(args.n_env):
                bo[ptr], bo2[ptr], ba[ptr], br[ptr], bd[ptr] = o[k], o2[k], a[k], r[k], float(done[k])
                ptr = (ptr + 1) % cap
                size = min(size + 1, cap)
            if size >= 2000:
                for _ in range(args.updates):
                    j = rng.integers(0, size, size=args.batch)
                    s_, a_, r_, s2_, d_ = (torch.from_numpy(x[j]) for x in (bo, ba, br, bo2, bd))
                    with torch.no_grad():
                        a2 = q(s2_).argmax(1, keepdim=True)                      # double DQN
                        y = r_ + args.gamma * (1 - d_) * qt(s2_).gather(1, a2).squeeze(1)
                    loss = F.smooth_l1_loss(q(s_).gather(1, a_[:, None]).squeeze(1), y)
                    opt.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(q.parameters(), 10.0)
                    opt.step()
                    with torch.no_grad():
                        for p, pt in zip(q.parameters(), qt.parameters()):
                            pt.mul_(0.995).add_(0.005 * p)
            if (it + 1) % args.eval_every == 0 or it + 1 == n_vec:
                v, rv = val_metric(q)
                hist.append((it + 1, v))
                if v < best_val:
                    best_val, best_it = v, it + 1
                    best_state = {k: x.clone() for k, x in q.state_dict().items()}
                print(f"  seed {seed} step {(it + 1) * args.n_env:>7d} eps {eps:.2f} val dNH4 {rv[0]:.4f} dSNO {rv[1]:.4f} "
                      f"score {v:.4f}{'  *best' if best_it == it + 1 else ''}", flush=True)

        q.load_state_dict(best_state)
        C.MODELS_DIR.mkdir(exist_ok=True)
        torch.save({"state": best_state, "arch": args.arch, "hist": H, "tod": use_tod, "mask": use_mask,
                    "d_in": d_in, "n_actions": nA, "estimator": tag, "budget": B, "seed": seed,
                    "val_score": float(best_val), "best_step": best_it * args.n_env},
                   C.MODELS_DIR / f"agent_{name}_{tag}_s{seed}.pt")
        for sname in args.eval_scenarios:
            Zx, dYx = Zs[sname], scn[sname]["dY"]
            Ms, acts = rollout(q, Zx)
            np.save(C.RESULTS_DIR / f"agent_sched_{name}_{tag}_{sname}_s{seed}.npy", acts)
            row = {"agent": name, "estimator": tag, "arch": args.arch, "budget": B, "seed": seed, "scenario": sname, "hist": H,
                   "tod": use_tod, "mask": use_mask, "switch_penalty": args.switch_penalty, "best_step": best_it * args.n_env}
            for split in ("val", "test"):
                r, zc = score(est, Zx, dYx, Ms, split)
                row[f"{split}_dNH4"], row[f"{split}_dSNO"] = r
                row[f"{split}_zc_dNH4"], row[f"{split}_zc_dSNO"] = zc
            tt = t_range("test")
            row["switch_rate_test"] = float(np.mean(acts[tt][1:] != acts[tt][:-1]))
            row["distinct_configs_test"] = int(len(set(acts[tt])))
            rows.append(row)
            print(f"seed {seed} [{sname}] test dNH4 {row['test_dNH4']:.4f} dSNO {row['test_dSNO']:.4f} | "
                  f"switch rate {row['switch_rate_test']:.2f}, {row['distinct_configs_test']} distinct subsets "
                  f"| {time.time() - t0:.0f}s", flush=True)

    out = C.RESULTS_DIR / "agent_runs.csv"
    df = pd.DataFrame(rows)
    if out.exists():
        df = pd.concat([pd.read_csv(out), df], ignore_index=True)
    df.to_csv(out, index=False)

    # ---------------- comparison with the baselines of the same budget
    bpath = C.RESULTS_DIR / f"baselines_{tag}.csv"
    mine = pd.DataFrame(rows)
    mine = mine[mine.scenario == "dry"]
    if len(mine) and bpath.exists():
        bl = pd.read_csv(bpath)
        bl = bl[(bl.scenario == "dry") & (bl.split == "test")].set_index("method")
        print(f"\n=== dry test, budget {B}: {name}, {len(mine)} seed(s) vs baselines ({bpath.name}) ===")
        print(f"{'method':<52}{'dNH4':>10}{'dSNO':>10}")
        m, s = mine[["test_dNH4", "test_dSNO"]].mean(), mine[["test_dNH4", "test_dSNO"]].std(ddof=0)
        print(f"{'AGENT ' + name + ' (mean over seeds)':<52}{m.iloc[0]:>10.4f}{m.iloc[1]:>10.4f}")
        if len(mine) > 1:
            print(f"{'   sd over seeds':<52}{s.iloc[0]:>10.4f}{s.iloc[1]:>10.4f}")
        for key in ["static: best on dry val", "round-robin", "random scheduler", "fixed %d: best" % B,
                    "temporal oracle", "all 12 sensors (mixed", "zero-change"]:
            for meth in [x for x in bl.index if x.startswith(key)]:
                print(f"{meth:<52}{bl.loc[meth, 'rmse_dNH4']:>10.4f}{bl.loc[meth, 'rmse_dSNO']:>10.4f}")
        st = bl.loc["static: best on dry val"]
        print(f"agent vs static-on-validation: dNH4 {100 * (m.iloc[0] / st['rmse_dNH4'] - 1):+.1f}%  "
              f"dSNO {100 * (m.iloc[1] / st['rmse_dSNO'] - 1):+.1f}%   (negative = agent better)")


if __name__ == "__main__":
    main()
# RETRAIN-PATCH
