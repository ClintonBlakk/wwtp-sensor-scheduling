"""patch_retrain.py - adds the retraining experiment to the existing scripts (makes *.bak copies first).

  export_bsm1.py       --first-variant K   (make only var K ... var N, so var1-var8 are not simulated again)
  01_train_estimator.py --name TAG         (choose the estimator file name, e.g. pool_e1)
  03_train_agent.py    --train-scenarios   (train the agent on several influents; default 'dry' = unchanged behaviour)
  07_generalise.py     --out-suffix TEXT   (so new results do not overwrite generalise_variants.csv)

Run once from the project root:   python patch_retrain.py
"""
import shutil
import sys


def patch(path, pairs):
    s = open(path, encoding="utf-8").read()
    if "# RETRAIN-PATCH" in s:
        print(f"{path}: already patched, skipped")
        return
    for old, new in pairs:
        if s.count(old) != 1:
            sys.exit(f"{path}: expected text not found exactly once:\n{old[:120]}...")
        s = s.replace(old, new)
    shutil.copy(path, path + ".bak")
    s = s.rstrip("\n") + "\n# RETRAIN-PATCH\n"
    open(path, "w", encoding="utf-8").write(s)
    print(f"{path}: patched OK")


patch("export_bsm1.py", [
    ('    ap.add_argument("--only-variants", action="store_true", help="skip the dry and rain scenarios")\n',
     '    ap.add_argument("--only-variants", action="store_true", help="skip the dry and rain scenarios")\n'
     '    ap.add_argument("--first-variant", type=int, default=1, help="first variant number to make (default 1)")\n'),
    ('    names += [f"var{k}" for k in range(1, args.variants + 1)]\n',
     '    names += [f"var{k}" for k in range(args.first_variant, args.variants + 1)]\n'),
])

patch("01_train_estimator.py", [
    ('    ap.add_argument("--seed", type=int, default=0)\n',
     '    ap.add_argument("--seed", type=int, default=0)\n'
     '    ap.add_argument("--name", default=None, help="estimator name (file tag), e.g. pool_e1; default is built from the scenarios")\n'),
    ('    p_full, p_fixed = (1.0, 0.0) if args.full_info else (0.10, 0.45)\n',
     '    if args.name:\n        tag = est_tag(args.name)\n'
     '    p_full, p_fixed = (1.0, 0.0) if args.full_info else (0.10, 0.45)\n'),
])

patch("03_train_agent.py", [
    ('    def __init__(self, Z, dY, est, CM, n_env, H, t_lo, t_hi, rng, ref, lam, use_tod, use_mask, fault=False):\n',
     '    def __init__(self, Z, dY, est, CM, n_env, H, t_lo, t_hi, rng, ref, lam, use_tod, use_mask, fault=False,\n'
     '                 n_scn=1, n_len=0):\n'),
    ('        self.n, self.H, self.t_lo, self.t_hi = n_env, H, t_lo, t_hi\n',
     '        self.n, self.H, self.t_lo, self.t_hi = n_env, H, t_lo, t_hi\n'
     '        self.n_scn, self.n_len = n_scn, n_len             # several influents stacked one after the other\n'
     '        self.hi = np.zeros(n_env, int)                    # last training step of each environment\n'),
    ('        self.t[idx] = self.rng.integers(self.t_lo, self.t_hi + 1, size=len(idx))\n',
     '        base = (self.rng.integers(0, self.n_scn, size=len(idx)) if self.n_scn > 1 else np.zeros(len(idx), int)) * self.n_len\n'
     '        self.t[idx] = base + self.rng.integers(self.t_lo, self.t_hi + 1, size=len(idx))\n'
     '        self.hi[idx] = base + self.t_hi\n'),
    ('        done = self.t > self.t_hi\n', '        done = self.t > self.hi\n'),
    ('    ap.add_argument("--eval-scenarios", nargs="+", default=["dry"])\n',
     '    ap.add_argument("--eval-scenarios", nargs="+", default=["dry"])\n'
     '    ap.add_argument("--train-scenarios", nargs="+", default=["dry"],\n'
     '                    help="influents the agent trains on (and is validated on); default: dry only")\n'),
    ('    scn = {s: load_scenario(s) for s in set(["dry"] + args.eval_scenarios)}\n',
     '    scn = {s: load_scenario(s) for s in set(["dry"] + args.eval_scenarios + args.train_scenarios)}\n'),
    ('    Z, dY = Zs["dry"], scn["dry"]["dY"]\n    N = len(Z)\n',
     '    Z, dY = Zs["dry"], scn["dry"]["dY"]\n    N = len(Z)\n'
     '    trs = args.train_scenarios\n'
     '    if any(len(Zs[s]) != N for s in trs):\n'
     '        raise SystemExit("training scenarios must all have the same length")\n'
     '    Zt = np.concatenate([Zs[s] for s in trs]) if len(trs) > 1 else Z\n'
     '    dYt = np.concatenate([scn[s]["dY"] for s in trs]) if len(trs) > 1 else dY\n'),
    ('    tmp = VecEnv(Z, dY, est, CM, 32, H, t_lo, t_hi, rng0, np.ones(2), 0.0, use_tod, use_mask)\n',
     '    tmp = VecEnv(Zt, dYt, est, CM, 32, H, t_lo, t_hi, rng0, np.ones(2), 0.0, use_tod, use_mask,\n'
     '                 n_scn=len(trs), n_len=N)\n'),
    ('    def val_metric(q):\n        Ms, _ = rollout(q, Z)\n        r, zc = score(est, Z, dY, Ms, "val")\n        return float((r / zc).sum()), r\n',
     '    def val_metric(q):\n'
     '        vs, rs = [], []\n'
     '        for s in trs:                                   # validation period of every training influent\n'
     '            Zv, dYv = Zs[s], scn[s]["dY"]\n'
     '            Ms, _ = rollout(q, Zv)\n'
     '            r, zc = score(est, Zv, dYv, Ms, "val")\n'
     '            vs.append(float((r / zc).sum()))\n'
     '            rs.append(r)\n'
     '        return float(np.mean(vs)), np.mean(rs, axis=0)\n'),
    ('        env = VecEnv(Z, dY, est, CM, args.n_env, H, t_lo, t_hi, rng, ref, args.switch_penalty, use_tod, use_mask,\n                     fault=args.fault_train)\n',
     '        env = VecEnv(Zt, dYt, est, CM, args.n_env, H, t_lo, t_hi, rng, ref, args.switch_penalty, use_tod, use_mask,\n'
     '                     fault=args.fault_train, n_scn=len(trs), n_len=N)\n'),
])

patch("07_generalise.py", [
    ('    ap.add_argument("--boot", type=int, default=5000)\n',
     '    ap.add_argument("--boot", type=int, default=5000)\n'
     '    ap.add_argument("--out-suffix", default="", help="added to the result file names, e.g. _retrained")\n'),
    ('"generalise_variants.csv")', 'f"generalise_variants{args.out_suffix}.csv")'),
    ('f"generalise_{s}.csv")', 'f"generalise_{s}{args.out_suffix}.csv")'),
])
