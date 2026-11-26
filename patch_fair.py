"""patch_fair.py - lets 07_generalise.py choose the static sensor and the rotations on the SAME pooled validation
periods the pooled agents were trained on (fair comparison), instead of on the dry validation period only.

    python patch_fair.py
then (budget 1):
    python 07_generalise.py --estimators pool pool_e1 pool_e2 pool_e3 pool_e4 --agents pooled ^
        --variants var5 ... var16 --stress dry rain --select-on dry var1 var2 var3 var4 --out-suffix _fair
"""
import shutil
import sys

path = "07_generalise.py"
s = open(path, encoding="utf-8").read()
if "# FAIR-PATCH" in s:
    sys.exit("already patched")
pairs = [
    ('def main():\n    ap = argparse.ArgumentParser()\n',
     'def select_on_pool(est, scn_names, CM, max_c=4000):\n'
     '    """Static sensor and period-3/4 rotations chosen on the validation periods of several influents (summed score)."""\n'
     '    R4 = importlib.import_module("04_rotation_baseline")\n'
     '    tv = t_range("val")\n'
     '    S = [load_scenario(n) for n in scn_names]\n'
     '    Zs = [est.standardise(s["X"]) for s in S]\n'
     '    zc = [rmse(np.zeros((len(tv), 2)), s["dY"][tv]) for s in S]\n'
     '    crit = np.zeros(len(CM))\n'
     '    for s, Z, z in zip(S, Zs, zc):\n'
     '        for c in range(len(CM)):\n'
     '            M = np.broadcast_to(CM[c], (len(tv), C.EST_STEPS, C.N_SENSORS))\n'
     '            crit[c] += float((rmse(est.predict(Z, tv, M), s["dY"][tv]) / z).sum())\n'
     '    sidx = int(np.argmin(crit))\n'
     '    rng = np.random.default_rng(0)\n'
     '    cycles = {}\n'
     '    for m in (3, 4):\n'
     '        cyc, _ = R4.candidates(len(CM), m, rng, max_c)\n'
     '        tot = np.zeros(len(cyc))\n'
     '        for s, Z, z in zip(S, Zs, zc):\n'
     '            tot += (R4.eval_cycles(est, Z, s["dY"], tv, CM, cyc) / z).sum(axis=1)\n'
     '        cycles[m] = [int(i) for i in cyc[int(np.argmin(tot))]]\n'
     '    return sidx, cycles\n\n\n'
     'def main():\n    ap = argparse.ArgumentParser()\n'
     '    ap.add_argument("--select-on", nargs="+", default=None,\n'
     '                    help="choose static sensor and rotations on the validation periods of these influents (default: dry only)")\n'),
    ('        agent_sets = {nm: R5.load_agents(tag, nm) for nm in args.agents}\n',
     '        if args.select_on:\n'
     '            sidx, cycles = select_on_pool(est, args.select_on, CM)\n'
     '        agent_sets = {nm: R5.load_agents(tag, nm) for nm in args.agents}\n'),
    ('"static (dry val)"', '"static (val)"'),
    ('f"rotation p{m} (dry val)"', 'f"rotation p{m} (val)"'),
    ('"rotation p3 (dry val)", "rotation p4 (dry val)"', '"rotation p3 (val)", "rotation p4 (val)"'),
]
for old, new in pairs:
    n = s.count(old)
    if n < 1:
        sys.exit(f"expected text not found:\n{old[:100]}")
    s = s.replace(old, new)
shutil.copy(path, path + ".bak2")
open(path, "w", encoding="utf-8").write(s.rstrip("\n") + "\n# FAIR-PATCH\n")
print("07_generalise.py patched OK")
