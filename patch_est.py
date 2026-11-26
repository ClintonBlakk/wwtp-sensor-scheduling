def patch(path, pairs):
    s = open(path, encoding="utf-8").read()
    for old, new in pairs:
        assert old in s, f"patch text not found in {path}: {old[:50]}"
        s = s.replace(old, new)
    open(path, "w", encoding="utf-8").write(s)

patch("01_train_estimator.py", [(
    '    tag = est_tag(("full_" if args.full_info else "") + "_".join(args.train_scenarios))\n',
    '    tag = est_tag(("full_" if args.full_info else "") + "_".join(args.train_scenarios)\n'
    '                  + (f"_e{args.seed}" if args.seed else ""))\n')])
patch("03_train_agent.py", [
    ('np.save(C.RESULTS_DIR / f"agent_sched_{name}_{sname}_s{seed}.npy", acts)',
     'np.save(C.RESULTS_DIR / f"agent_sched_{name}_{tag}_{sname}_s{seed}.npy", acts)'),
    ('row = {"agent": name, "arch": args.arch,', 'row = {"agent": name, "estimator": tag, "arch": args.arch,'),
    ('bpath = C.RESULTS_DIR / ("baselines_dry.csv" if B == 4 else f"baselines_dry_b{B}.csv")',
     'bpath = C.RESULTS_DIR / f"baselines_{tag}.csv"'),
])
print("patched OK")