p = "export_bsm1.py"
s = open(p, encoding="utf-8").read()
pairs = []
pairs.append(("def build_input(dyn, mean_row, spin_days):\n", '''def perturb(dry, k):
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
'''))
pairs.append(('''    ap.add_argument("--rain-file", default=None, help="optional path to an official rain influent file")
    args = ap.parse_args()
''', '''    ap.add_argument("--rain-file", default=None, help="optional path to an official rain influent file")
    ap.add_argument("--variants", type=int, default=0, help="also make N perturbed dry influents: var1 ... varN")
    ap.add_argument("--only-variants", action="store_true", help="skip the dry and rain scenarios")
    args = ap.parse_args()
'''))
pairs.append(('''    for name in args.scenarios:
''', '''    names = [] if args.only_variants else list(args.scenarios)
    names += [f"var{k}" for k in range(1, args.variants + 1)]
    for name in names:
'''))
pairs.append(('''        if name == "dry":
            dyn = dry
        else:
''', '''        if name == "dry":
            dyn = dry
        elif name.startswith("var"):
            dyn = perturb(dry, int(name[3:]))
        else:
'''))
for old, new in pairs:
    assert s.count(old) == 1, "patch text not found - tell Claude: " + old[:60]
    s = s.replace(old, new)
open(p, "w", encoding="utf-8").write(s)
print("patched OK")