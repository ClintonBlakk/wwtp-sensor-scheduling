p = "03_train_agent.py"
s = open(p, encoding="utf-8").read()
old1 = """def err_for_masks(est, Z, dY, t, Mseq):
    M = Mseq[t[:, None] + OFFS[None, :]]
    return est.predict(Z, t, M) - dY[t]
"""
new1 = """def pred_for_masks(est, Z, t, Mseq):
    M = Mseq[t[:, None] + OFFS[None, :]]
    return est.predict(Z, t, M)
"""
old2 = "    r = rmse(err_for_masks(est, Z, dY, t, Ms), dY[t])\n"
new2 = "    r = rmse(pred_for_masks(est, Z, t, Ms), dY[t])\n"
assert old1 in s and old2 in s, "patch text not found - tell Claude"
s = s.replace(old1, new1).replace(old2, new2)
open(p, "w", encoding="utf-8").write(s)
print("patched OK")