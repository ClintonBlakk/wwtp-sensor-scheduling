p = "03_train_agent.py"
s = open(p, encoding="utf-8").read()
old = "        q.load_state_dict(best_state)\n"
new = '''        q.load_state_dict(best_state)
        C.MODELS_DIR.mkdir(exist_ok=True)
        torch.save({"state": best_state, "arch": args.arch, "hist": H, "tod": use_tod, "mask": use_mask,
                    "d_in": d_in, "n_actions": nA, "estimator": tag, "budget": B, "seed": seed,
                    "val_score": float(best_val), "best_step": best_it * args.n_env},
                   C.MODELS_DIR / f"agent_{name}_{tag}_s{seed}.pt")
'''
assert s.count(old) == 1, "patch text not found - tell Claude"
open(p, "w", encoding="utf-8").write(s.replace(old, new))
print("patched OK")