"""01_train_estimator.py - train and freeze the masked-history estimator.

    python 01_train_estimator.py                       # trained on the dry scenario (default)
    python 01_train_estimator.py --train-scenarios dry rain
    python 01_train_estimator.py --epochs 5 --samples 5000      # quick test
"""
import argparse
import copy
import time

import joblib
import numpy as np
from sklearn.neural_network import MLPRegressor

import config as C
from common import MaskedEstimator, all_configs, build_features, est_tag, load_scenario, rmse, sample_masks, t_range


def make_batch(scens, Zs, split, n, rng, CM, p_full=0.10, p_fixed=0.45):
    """n random (time, acquisition-history) samples drawn evenly from the given scenarios."""
    Fs, Ys = [], []
    which = rng.integers(0, len(scens), size=n)
    tr = t_range(split)
    for k, s in enumerate(scens):
        m = int((which == k).sum())
        if m == 0:
            continue
        t = rng.choice(tr, size=m)
        M = sample_masks(rng, m, CM, p_full, p_fixed)
        Fs.append(build_features(Zs[k], t, M))
        Ys.append(s["dY"][t])
    return np.vstack(Fs), np.vstack(Ys)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-scenarios", nargs="+", default=["dry"])
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--samples", type=int, default=50000, help="fresh samples per epoch")
    ap.add_argument("--patience", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--name", default=None, help="estimator name (file tag), e.g. pool_e1; default is built from the scenarios")
    ap.add_argument("--full-info", action="store_true",
                    help="train an all-12-sensor estimator (the fair upper reference)")
    args = ap.parse_args()

    C.MODELS_DIR.mkdir(exist_ok=True)
    C.RESULTS_DIR.mkdir(exist_ok=True)
    tag = est_tag(("full_" if args.full_info else "") + "_".join(args.train_scenarios)
                  + (f"_e{args.seed}" if args.seed else ""))
    if args.name:
        tag = est_tag(args.name)
    p_full, p_fixed = (1.0, 0.0) if args.full_info else (0.10, 0.45)
    print(f"pool = {C.POOL_NAME}: {C.POOL}")
    print(f"budget = {C.BUDGET} sensors active; estimator tag = {tag}")
    scens = [load_scenario(n) for n in args.train_scenarios]
    _, CM = all_configs()

    # standardisation statistics from the TRAINING period only
    tr = t_range("train")
    Xtr = np.vstack([s["X"][tr] for s in scens])
    mu, sd = Xtr.mean(axis=0), Xtr.std(axis=0)
    sd = np.where(sd < 1e-6, 1.0, sd)                      # protects near-constant sensors
    dy_sd = np.vstack([s["dY"][tr] for s in scens]).std(axis=0)
    Zs = [(s["X"] - mu) / sd for s in scens]

    models = [MLPRegressor(hidden_layer_sizes=(128, 64), activation="relu", solver="adam",
                           learning_rate_init=1e-3, batch_size=256, alpha=1e-4,
                           random_state=args.seed + j) for j in range(2)]

    rng_val = np.random.default_rng(12345)                 # fixed validation set
    Fv, Yv = make_batch(scens, Zs, "val", 20000, rng_val, CM, p_full, p_fixed)
    rng = np.random.default_rng(args.seed)

    best, best_score, best_epoch, bad, curve = None, np.inf, -1, 0, []
    t0 = time.time()
    for ep in range(1, args.epochs + 1):
        F, Y = make_batch(scens, Zs, "train", args.samples, rng, CM, p_full, p_fixed)
        for j in range(2):
            models[j].partial_fit(F, Y[:, j] / dy_sd[j])
        pred = np.column_stack([m.predict(Fv) * k for m, k in zip(models, dy_sd)])
        r = rmse(pred, Yv)
        score = float(np.mean(r / dy_sd))                  # validation error, scaled per target
        curve.append(score)
        if score < best_score - 1e-4:
            best, best_score, best_epoch, bad = copy.deepcopy(models), score, ep, 0
        else:
            bad += 1
        print(f"epoch {ep:3d}  val RMSE dNH4 {r[0]:.4f}  dSNO {r[1]:.4f}  scaled {score:.4f}"
              f"  {'*' if bad == 0 else ''}  ({time.time() - t0:.0f}s)")
        if bad >= args.patience:
            print("early stopping")
            break

    est = MaskedEstimator(mu, sd, dy_sd, best)
    path = C.MODELS_DIR / f"estimator_{tag}.joblib"
    joblib.dump({"estimator": est, "pool": C.POOL, "train_scenarios": args.train_scenarios,
                 "best_epoch": best_epoch, "val_curve": curve, "budget": C.BUDGET, "full_info": args.full_info}, path)
    print(f"\nbest epoch {best_epoch}; saved {path}")

    # quick sanity check on every available scenario
    print("\n=== sanity check (RMSE of the 30-min change) ===")
    for name in ["dry", "rain"]:
        try:
            s = load_scenario(name)
        except FileNotFoundError:
            continue
        Z = est.standardise(s["X"])
        for split in ["val", "test"]:
            t = t_range(split)
            truth = s["dY"][t]
            zc = rmse(np.zeros_like(truth), truth)
            full = est.predict(Z, t, np.ones((len(t), C.EST_STEPS, C.N_SENSORS)))
            rng_c = np.random.default_rng(1)
            rr = []
            for _ in range(0 if args.full_info else 20):
                c = CM[rng_c.integers(len(CM))]
                rr.append(rmse(est.predict(Z, t, np.broadcast_to(c, (len(t), C.EST_STEPS, C.N_SENSORS)).copy()), truth))
            rr = np.mean(rr, axis=0) if rr else np.array([np.nan, np.nan])
            print(f"{name:5s} {split:4s} zero-change {zc.round(4)}  full-12 {rmse(full, truth).round(4)}  "
                  f"20 random fixed configs (mean) {rr.round(4)}")


if __name__ == "__main__":
    main()
# RETRAIN-PATCH
