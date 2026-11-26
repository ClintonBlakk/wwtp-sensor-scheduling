import pandas as pd
import glob
pd.set_option("display.width", 200)
files = {"budget 4": "results/baselines_dry.csv"}
for b in (1, 2, 3, 6):
    files[f"budget {b}"] = f"results/baselines_dry_b{b}.csv"
for name, path in files.items():
    df = pd.read_csv(path)
    d = df[(df.scenario == "dry") & (df.split == "test")][["method", "rmse_dNH4", "rmse_dSNO"]]
    print(f"\n##### {name} #####")
    print(d.round(4).to_string(index=False))