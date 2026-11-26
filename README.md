# Budgeted sensor scheduling on BSM1: Deep Q-learning versus tuned fixed rotations

Code for the manuscript *"Budgeted Sensor Scheduling for Effluent Ammonium and Nitrate Estimation in a Simulated Wastewater Plant: Deep Q-Learning versus Tuned Fixed Rotations on BSM1"* (Blakk, Nwulu, Akinlabi).

A frozen estimator predicts the 30-minute change in effluent NH4-N and nitrate from masked readings of a 12-sensor pool. A recurrent Double DQN picks one sensor per 15-minute step (budget K = 1). It is compared with static, random, round-robin, heuristic and validation-tuned rotation schedules and with ridge baselines. Everything is simulated (BSM1, open loop, dry weather, perturbed influents and a storm influent); no plant data are used.

## Requirements
Python 3.12.10, scikit-learn 1.9.1, PyTorch 2.15.0 (CPU), NumPy 2.5.5, pandas 3.1.1 and bsm2-python (see `requirements.txt`). Experiments were run on a Windows laptop without a GPU.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

## Files
| File | Purpose |
|---|---|
| `config.py`, `common.py` | settings (splits, sensor pool, budget) and shared code |
| `export_bsm1.py` | simulates BSM1 influents and writes the raw data (`data/raw/`, not committed) |
| `01_train_estimator.py` | trains the masked-history estimator |
| `02_baselines.py`, `04_rotation_baseline.py` | static, random, round-robin, heuristic and tuned-rotation schedules |
| `03_train_agent.py` | trains the Double DQN scheduler |
| `05_robustness.py` | noise, drift, failed-sensor and dropout tests |
| `06_stats.py` | block-bootstrap statistics (12 h blocks, 5,000 resamples) |
| `07_generalise.py` | perturbed-influent and storm evaluation |
| `08_linear_baseline.py` | ridge baselines |
| `patch_*.py` | small patches applied in sequence while the study developed; `patch_retrain.py` and `patch_fair.py` add the pooled-training and fair-selection options |
| `run_retrain_experiment.ps1` | the pooled-influent retraining experiment, start to finish |
| `results/` | the CSV result tables behind every table and figure |

## Reproducing the results
Set the budget used in the paper and run the scripts from the project root with the virtual environment active:

```powershell
$env:WWTP_BUDGET = "1"
python export_bsm1.py
# then 01 -> 02 -> 04 -> 03 -> 05 -> 06 -> 07 in that order; each script's options are listed with --help
.\run_retrain_experiment.ps1      # pooled retraining; several hours on a CPU laptop
python 07_generalise.py --estimators pool pool_e1 pool_e2 pool_e3 pool_e4 --agents pooled `
  --variants var5 var6 var7 var8 var9 var10 var11 var12 var13 var14 var15 var16 `
  --stress dry rain --select-on dry var1 var2 var3 var4 --out-suffix _fair
```

Seeds are fixed in `config.py` (agents 11, 22, 33; ablations 11, 22). Results will match the paper up to run-to-run differences in PyTorch on different hardware.

## Limitations
Simulation only; one base influent (the held-out influents are perturbations of it); budget of one sensor; open-loop operation. See Section 5 of the paper.

## Citation
Please cite the paper (reference to be added on publication) and this repository.

## Licence
MIT, see `LICENSE`.
