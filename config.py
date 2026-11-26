"""config.py - every project setting lives here. Change settings here, nowhere else."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RAW_DIR = ROOT / "data" / "raw"
RESULTS_DIR = ROOT / "results"
MODELS_DIR = ROOT / "models"

STEPS_PER_DAY = 96          # one scheduling decision every 15 minutes
HORIZON = 2                 # predict the change over 2 steps = 30 minutes
EST_STEPS = 5               # estimator sees the current step and the 4 previous steps
BUDGET = int(os.environ.get("WWTP_BUDGET", "4"))   # sensors active at every decision (use 1, 2, 3, 4 or 6)
                                                   # PowerShell example: $env:WWTP_BUDGET="2"
N_SENSORS = 12

# Two candidate sensor pools (12 sensors each).
#  "recommended": drops DO in the two unaerated reactors (R1, R2), which are ~constant at ~0,
#                 and uses nitrate (SNO) in R2-R5 instead.
#  "paper1":      the pool used in the first manuscript.
POOLS = {
    "recommended": ["DO_R3", "DO_R4", "DO_R5",
                    "NH_R1", "NH_R2", "NH_R3", "NH_R4", "NH_R5",
                    "SNO_R2", "SNO_R3", "SNO_R4", "SNO_R5"],
    "paper1": ["DO_R1", "DO_R2", "DO_R3", "DO_R4", "DO_R5",
               "NH_R1", "NH_R2", "NH_R3", "NH_R4", "NH_R5",
               "SNO_R3", "SNO_R5"],
}
POOL_NAME = os.environ.get("WWTP_POOL", "recommended")   # PowerShell: $env:WWTP_POOL="paper1"
POOL = POOLS[POOL_NAME]

TARGETS = ["eff_SNH", "eff_SNO"]          # effluent NH4-N and nitrate/nitrite
TARGET_LABELS = ["dNH4", "dSNO"]

# Decision indices t (inclusive). A decision at t needs t-4 >= 0 and t+2 <= 1343.
SPLITS = {"train": (4, 690), "val": (691, 862), "test": (863, 1340)}

SEEDS = [11, 22, 33, 44, 55, 66, 77, 88, 99, 111]