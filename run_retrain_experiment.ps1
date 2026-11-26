# run_retrain_experiment.ps1 - the retraining experiment, start to finish (run from the project root, venv active).
#   .\run_retrain_experiment.ps1
# It can be stopped and restarted: finished estimators/agents are simply trained again, so run it in one go
# (expect several hours; the agent runs dominate).  Every step prints what it is doing.
$ErrorActionPreference = "Stop"
$env:WWTP_BUDGET = "1"
$train  = "dry","var1","var2","var3","var4"                       # influents used for training and validation
$held   = "var5","var6","var7","var8","var9","var10","var11","var12","var13","var14","var15","var16"   # never used for training
$names  = "pool","pool_e1","pool_e2","pool_e3","pool_e4"          # five estimator seeds

Write-Host "== 1. extra independent influents (var9-var16)"
python export_bsm1.py --only-variants --variants 16 --first-variant 9

Write-Host "== 2. linear baselines (fitted on dry only, and on the pooled influents)"
python 08_linear_baseline.py
python 08_linear_baseline.py --train-scenarios $train --out linear_baseline_pooled.csv

Write-Host "== 3. estimators, baselines, rotations and agents trained on the pooled influents"
for ($s = 0; $s -lt 5; $s++) {
    $n = $names[$s]
    Write-Host "---- estimator $n"
    python 01_train_estimator.py --train-scenarios $train --seed $s --name $n
    python 02_baselines.py --estimator $n --scenarios dry
    python 04_rotation_baseline.py --estimator $n
    python 03_train_agent.py --estimator $n --seeds 11 22 33 --steps 40000 --train-scenarios $train --name pooled
}

Write-Host "== 4. evaluation on the held-out influents"
python 07_generalise.py --estimators $names --agents pooled --variants $held --stress rain --out-suffix _retrained

Write-Host "== 5. the original dry-trained agents on the extra influents (more test blocks for the first result)"
python 07_generalise.py --variants var9 var10 var11 var12 var13 var14 var15 var16 --stress rain --out-suffix _extra

Write-Host "== done. Bundle the results:"
Write-Host '   Compress-Archive -Path results\*.csv -DestinationPath results_bundle2.zip -Force'
