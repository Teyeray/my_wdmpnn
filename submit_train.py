import os
import subprocess

PROPERTIES = ["Tg", "FFV", "Tc", "Density", "Rg"]
#PROPERTIES = ["Tc"]
MODELS = ["xgb", "lgb", "cat"]
#MODELS = ["lgb"]

train_path = "datasets/train_orig_vanda1.csv"
test_path = "datasets/test_orig_vanda1.csv"
folds = 10
N_ITER = 20   # 迭代次数

pbs_template = """#!/bin/bash
#PBS -N {job_name}
#PBS -P personal-e1350261
#PBS -l select=1:ncpus=32:mem=32gb
#PBS -l walltime=12:00:00
#PBS -j oe
#PBS -o logs/{job_name}.out

set -euo pipefail
cd $PBS_O_WORKDIR

source /scratch/e1350261/venvs/ai39/bin/activate

which python
python --version

seed=42

for i in $(seq 1 {n_iter}); do
    echo "[Iter $i] Running with seed=$seed"
    python -m tools.train_model \\
        --model {model} \\
        --target {target} \\
        --config configs/best_{model}_{target}.json \\
        --train-path {train_path} \\
        --test-path {test_path} \\
        --folds {folds} \\
        --seed $seed
done
"""

os.makedirs("logs", exist_ok=True)
os.makedirs("script", exist_ok=True)

for model in MODELS:
    for target in PROPERTIES:
        job_name = f"{model}_{target}_train"
        script_name = f"{job_name}.pbs"

        with open(f'script/{script_name}', "w") as f:
            f.write(pbs_template.format(
                job_name=job_name,
                model=model,
                target=target,
                train_path=train_path,
                test_path=test_path,
                folds=folds,
                n_iter=N_ITER
            ))

        print(f">>> Submitting {script_name}")
        subprocess.run(["qsub", f'script/{script_name}'])
