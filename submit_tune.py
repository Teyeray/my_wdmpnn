import os
import subprocess

# 属性和模型
PROPERTIES = ["Tg", "FFV", "Tc", "Density", "Rg"]
MODELS = ["xgb", "lgb", "cat"]

train_path = "datasets/train_orig_vanda2.csv"
test_path = "datasets/test-orig_vanda2.csv"
search_space = "configs/search_space.json"
n_trials = 100

pbs_template = """#!/bin/bash
#PBS -N {job_name}
#PBS -P personal-e1350261
#PBS -l select=1:ncpus=32:mem=64gb
#PBS -l walltime=12:00:00
#PBS -j oe
#PBS -o logs/{job_name}.out

set -euo pipefail
cd $PBS_O_WORKDIR


source /scratch/e1350261/venvs/kaggle39/bin/activate


which python
python --version


{cmd}
"""

for model in MODELS:
    base_config = f"configs/{model}_base.json"
    for target in PROPERTIES:
        study_name = f"{model}_{target}_finetune_vanda_2"

        cmd = (
            f"python -m tools.tune "
            f"--model {model} "
            f"--target {target} "
            f"--base-config {base_config} "
            f"--search-space {search_space} "
            f"--train-path {train_path} "
            f"--test-path {test_path} "
            f"--n-trials {n_trials} "
            f"--study-name {study_name}"
        )

        job_name = f"{model}_{target}_2finetune"
        script_name = f"{job_name}.pbs"

        # 写 PBS 脚本
        os.makedirs("logs", exist_ok=True)
        with open(f'script/{script_name}', "w") as f:
            f.write(pbs_template.format(job_name=job_name, cmd=cmd))

        # 提交作业
        print(f">>> Submitting {script_name}")
        subprocess.run(["qsub", script_name])