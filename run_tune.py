import subprocess

# 属性和模型
PROPERTIES = ["Tg", "FFV", "Tc", "Density", "Rg"]
MODELS = ["xgb", "lgb", "cat"]

train_path = "datasets/train_orig_testing1.csv"
test_path = "datasets/test-orig-testing1.csv"   # 注意这里原命令里是 test-orig-testing1
search_space = "configs/search_space.json"
n_trials = 50

for model in MODELS:
    base_config = f"configs/{model}_base.json"
    for target in PROPERTIES:
        study_name = f"{model}_{target}_finetune_vanda"

        cmd = [
            "python", "-m", "tools.tune",
            "--model", model,
            "--target", target,
            "--base-config", base_config,
            "--search-space", search_space,
            "--train-path", train_path,
            "--test-path", test_path,
            "--n-trials", str(n_trials),
            "--study-name", study_name,
        ]

        print(">>> Running:", " ".join(cmd))
        subprocess.run(cmd, check=True)