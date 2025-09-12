import subprocess

# 属性和模型
PROPERTIES = ["Tg", "FFV", "Tc", "Density", "Rg"]
MODELS = ["xgb", "lgb", "cat"]

train_path = "datasets/train_orig_testing1.csv"
test_path = "datasets/test_orig_testing1.csv"
folds = 5

for model in MODELS:
    for target in PROPERTIES:

        config = f"configs/{model}_base.json"
        cmd = [
            "python", "-m", "tools.train_model",
            "--model", model,
            "--target", target,
            "--config", config,
            "--train-path", train_path,
            "--test-path", test_path,
            "--folds", str(folds)
        ]
        print(">>> Running:", " ".join(cmd))
        subprocess.run(cmd, check=True)