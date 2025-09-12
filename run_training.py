import subprocess

# 属性和模型
#PROPERTIES = ["Tg", "FFV", "Tc", "Density", "Rg"]
PROPERTIES = ["Tg"]
#MODELS = ["xgb", "lgb", "cat"]
MODELS = ["xgb"]

train_path = "datasets/train_orig_vanda1.csv"
test_path = "datasets/test_orig_vanda1.csv"
folds = 10

for _ in range (2):
    for model in MODELS:
        for target in PROPERTIES:
            config = f"configs/best_{model}_{target}.json"
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