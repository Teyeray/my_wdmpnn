import subprocess

# 属性和模型
#PROPERTIES = ["Tg", "FFV", "Tc", "Density", "Rg"]
PROPERTIES = ["Tc"]
MODELS = ["xgb", "lgb", "cat"]
#MODELS = ["cat"]

train_path = "datasets/train_orig_vanda2.csv"
test_path = "datasets/test_orig_vanda2.csv"
folds = 10

for _ in range (1):
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