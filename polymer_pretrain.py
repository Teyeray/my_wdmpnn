import os
import torch
import optuna
import pandas as pd
from torch_geometric.loader import DataLoader

# ====== 你写的工具函数 ======
from data import build_pyg_dataset, filter_train_data, clean_smiles, add_extra_data, replace_all_R_with_C
from train import run_training, evaluate, compute_task_stats, WMAELoss
from model import WDMPNNModel

device = "cuda" if torch.cuda.is_available() else "cpu"

# ================== 参数搜索空间 ==================
def suggest_params(trial):
    return {
        # === Encoder ===
        "hidden_dim": trial.suggest_categorical("hidden_dim", [64, 128, 256, 512]),
        "num_layers": trial.suggest_int("num_layers", 2, 5),
        "act": trial.suggest_categorical("act", ["relu", "silu", "mish"]),
        "dropout": trial.suggest_float("dropout", 0.0, 0.3),

        # === Attention ===
        "use_edge_attn": trial.suggest_categorical("use_edge_attn", [True, False]),
        "att_hidden": trial.suggest_int("att_hidden", 32, 128),

        # === Pooling ===
        "pool": trial.suggest_categorical("pool", ["mean", "att"]),

        # === Adapter ===
        "adapter_kind": trial.suggest_categorical("adapter_kind", ["none", "linear", "mlp"]),
        "adapter_hidden": trial.suggest_int("adapter_hidden", 32,64,128),
        "adapter_dropout": trial.suggest_float("adapter_dropout", 0.0, 0.3),

        # === Head ===
        "mlp_hidden": trial.suggest_categorical("mlp_hidden", [
            (128, 64),
            (256, 128),
            (256, 128, 64),
        ]),
        "head_dropout": trial.suggest_float("head_dropout", 0.0, 0.3),

        # === Optimizer ===
        "lr_encoder": trial.suggest_float("lr_encoder", 1e-5, 5e-4, log=True),
        "lr_adapter": trial.suggest_float("lr_adapter", 5e-5, 5e-3, log=True),
        "lr_head": trial.suggest_float("lr_head", 5e-5, 5e-3, log=True),
        "weight_decay": trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True),

        # === Training ===
        "batch_size": trial.suggest_categorical("batch_size", [32, 64, 128, 256]),
    }


# ================== 数据加载 ==================
def load_polymer(batch_size=64):
    # 1) 加载原始 Kaggle 数据
    train = pd.read_csv("kaggle/input/neurips-open-polymer-prediction-2025/train.csv")
    train = add_extra_data(train)
    train.rename(columns={"SMILES": "SMILES_raw"}, inplace=True)
    train["SMILES"] = train["SMILES_raw"].apply(replace_all_R_with_C)
    train = clean_smiles(train)
    train = filter_train_data(train)

    # 2) 定义任务
    TASKS = ["Tg", "FFV","Tc", "Density", "Rg"]

    # 3) 构建 PyG dataset
    ds = build_pyg_dataset(
        train["SMILES"],
        targets=train[TASKS].values,
        cache_path="train_polymer.pkl",
    )

    # 4) DataLoader
    train_size = int(0.9 * len(ds))
    val_size = len(ds) - train_size
    train_ds, val_ds = torch.utils.data.random_split(ds, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, train, TASKS


# ================== Optuna 目标函数 ==================
def objective(trial):
    params = suggest_params(trial)

    # === 数据 ===
    train_loader, val_loader, df, tasks = load_polymer(batch_size=params["batch_size"])
    node_dim = train_loader.dataset[0].x.size(1)
    edge_dim = train_loader.dataset[0].edge_attr.size(1)

    # === 模型 ===
    model = WDMPNNModel(
        node_dim=node_dim,
        edge_dim=edge_dim,
        hidden_dim=params["hidden_dim"],
        num_layers=params["num_layers"],
        tasks=tasks,
        mlp_hidden=list(params["mlp_hidden"]),
        use_edge_attn=params["use_edge_attn"],
        dropout=params["dropout"],
        act=params["act"],
        pool=params["pool"],
        adapter_kind=params["adapter_kind"],
        adapter_hidden=params["adapter_hidden"],
        adapter_dropout=params["adapter_dropout"],
    ).to(device)

    # === Optimizer ===
    groups = model.param_groups()
    optimizer = torch.optim.Adam([
        {"params": groups["encoder"], "lr": params["lr_encoder"], "weight_decay": params["weight_decay"]},
        {"params": groups["adapter"], "lr": params["lr_adapter"], "weight_decay": params["weight_decay"]},
        {"params": groups["head"], "lr": params["lr_head"], "weight_decay": params["weight_decay"]},
    ])

    # === 训练 ===
    model, history = run_training(
        model,
        train_loader,
        val_loader,
        optimizer,
        tasks,
        df,
        device=device,
        max_epochs=100,
        patience=15,
    )
    for h in history:
        print(f"[Trial {trial.number}] Epoch {h['epoch']}: "
              f"Train={h['train_loss']:.4f}, Val={h['val_loss']:.4f}")

    # === 验证集总 loss ===
    n_dict, r_dict = compute_task_stats(df, tasks)
    loss_fn = WMAELoss(tasks, n_dict, r_dict)
    val_loss, _ = evaluate(model, val_loader, loss_fn, device, tasks)

    # === 保存权重 ===
    os.makedirs("checkpoints", exist_ok=True)
    save_path = f"checkpoints/polymer_trial_{trial.number}.pt"
    torch.save(model.state_dict(), save_path)
    print(f"[Trial {trial.number}] Saved model → {save_path} (val_loss={val_loss:.4f})")

    return val_loss


# ================== 主入口 ==================
if __name__ == "__main__":
    study = optuna.create_study(
        study_name="polymer_pretrain_study",
        storage="sqlite:///optuna_polymer_pretrain.db",
        load_if_exists=True,
        direction="minimize",
    )
    study.optimize(objective, n_trials=50)

    print("Best trial params:", study.best_trial.params)
    print("Best val_loss:", study.best_trial.value)