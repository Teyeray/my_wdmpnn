import torch
import optuna
import pandas as pd
from torch_geometric.datasets import QM9
from torch_geometric.loader import DataLoader
from sklearn.model_selection import train_test_split

# ====== 你之前写的工具函数 ======
from train import run_training, evaluate, compute_task_stats, WMAELoss
from model import WDMPNNModel   # 就是你写的带 adapter 的 model.py


device = "cuda" if torch.cuda.is_available() else "cpu"


# -------------------- 参数搜索空间 --------------------
def suggest_params(trial):
    return {
        # === Encoder ===
        "hidden_dim": trial.suggest_categorical("hidden_dim", [128, 256, 512]),
        "num_layers": trial.suggest_int("num_layers", 2, 5),
        "act": trial.suggest_categorical("act", ["relu", "silu", "mish"]),
        "dropout": trial.suggest_float("dropout", 0.0, 0.5),

        # === Attention ===
        "use_edge_attn": trial.suggest_categorical("use_edge_attn", [True, False]),
        "att_hidden": trial.suggest_int("att_hidden", 32, 128),

        # === Pooling ===
        "pool": trial.suggest_categorical("pool", ["mean", "att"]),

        # === Adapter ===
        "adapter_kind": trial.suggest_categorical("adapter_kind", ["none", "linear", "mlp"]),
        "adapter_hidden": trial.suggest_int("adapter_hidden", 16, 128),
        "adapter_dropout": trial.suggest_float("adapter_dropout", 0.0, 0.3),

        # === Head ===
        "mlp_hidden": trial.suggest_categorical("mlp_hidden", [
            (128, 64),
            (256, 128),
            (256, 128, 64),
        ]),
        "head_dropout": trial.suggest_float("head_dropout", 0.0, 0.5),

        # === Optimizer ===
        "lr_encoder": trial.suggest_float("lr_encoder", 1e-5, 5e-4, log=True),
        "lr_adapter": trial.suggest_float("lr_adapter", 1e-4, 1e-3, log=True),
        "lr_head": trial.suggest_float("lr_head", 1e-4, 1e-3, log=True),
        "weight_decay": trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True),

        # === Training ===
        "batch_size": trial.suggest_categorical("batch_size", [32, 64, 128]),
    }


# -------------------- 数据加载 --------------------
def load_qm9(batch_size=64, num_workers=0):
    dataset = QM9(root="kaggle/working/qm9")
    idx = list(range(len(dataset)))
    train_idx, val_idx = train_test_split(idx, test_size=0.1, random_state=42)

    train_ds = dataset[train_idx]
    val_ds = dataset[val_idx]

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    # 转换成 DataFrame 方便统计 n_dict / r_dict
    y = dataset._data.y.numpy()
    columns = [f"task_{i}" for i in range(y.shape[1])]
    df = pd.DataFrame(y, columns=columns)

    return train_loader, val_loader, df, columns, dataset


# -------------------- Optuna 目标函数 --------------------
def objective(trial):
    params = suggest_params(trial)

    # === 数据 ===
    train_loader, val_loader, df, tasks, dataset = load_qm9(batch_size=params["batch_size"])
    node_dim = dataset.num_node_features
    edge_dim = dataset.num_edge_features

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
    model = run_training(
        model,
        train_loader,
        val_loader,
        optimizer,
        tasks,
        df,
        device=device,
        max_epochs=30,   # 可改大一些
        patience=10,
    )

    # === 验证集总 loss ===
    n_dict, r_dict = compute_task_stats(df, tasks)
    loss_fn = WMAELoss(tasks, n_dict, r_dict)
    val_loss, _ = evaluate(model, val_loader, loss_fn, device, tasks)

    return val_loss


# -------------------- 主入口 --------------------
if __name__ == "__main__":
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=50)

    print("Best trial:", study.best_trial.params)
    print("Best val_loss:", study.best_trial.value)