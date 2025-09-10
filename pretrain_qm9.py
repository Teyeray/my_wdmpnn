import os
import torch
import optuna
import datetime
import pandas as pd
from data import load_qm9
from model import WDMPNNModel
from torch_geometric.loader import DataLoader
from train import run_training, evaluate, compute_task_stats, WMAELoss

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
            "128-64",
            "256-128",
            "256-128-64",
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


# -------------------- Optuna 目标函数 --------------------
def make_objective(train_loader, val_loader, df, tasks, dataset, max_epochs=30, patience=10):
    def objective(trial):
        params = suggest_params(trial)

        mlp_hidden = list(map(int, params["mlp_hidden"].split("-")))
        node_dim = dataset.num_node_features
        edge_dim = dataset.num_edge_features


        train_loader = DataLoader(train_ds, batch_size=params["batch_size"], shuffle=True)
        val_loader   = DataLoader(val_ds, batch_size=params["batch_size"], shuffle=False)
        
        # === 模型 ===
        model = WDMPNNModel(
            node_dim=node_dim,
            edge_dim=edge_dim,
            hidden_dim=params["hidden_dim"],
            num_layers=params["num_layers"],
            tasks=tasks,
            mlp_hidden=mlp_hidden,
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
        model, _ = run_training(
            model,
            train_loader,
            val_loader,
            optimizer,
            tasks,
            df,
            device=device,
            max_epochs=max_epochs,
            patience=patience,
        )

        # === 验证集总 loss ===
        n_dict, r_dict = compute_task_stats(df, tasks)
        loss_fn = WMAELoss(tasks, n_dict, r_dict)
        val_loss, val_task_loss = evaluate(model, val_loader, loss_fn, device, tasks)

        # === 保存权重 ===
        os.makedirs("checkpoints", exist_ok=True)
        timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        trial_dir = os.path.join(
            "checkpoints", f"{trial.study.study_name}_trial{trial.number}_{timestamp}"
        )
        os.makedirs(trial_dir, exist_ok=True)

        model.save_encoder(os.path.join(trial_dir, "encoder.pt"))
        model.save_adapter(os.path.join(trial_dir, "adapter.pt"))
        model.save_head(os.path.join(trial_dir, "head.pt"))
        model.save_full(os.path.join(trial_dir, "full.pt"))

        print(f"[Trial {trial.number}] Weights saved in {trial_dir} (val_loss={val_loss:.4f})")

        # === 打印 trial summary ===
        task_str = " | ".join([f"{t}: {val_task_loss[t]:.4f}" for t in tasks])
        print(f"[Trial {trial.number}] val_loss={val_loss:.4f} || {task_str}")

        if trial.number > 0:
            try:
                best_trial = trial.study.best_trial
                if best_trial is not None and best_trial.number != trial.number:
                    print(f"[Best so far] Trial {best_trial.number}: val_loss={best_trial.value:.4f}")
            except ValueError:
                pass

        return val_loss
    return objective


# -------------------- 主入口 --------------------
if __name__ == "__main__":
    train_ds, val_ds, df, tasks, dataset = load_qm9(batch_size=512, num_workers=4, root="kaggle/input/my-qm9/qm9")

    study = optuna.create_study(
        study_name="qm9_pretrain_study",
        storage="sqlite:///optuna_qm9_pretrain.db",
        load_if_exists=True,
        direction="minimize",
    )
    study.optimize(
        make_objective(
            train_ds, val_ds, df, tasks, dataset,
            max_epochs=2,
            patience=1
        ),
        n_trials=5,
    )

    print("Best trial params:", study.best_trial.params)
    print("Best val_loss:", study.best_trial.value)