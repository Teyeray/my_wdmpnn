import torch
from torch.optim.lr_scheduler import CosineAnnealingLR
import numpy as np
from typing import Dict, List
import pandas as pd


# ------------------ wMAE Loss ------------------
class WMAELoss(torch.nn.Module):
    def __init__(self, tasks: List[str], n_dict: Dict[str, int], r_dict: Dict[str, float]):
        super().__init__()
        self.tasks = tasks
        self.weights = self._compute_task_weights(n_dict, r_dict, tasks)

    def _compute_task_weights(self, n_dict, r_dict, tasks):
        K = len(tasks)
        inv_sqrt_ns = [(1.0 / np.sqrt(max(n_dict[t], 1))) for t in tasks]
        denom = sum(inv_sqrt_ns)
        weights = {}
        for i, t in enumerate(tasks):
            ri = max(r_dict[t], 1e-12)
            wi = (1.0 / ri) * (K * inv_sqrt_ns[i] / denom)
            weights[t] = wi
        return weights

    def forward(self, outputs, targets, return_per_task=False):
        total = None
        count = 0
        per_task = {}
        for t in self.tasks:
            yhat, y = outputs[t], targets[t]
            mask = torch.isfinite(y)   # 只在非 NaN 样本上算
            if mask.sum() == 0:
                continue
            err = (yhat[mask] - y[mask]).abs().mean()
            weighted = self.weights[t] * err
            total = weighted if total is None else total + weighted
            count += 1
            per_task[t] = err.item()

        if total is None:
            loss = torch.tensor(0.0, requires_grad=True)
        else:
            loss = total / max(count, 1)

        if return_per_task:
            return loss, per_task
        return loss


def compute_task_stats(df: pd.DataFrame, tasks: List[str]):
    """
    计算 n_dict 和 r_dict
    - n_dict[t]: 非缺失样本数
    - r_dict[t]: 取值范围 (max-min)
    """
    n_dict, r_dict = {}, {}
    for t in tasks:
        vals = df[t].dropna().values
        n_dict[t] = len(vals)
        if len(vals) > 0:
            r_dict[t] = float(np.nanmax(vals) - np.nanmin(vals))
        else:
            r_dict[t] = 1.0
    return n_dict, r_dict


# ------------------ EarlyStopping ------------------
class EarlyStopping:
    def __init__(self, patience=20, verbose=True):
        self.patience = patience
        self.best_loss = float("inf")
        self.counter = 0
        self.early_stop = False
        self.verbose = verbose

    def step(self, val_loss):
        if val_loss < self.best_loss:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.verbose:
                print(f"EarlyStopping counter: {self.counter}/{self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True


# ------------------ Train / Eval ------------------
def train_one_epoch(model, loader, optimizer, loss_fn, device, tasks):
    model.train()
    total_loss = 0.0
    per_task_accum = {t: 0.0 for t in tasks}
    n_batches = 0

    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        outputs = model(batch)  # dict {task: [B]}

        targets = {t: batch.y[:, i] for i, t in enumerate(tasks)}
        loss, per_task = loss_fn(outputs, targets, return_per_task=True)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        for t in per_task:
            per_task_accum[t] += per_task[t]
        n_batches += 1

    avg_loss = total_loss / n_batches
    avg_task_loss = {t: per_task_accum[t] / n_batches for t in tasks}
    return avg_loss, avg_task_loss


@torch.no_grad()
def evaluate(model, loader, loss_fn, device, tasks):
    model.eval()
    total_loss = 0.0
    per_task_accum = {t: 0.0 for t in tasks}
    n_batches = 0

    for batch in loader:
        batch = batch.to(device)
        outputs = model(batch)

        targets = {t: batch.y[:, i] for i, t in enumerate(tasks)}
        loss, per_task = loss_fn(outputs, targets, return_per_task=True)
        total_loss += loss.item()
        for t in per_task:
            per_task_accum[t] += per_task[t]
        n_batches += 1

    avg_loss = total_loss / n_batches
    avg_task_loss = {t: per_task_accum[t] / n_batches for t in tasks}
    return avg_loss, avg_task_loss


# ------------------ Main Loop ------------------
def run_training(model, train_loader, val_loader,
                 optimizer, tasks, train_df,
                 device="cuda", max_epochs=100, patience=20):
    # 1) 准备 wMAE Loss
    n_dict, r_dict = compute_task_stats(train_df, tasks)
    loss_fn = WMAELoss(tasks, n_dict, r_dict)

    # 2) Scheduler + EarlyStopping
    scheduler = CosineAnnealingLR(optimizer, T_max=max_epochs)
    early_stopper = EarlyStopping(patience=patience)

    history = []
    # 3) Loop
    for epoch in range(1, max_epochs + 1):
        train_loss, train_task_loss = train_one_epoch(model, train_loader, optimizer, loss_fn, device, tasks)
        val_loss, val_task_loss = evaluate(model, val_loader, loss_fn, device, tasks)
        scheduler.step()

        # 打印结果
        task_str = " | ".join([f"{t}: {val_task_loss[t]:.4f}" for t in tasks])
        print(f"Epoch {epoch:03d}: "
              f"Train={train_loss:.4f}, Val={val_loss:.4f} || {task_str}")

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_task_loss": val_task_loss,
        })
        
        early_stopper.step(val_loss)
        if early_stopper.early_stop:
            print("Early stopping triggered.")
            break

    return model