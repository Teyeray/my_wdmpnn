"""
统一评估指标计算，包含竞赛指标 wMAE
"""
import numpy as np
from typing import Dict, List
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """计算回归基础指标"""
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    
    return {
        'mae': mae,
        'mse': mse,
        'rmse': rmse,
        'r2': r2
    }


def compute_wmae(y_true_dict: Dict[str, np.ndarray], 
                 y_pred_dict: Dict[str, np.ndarray], 
                 ranges: Dict[str, float], 
                 counts: Dict[str, int]) -> float:
    """
    计算竞赛 weighted MAE (wMAE)
    
    公式：
    w_i = (1/r_i) * (K * sqrt(1/n_i) / Z)
    其中 Z = sum(sqrt(1/n_j) for j in tasks)
    wMAE = mean_over_samples(sum_i w_i * |y_hat_i - y_i|)
    
    Args:
        y_true_dict: {task: y_true_array}
        y_pred_dict: {task: y_pred_array}  
        ranges: {task: range_value}
        counts: {task: sample_count}
    
    Returns:
        wMAE score
    """
    tasks = list(y_true_dict.keys())
    K = len(tasks)
    
    # 计算 Z = sum(sqrt(1/n_j) for j in tasks)
    Z = sum(np.sqrt(1.0 / counts[task]) for task in tasks)
    
    # 计算权重 w_i
    weights = {}
    for task in tasks:
        r_i = ranges[task]
        n_i = counts[task]
        w_i = (1.0 / r_i) * (K * np.sqrt(1.0 / n_i) / Z)
        weights[task] = w_i
    
    # 计算加权绝对误差
    total_weighted_error = 0.0
    total_samples = 0
    
    for task in tasks:
        y_true = y_true_dict[task]
        y_pred = y_pred_dict[task]
        w_i = weights[task]
        
        # 确保数组长度一致
        assert len(y_true) == len(y_pred), f"Length mismatch for {task}"
        
        weighted_errors = w_i * np.abs(y_pred - y_true)
        total_weighted_error += np.sum(weighted_errors)
        total_samples += len(y_true)
    
    wmae = total_weighted_error / total_samples
    return wmae


def compute_single_wmae(y_true: np.ndarray, y_pred: np.ndarray, 
                       task: str, ranges: Dict[str, float], 
                       counts: Dict[str, int]) -> float:
    """计算单个任务的wMAE（用于CV过程）"""
    y_true_dict = {task: y_true}
    y_pred_dict = {task: y_pred}
    return compute_wmae(y_true_dict, y_pred_dict, ranges, counts)


def print_cv_summary(task: str, model: str, fold_metrics: List[dict], 
                    cv_wmae: float = None) -> None:
    """打印CV汇总结果"""
    print("=" * 60)
    print(f"CV Summary: {model.upper()} - {task}")
    print("=" * 60)
    
    # 打印每折结果
    for i, metrics in enumerate(fold_metrics):
        print(f"Fold {i+1:2d}: MAE={metrics['mae']:.6f} | "
              f"RMSE={metrics['rmse']:.6f} | R²={metrics['r2']:.4f}")
    
    # 计算平均指标
    avg_metrics = {}
    for key in ['mae', 'rmse', 'r2']:
        values = [m[key] for m in fold_metrics]
        avg_metrics[key] = np.mean(values)
        std_metrics = np.std(values)
        print(f"\nCV {key.upper()}: {avg_metrics[key]:.6f} ± {std_metrics:.6f}")
    
    if cv_wmae is not None:
        print(f"CV wMAE: {cv_wmae:.6f}")
    
    print("=" * 60)


def get_default_ranges() -> Dict[str, float]:
    """获取默认的属性范围（从训练数据估计）"""
    return {
        'Tg': 691.400000,
        'Tc': 0.477500,
        'Rg': 20.308271,
        'FFV': 0.336905,
        'Density': 0.981950,
    }

def get_default_counts() -> Dict[str, int]:
    """获取默认的样本数量（需要根据实际数据调整）"""
    return {
        'Tg': 1149,
        'Tc': 857,
        'Rg': 610,
        'FFV': 7886,
        'Density': 1238,
    }