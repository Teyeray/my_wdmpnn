"""
模型集成脚本
支持简单加权平均和Stacking两种方法
"""
import os
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import KFold
from sklearn.linear_model import Ridge
import warnings

from tools.utils import set_seed, load_json, save_json, setup_logger, ensure_dir
from tools.evaluate import regression_metrics, compute_single_wmae, print_cv_summary, get_default_ranges, get_default_counts

warnings.filterwarnings('ignore')


def load_oof_predictions(target: str, model_list: list, logger):
    """加载所有模型的OOF预测结果"""
    oof_data = {}
    train_ids = None
    
    for model_name in model_list:
        oof_path = f"outputs/oof/{model_name}_{target}.csv"
        
        if not os.path.exists(oof_path):
            logger.warning(f"OOF file not found: {oof_path}")
            continue
            
        df = pd.read_csv(oof_path)
        oof_data[model_name] = df['oof_pred'].values
        
        if train_ids is None:
            train_ids = df['id'].values
        else:
            # 确保ID顺序一致
            assert np.array_equal(train_ids, df['id'].values), f"ID mismatch for {model_name}"
    
    logger.info(f"Loaded OOF from {len(oof_data)} models: {list(oof_data.keys())}")
    return oof_data, train_ids


def load_test_predictions(target: str, model_list: list, logger):
    """加载所有模型的测试集预测结果"""
    test_data = {}
    test_ids = None
    
    for model_name in model_list:
        test_path = f"outputs/preds/{model_name}_{target}.csv"
        
        if not os.path.exists(test_path):
            logger.warning(f"Test prediction file not found: {test_path}")
            continue
            
        df = pd.read_csv(test_path)
        test_data[model_name] = df['pred'].values
        
        if test_ids is None:
            test_ids = df['id'].values
        else:
            # 确保ID顺序一致
            assert np.array_equal(test_ids, df['id'].values), f"ID mismatch for {model_name}"
    
    logger.info(f"Loaded test predictions from {len(test_data)} models")
    return test_data, test_ids


def load_target_values(target: str, train_ids: np.ndarray, logger):
    """加载目标值"""
    try:
        # 首先尝试从data.py获取
        from data import process_train_test_data
        train_df, _ = process_train_test_data()
        train_df = train_df[train_df[target].notna()].copy()
        
        # 按ID排序确保顺序一致
        train_df = train_df.set_index('id').loc[train_ids].reset_index()
        y_true = train_df[target].values
        
    except Exception as e:
        logger.warning(f"Failed to load from data.py: {e}")
        
        # 备选：从CSV文件加载
        train_path = f"datasets/target_datasets/train_{target}.csv"
        if not os.path.exists(train_path):
            raise FileNotFoundError(f"Training data not found: {train_path}")
        
        train_df = pd.read_csv(train_path)
        train_df = train_df.set_index('id').loc[train_ids].reset_index()
        y_true = train_df[target].values
    
    logger.info(f"Loaded {len(y_true)} target values for {target}")
    return y_true


def weighted_average_ensemble(oof_data: dict, test_data: dict, weights: dict, logger):
    """加权平均集成"""
    model_names = list(oof_data.keys())
    
    # 如果没有提供权重，使用等权重
    if not weights:
        weights = {model: 1.0/len(model_names) for model in model_names}
        logger.info("Using equal weights for all models")
    else:
        # 归一化权重
        total_weight = sum(weights.values())
        weights = {k: v/total_weight for k, v in weights.items()}
        logger.info(f"Using custom weights: {weights}")
    
    # 计算加权平均 - OOF
    oof_pred = np.zeros(len(list(oof_data.values())[0]))
    for model_name in model_names:
        if model_name in oof_data and model_name in weights:
            oof_pred += weights[model_name] * oof_data[model_name]
    
    # 计算加权平均 - 测试集
    test_pred = None
    if test_data:
        test_pred = np.zeros(len(list(test_data.values())[0]))
        for model_name in model_names:
            if model_name in test_data and model_name in weights:
                test_pred += weights[model_name] * test_data[model_name]
    
    return oof_pred, test_pred


def stacking_ensemble(oof_data: dict, test_data: dict, target: str, 
                     y_true: np.ndarray, stacking_config: dict, logger):
    """Stacking集成"""
    model_names = list(oof_data.keys())
    n_models = len(model_names)
    
    if n_models < 2:
        raise ValueError("Stacking requires at least 2 base models")
    
    # 准备特征矩阵
    X_stack = np.column_stack([oof_data[model] for model in model_names])
    
    # 获取stacking参数
    meta_model_type = stacking_config.get('meta_model', 'ridge')
    cv_folds = stacking_config.get('cv_folds', 5)
    random_state = stacking_config.get('random_state', 42)
    
    logger.info(f"Stacking with {meta_model_type} meta-model, {cv_folds} CV folds")
    
    # 交叉验证训练meta model
    kf = KFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    
    meta_oof = np.zeros(len(y_true))
    meta_test_preds = []
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(X_stack)):
        X_train, X_val = X_stack[train_idx], X_stack[val_idx]
        y_train, y_val = y_true[train_idx], y_true[val_idx]
        
        # 训练meta model
        if meta_model_type == 'ridge':
            alpha = stacking_config.get('alpha', 1.0)
            meta_model = Ridge(alpha=alpha, random_state=random_state)
        else:
            raise ValueError(f"Unsupported meta model: {meta_model_type}")
        
        meta_model.fit(X_train, y_train)
        
        # 验证集预测
        val_pred = meta_model.predict(X_val)
        meta_oof[val_idx] = val_pred
        
        # 测试集预测
        if test_data:
            X_test = np.column_stack([test_data[model] for model in model_names])
            test_pred = meta_model.predict(X_test)
            meta_test_preds.append(test_pred)
        
        logger.info(f"Stacking fold {fold+1}: trained meta model")
    
    # 平均测试集预测
    final_test_pred = None
    if meta_test_preds:
        final_test_pred = np.mean(meta_test_preds, axis=0)
    
    return meta_oof, final_test_pred


def evaluate_ensemble(target: str, y_true: np.ndarray, oof_pred: np.ndarray, 
                     method: str, logger):
    """评估集成结果"""
    # 基础回归指标
    metrics = regression_metrics(y_true, oof_pred)
    
    # wMAE指标
    ranges = get_default_ranges()
    counts = get_default_counts()
    wmae = compute_single_wmae(y_true, oof_pred, target, ranges, counts)
    
    # 输出结果
    logger.info(f"Ensemble Results ({method}):")
    logger.info(f"  MAE: {metrics['mae']:.6f}")
    logger.info(f"  RMSE: {metrics['rmse']:.6f}")
    logger.info(f"  R²: {metrics['r2']:.6f}")
    logger.info(f"  wMAE: {wmae:.6f}")
    
    return metrics, wmae


def save_ensemble_predictions(target: str, method: str, oof_pred: np.ndarray, 
                            test_pred: np.ndarray, train_ids: np.ndarray, 
                            test_ids: np.ndarray, logger):
    """保存集成预测结果"""
    ensure_dir("outputs/oof")
    ensure_dir("outputs/preds")
    
    # 保存OOF预测
    oof_df = pd.DataFrame({
        'id': train_ids,
        'oof_pred': oof_pred
    })
    oof_path = f"outputs/oof/{method}_{target}.csv"
    oof_df.to_csv(oof_path, index=False)
    logger.info(f"Ensemble OOF saved: {oof_path}")
    
    # 保存测试集预测
    if test_pred is not None:
        test_df = pd.DataFrame({
            'id': test_ids,
            'pred': test_pred
        })
        test_path = f"outputs/preds/{method}_{target}.csv"
        test_df.to_csv(test_path, index=False)
        logger.info(f"Ensemble test predictions saved: {test_path}")


def log_final_result(target: str, method: str, metrics: dict, wmae: float):
    """输出最终结果"""
    result_line = (f"[ENSEMBLE_RESULT] Target={target} | Method={method.upper()} | "
                  f"CV_MAE={metrics['mae']:.6f} | CV_RMSE={metrics['rmse']:.6f} | "
                  f"CV_R2={metrics['r2']:.6f} | CV_wMAE={wmae:.6f}")
    print(result_line)


def main():
    parser = argparse.ArgumentParser(description='Ensemble multiple models')
    parser.add_argument('--target', required=True,
                       choices=['Tg', 'Tc', 'Rg', 'FFV', 'Density'],
                       help='Target variable')
    parser.add_argument('--models', required=True, nargs='+',
                       help='List of model names to ensemble')
    parser.add_argument('--method', default='weighted',
                       choices=['weighted', 'stacking'],
                       help='Ensemble method')
    parser.add_argument('--config', 
                       help='JSON config file for ensemble parameters')
    parser.add_argument('--output-name',
                       help='Custom output name (default: method name)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    
    args = parser.parse_args()
    
    # 设置随机种子
    set_seed(args.seed)
    
    # 设置日志
    method_name = args.output_name or args.method
    log_file = f"outputs/logs/ensemble_{method_name}_{args.target}.log"
    logger = setup_logger(f"ensemble_{method_name}_{args.target}", log_file)
    
    logger.info(f"Starting ensemble: {args.method} for {args.target}")
    logger.info(f"Models: {args.models}")
    
    try:
        # 加载配置
        config = {}
        if args.config:
            config = load_json(args.config)
            logger.info(f"Config loaded: {args.config}")
        
        # 加载OOF和测试集预测
        oof_data, train_ids = load_oof_predictions(args.target, args.models, logger)
        test_data, test_ids = load_test_predictions(args.target, args.models, logger)
        
        if not oof_data:
            raise ValueError("No OOF predictions found")
        
        # 加载真实目标值
        y_true = load_target_values(args.target, train_ids, logger)
        
        # 执行集成
        if args.method == 'weighted':
            weights = config.get('weights', {})
            oof_pred, test_pred = weighted_average_ensemble(
                oof_data, test_data, weights, logger)
        
        elif args.method == 'stacking':
            stacking_config = config.get('stacking', {})
            oof_pred, test_pred = stacking_ensemble(
                oof_data, test_data, args.target, y_true, stacking_config, logger)
        
        else:
            raise ValueError(f"Unknown ensemble method: {args.method}")
        
        # 评估结果
        metrics, wmae = evaluate_ensemble(args.target, y_true, oof_pred, 
                                        args.method, logger)
        
        # 保存预测结果
        save_ensemble_predictions(args.target, method_name, oof_pred, test_pred,
                                train_ids, test_ids, logger)
        
        # 输出最终结果
        log_final_result(args.target, method_name, metrics, wmae)
        
        logger.info("Ensemble completed successfully")
        
    except Exception as e:
        logger.error(f"Ensemble failed: {str(e)}")
        raise


if __name__ == "__main__":
    main()
