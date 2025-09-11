"""
单模型训练脚本
支持 XGBoost / LightGBM / CatBoost 的交叉验证训练
"""
import os
import argparse
import warnings
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import KFold

from tools.utils import set_seed, load_json, save_json, setup_logger, ensure_dir
from tools.evaluate import regression_metrics, compute_single_wmae, print_cv_summary, get_default_ranges, get_default_counts

warnings.filterwarnings('ignore')


def build_model(model_name: str, params: dict):
    """构建指定的模型"""
    if model_name == 'xgb':
        import xgboost as xgb
        return xgb.XGBRegressor(**params)
    elif model_name == 'lgb':
        import lightgbm as lgb
        return lgb.LGBMRegressor(**params, verbose=-1)
    elif model_name == 'cat':
        import catboost as cat
        return cat.CatBoostRegressor(**params, verbose=False)
    else:
        raise ValueError(f"Unsupported model: {model_name}")


def load_data(target: str, logger):
    """加载训练和测试数据，优先使用data.py的处理结果"""
    
    # 方法1：直接从data.py获取处理后的数据（推荐）
    try:
        from data import process_train_test_data
        logger.info("Loading processed data from data.py...")
        train_df, test_df = process_train_test_data()
        
        # 过滤出有目标值的样本
        train_df = train_df[train_df[target].notna()].copy()
        logger.info(f"Successfully loaded data from data.py")
        
    except Exception as e:
        logger.warning(f"Failed to load from data.py: {e}")
        
        # 方法2：从预处理的CSV加载（data.py的输出）
        processed_train_paths = [
            f"datasets/train_orig_testing1.csv",
            f"datasets/train_orig_1.csv", 
            f"datasets/cleaned_train.csv"
        ]
        
        processed_test_paths = [
            f"datasets/test_orig_testing1.csv",
            f"datasets/test_orig_1.csv",
            f"datasets/cleaned_test.csv"
        ]
        
        train_df = None
        test_df = None
        
        # 尝试加载预处理的完整数据
        for train_path in processed_train_paths:
            if os.path.exists(train_path):
                logger.info(f"Loading processed train data from: {train_path}")
                train_df = pd.read_csv(train_path)
                train_df = train_df[train_df[target].notna()].copy()
                break
        
        for test_path in processed_test_paths:
            if os.path.exists(test_path):
                logger.info(f"Loading processed test data from: {test_path}")
                test_df = pd.read_csv(test_path)
                break
        
        # 方法3：从target-specific数据集加载（备选）
        if train_df is None:
            target_train_path = f"datasets/target_datasets/train_{target}.csv"
            if os.path.exists(target_train_path):
                logger.info(f"Loading target-specific data from: {target_train_path}")
                train_df = pd.read_csv(target_train_path)
            else:
                raise FileNotFoundError(
                    f"No training data found. Tried:\n" +
                    f"- data.py: {e}\n" +
                    f"- Processed files: {processed_train_paths}\n" +
                    f"- Target-specific: {target_train_path}"
                )
        
        if test_df is None:
            # 如果没有找到预处理的测试集，尝试原始测试集
            original_test_path = "datasets/test_orig_testing1.csv"
            if os.path.exists(original_test_path):
                logger.warning(f"Using original test data: {original_test_path}")
                test_df = pd.read_csv(original_test_path)
            else:
                logger.warning("No test data found - training only mode")
    
    # 准备特征和标签
    all_targets = ['Tg', 'Tc', 'Rg', 'FFV', 'Density']
    
    # 智能识别特征列（排除已知的非特征列）
    exclude_cols = ['id', 'SMILES'] + all_targets
    feature_cols = [col for col in train_df.columns if col not in exclude_cols]
    
    # 检查是否有有效特征
    if not feature_cols:
        raise ValueError(f"No feature columns found in training data. Available columns: {train_df.columns.tolist()}")
    
    X = train_df[feature_cols].values
    y = train_df[target].values
    train_ids = train_df['id'].values if 'id' in train_df.columns else np.arange(len(train_df))
    
    logger.info(f"Training data loaded:")
    logger.info(f"  - Shape: X={X.shape}, y={y.shape}")
    logger.info(f"  - Target '{target}': {np.sum(~np.isnan(y))} valid samples")
    logger.info(f"  - Features: {len(feature_cols)}")
    logger.info(f"  - Feature types: {train_df[feature_cols].dtypes.value_counts().to_dict()}")
    
    # 处理测试集
    X_test, test_ids = None, None
    if test_df is not None:
        # 确保测试集有训练集的所有特征
        missing_features = [col for col in feature_cols if col not in test_df.columns]
        if missing_features:
            logger.warning(f"Test data missing {len(missing_features)} features - filling with 0")
            for col in missing_features:
                test_df[col] = 0
        
        # 确保测试集特征顺序与训练集一致
        try:
            X_test = test_df[feature_cols].values
            test_ids = test_df['id'].values if 'id' in test_df.columns else np.arange(len(test_df))
            logger.info(f"Test data loaded: {X_test.shape}")
        except KeyError as e:
            logger.error(f"Failed to extract test features: {e}")
            logger.info(f"Available test columns: {test_df.columns.tolist()}")
            X_test, test_ids = None, None
    else:
        logger.info("No test data available")
    
    return X, y, X_test, test_ids, train_ids


def run_cv(model_name: str, target: str, config: dict, n_folds: int, seed: int, logger):
    """执行交叉验证训练"""
    logger.info(f"Starting CV: {model_name.upper()} for {target}")
    
    # 加载数据
    X, y, X_test, test_ids, train_ids = load_data(target, logger)
    
    # 准备配置
    params = config.get('params', {})
    fit_params = config.get('fit', {})
    
    # 交叉验证
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
    
    fold_metrics = []
    oof_predictions = np.zeros(len(y))
    test_predictions = []
    
    # wMAE计算所需参数
    ranges = get_default_ranges()
    counts = get_default_counts()
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
        logger.info(f"Training fold {fold + 1}/{n_folds}")
        
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        
        # 构建和训练模型
        model = build_model(model_name, params)
        
        # 根据不同模型使用不同的训练方式
        if model_name == 'xgb':
            # XGBoost 训练
            model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                early_stopping_rounds=fit_params.get('early_stopping_rounds', 100),
                verbose=False
            )
        elif model_name == 'lgb':
            # LightGBM 训练
            model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                callbacks=[
                    lgb.early_stopping(fit_params.get('early_stopping_rounds', 100)),
                    lgb.log_evaluation(0)  # 关闭训练日志
                ]
            )
        elif model_name == 'cat':
            # CatBoost 训练
            model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                early_stopping_rounds=fit_params.get('early_stopping_rounds', 100),
                verbose=False
            )
        
        # 验证集预测
        val_pred = model.predict(X_val)
        oof_predictions[val_idx] = val_pred
        
        # 计算fold指标
        fold_metric = regression_metrics(y_val, val_pred)
        fold_metrics.append(fold_metric)
        
        logger.info(f"Fold {fold + 1}: MAE={fold_metric['mae']:.6f}")
        
        # 测试集预测
        if X_test is not None:
            test_pred = model.predict(X_test)
            test_predictions.append(test_pred)
        
        # 可选：保存模型
        model_path = f"outputs/models/{model_name}_{target}_fold{fold}.bin"
        ensure_dir(os.path.dirname(model_path))
        try:
            if model_name == 'xgb':
                model.save_model(model_path)
            elif model_name == 'lgb':
                model.booster_.save_model(model_path)
            elif model_name == 'cat':
                model.save_model(model_path)
        except Exception as e:
            logger.warning(f"Failed to save model: {e}")
    
    # 计算CV指标
    cv_metrics = regression_metrics(y, oof_predictions)
    cv_wmae = compute_single_wmae(y, oof_predictions, target, ranges, counts)
    
    # 打印汇总
    print_cv_summary(target, model_name, fold_metrics, cv_wmae)
    
    # 平均测试集预测
    if test_predictions:
        final_test_pred = np.mean(test_predictions, axis=0)
    else:
        final_test_pred = None
    
    return {
        'oof_predictions': oof_predictions,
        'test_predictions': final_test_pred,
        'train_ids': train_ids,
        'test_ids': test_ids,
        'cv_metrics': cv_metrics,
        'cv_wmae': cv_wmae,
        'fold_metrics': fold_metrics
    }


def save_predictions(results: dict, model_name: str, target: str, logger):
    """保存OOF和测试集预测"""
    ensure_dir("outputs/oof")
    ensure_dir("outputs/preds")
    
    # 保存OOF预测
    oof_df = pd.DataFrame({
        'id': results['train_ids'],
        'oof_pred': results['oof_predictions']
    })
    oof_path = f"outputs/oof/{model_name}_{target}.csv"
    oof_df.to_csv(oof_path, index=False)
    logger.info(f"OOF predictions saved: {oof_path}")
    
    # 保存测试集预测
    if results['test_predictions'] is not None:
        test_df = pd.DataFrame({
            'id': results['test_ids'],
            'pred': results['test_predictions']
        })
        test_path = f"outputs/preds/{model_name}_{target}.csv"
        test_df.to_csv(test_path, index=False)
        logger.info(f"Test predictions saved: {test_path}")


def log_final_result(target: str, model_name: str, n_folds: int, results: dict):
    """输出最终结果（供Optuna解析）"""
    cv_metrics = results['cv_metrics']
    cv_wmae = results['cv_wmae']
    
    result_line = (f"[RESULT] Target={target} | Model={model_name.upper()} | "
                  f"Folds={n_folds} | CV_MAE={cv_metrics['mae']:.6f} | "
                  f"CV_RMSE={cv_metrics['rmse']:.6f} | CV_R2={cv_metrics['r2']:.6f} | "
                  f"CV_wMAE={cv_wmae:.6f}")
    
    print(result_line)


def main():
    parser = argparse.ArgumentParser(description='Train single model with CV')
    parser.add_argument('--model', required=True, choices=['xgb', 'lgb', 'cat'],
                       help='Model type')
    parser.add_argument('--target', required=True, 
                       choices=['Tg', 'Tc', 'Rg', 'FFV', 'Density'],
                       help='Target variable')
    parser.add_argument('--config', required=True,
                       help='JSON config file path')
    parser.add_argument('--folds', type=int, default=5,
                       help='Number of CV folds')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    
    args = parser.parse_args()
    
    # 设置随机种子
    set_seed(args.seed)
    
    # 设置日志
    log_file = f"outputs/logs/{args.model}_{args.target}.log"
    logger = setup_logger(f"{args.model}_{args.target}", log_file)
    
    logger.info(f"Starting training: {args.model} for {args.target}")
    logger.info(f"Config: {args.config}")
    logger.info(f"Folds: {args.folds}, Seed: {args.seed}")
    
    try:
        # 加载配置
        config = load_json(args.config)
        logger.info("Config loaded successfully")
        
        # 执行训练
        results = run_cv(args.model, args.target, config, args.folds, args.seed, logger)
        
        # 保存预测结果
        save_predictions(results, args.model, args.target, logger)
        
        # 输出最终结果
        log_final_result(args.target, args.model, args.folds, results)
        
        logger.info("Training completed successfully")
        
    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        raise


if __name__ == "__main__":
    main()