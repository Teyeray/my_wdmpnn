"""
单模型训练脚本
支持 XGBoost / LightGBM / CatBoost 的交叉验证训练
"""
import os
import glob
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
        cat_params = params.copy()
        if 'verbose' not in cat_params:
            cat_params['verbose'] = False
        return cat.CatBoostRegressor(**cat_params)
    else:
        raise ValueError(f"Unsupported model: {model_name}")


def load_data(target: str, logger, train_path: str, test_path: str = None):
    """加载训练和测试数据从指定的CSV文件路径"""
    
    # 加载训练数据
    try:
        logger.info(f"Loading training data from: {train_path}")
        if not os.path.exists(train_path):
            raise FileNotFoundError(f"Training data file not found: {train_path}")
        
        train_df = pd.read_csv(train_path)
        logger.info(f"Training data loaded: {train_df.shape}")
        
        # 过滤出有目标值的样本
        original_size = len(train_df)
        train_df = train_df[train_df[target].notna()].copy()
        filtered_size = len(train_df)
        
        if filtered_size == 0:
            raise ValueError(f"No valid samples found for target '{target}' in training data")
        
        logger.info(f"Filtered training data: {original_size} -> {filtered_size} samples with valid '{target}' values")
        
    except Exception as e:
        logger.error(f"Failed to load training data: {e}")
        raise
    
    # 加载测试数据（可选）
    test_df = None
    if test_path and os.path.exists(test_path):
        try:
            logger.info(f"Loading test data from: {test_path}")
            test_df = pd.read_csv(test_path)
            logger.info(f"Test data loaded: {test_df.shape}")
        except Exception as e:
            logger.warning(f"Failed to load test data: {e}")
            test_df = None
    elif test_path:
        logger.warning(f"Test data file not found: {test_path}")
    else:
        logger.info("No test data path provided")
        
    
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


def save_best_model_if_improved(model, model_name: str, target: str, current_wmae: float, logger):
    """只有比历史最佳wMAE更好时才保存模型，文件名包含指标值"""
    
    models_dir = f"outputs/models"
    ensure_dir(models_dir)
    
    pattern = f"{models_dir}/{target}_{model_name}_best_*.bin"
    existing_files = glob.glob(pattern)
    
    best_wmae = float('inf')
    old_model_path = None
    
    # 从文件名中提取历史最佳wMAE
    for file_path in existing_files:
        try:
            # 提取文件名中的wMAE值
            filename = os.path.basename(file_path)
            # 格式: {target}_{model}_best_{wmae}.bin
            wmae_str = filename.split('_best_')[1].replace('.bin', '')
            file_wmae = float(wmae_str)
            
            if file_wmae < best_wmae:
                best_wmae = file_wmae
                old_model_path = file_path
                
        except (ValueError, IndexError) as e:
            logger.warning(f"Cannot parse wMAE from filename {file_path}: {e}")
            continue
    
    # 检查是否需要保存新模型
    if current_wmae < best_wmae:
        # 构建新的文件路径
        new_model_path = f"{models_dir}/{target}_{model_name}_best_{current_wmae:.6f}.bin"
        
        try:
            # 保存新模型
            if model_name == 'xgb':
                model.save_model(new_model_path)
            elif model_name == 'lgb':
                model.booster_.save_model(new_model_path)
            elif model_name == 'cat':
                model.save_model(new_model_path)
            
            # 删除旧的最佳模型（如果存在）
            if old_model_path and os.path.exists(old_model_path):
                os.remove(old_model_path)
                logger.info(f"Removed old best model: {old_model_path}")
            
            improvement = best_wmae - current_wmae
            logger.info(f"🎉 New best model saved! wMAE improved by {improvement:.6f}")
            logger.info(f"   Previous best: {best_wmae:.6f}")
            logger.info(f"   Current best:  {current_wmae:.6f}")
            logger.info(f"   Saved to: {new_model_path}")
            
            return True, new_model_path
            
        except Exception as e:
            logger.error(f"Failed to save improved model: {e}")
            return False, None
    else:
        gap = current_wmae - best_wmae
        logger.info(f"Model not saved - wMAE {current_wmae:.6f} vs best {best_wmae:.6f} (gap: +{gap:.6f})")
        if old_model_path:
            logger.info(f"Current best model: {old_model_path}")
        return False, old_model_path


def get_best_model_path(target: str, model_name: str):
    """获取当前最佳模型路径和分数"""
    models_dir = f"outputs/models"
    pattern = f"{models_dir}/{target}_{model_name}_best_*.bin"
    existing_files = glob.glob(pattern)
    
    if not existing_files:
        return None, float('inf')
    
    best_wmae = float('inf')
    best_path = None
    
    for file_path in existing_files:
        try:
            filename = os.path.basename(file_path)
            wmae_str = filename.split('_best_')[1].replace('.bin', '')
            file_wmae = float(wmae_str)
            
            if file_wmae < best_wmae:
                best_wmae = file_wmae
                best_path = file_path
                
        except (ValueError, IndexError):
            continue
    
    return best_path, best_wmae

def save_feature_importance(model, model_name: str, target: str, fold: int, logger):
    """保存特征重要性"""
    try:
        if model_name == 'xgb':
            imp = model.get_booster().get_score(importance_type="gain")
            df_imp = pd.DataFrame(list(imp.items()), columns=["feature", "importance"])
        elif model_name == 'lgb':
            imp = model.booster_.feature_importance(importance_type="gain")
            df_imp = pd.DataFrame({
                "feature": model.booster_.feature_name(),
                "importance": imp
            })
        elif model_name == 'cat':
            imp = model.get_feature_importance()
            df_imp = pd.DataFrame({
                "feature": model.feature_names_,
                "importance": imp
            })
        else:
            logger.warning(f"Feature importance not supported for {model_name}")
            return

        imp_path = f"outputs/importance/{model_name}_{target}_fold{fold}.csv"
        ensure_dir(os.path.dirname(imp_path))
        df_imp.to_csv(imp_path, index=False)
        logger.info(f"Feature importance saved: {imp_path}")
        
    except Exception as e:
        logger.warning(f"Failed to save feature importance: {e}")

def run_cv(model_name: str, target: str, config: dict, n_folds: int, seed: int, logger,
           train_path: str, test_path: str = None):
    """执行交叉验证训练"""
    logger.info(f"Starting CV: {model_name.upper()} for {target}")
    
    # 显示当前最佳记录
    current_best_path, current_best_wmae = get_best_model_path(target, model_name)
    if current_best_path:
        logger.info(f"Current best model: {os.path.basename(current_best_path)} (wMAE: {current_best_wmae:.6f})")
    else:
        logger.info("No previous best model found")
    
    # 加载数据
    X, y, X_test, test_ids, train_ids = load_data(target, logger, train_path, test_path)
    
    # 准备配置
    params = config.get('params', {})
    fit_params = config.get('fit', {})
    
    # 交叉验证
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
    
    fold_metrics = []
    oof_predictions = np.zeros(len(y))
    test_predictions = []
    
    # 跟踪最佳fold模型
    best_fold_score = float('inf')
    best_fold_model = None
    best_fold_idx = -1
    
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
            model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                verbose=False
            )
        elif model_name == 'lgb':
            import lightgbm as lgb
            model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                callbacks=[
                    lgb.early_stopping(fit_params.get('early_stopping_rounds', 100)),
                    lgb.log_evaluation(0)
                ]
            )
        elif model_name == 'cat':
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
        fold_wmae = compute_single_wmae(y_val, val_pred, target, ranges, counts)
        fold_metrics.append(fold_metric)
        
        # 检查是否是最佳fold
        if fold_wmae < best_fold_score:
            best_fold_score = fold_wmae
            best_fold_model = model
            best_fold_idx = fold
            logger.info(f"Fold {fold + 1}: New best wMAE={fold_wmae:.6f}")
        else:
            logger.info(f"Fold {fold + 1}: wMAE={fold_wmae:.6f}")
        
        # 测试集预测
        if X_test is not None:
            test_pred = model.predict(X_test)
            test_predictions.append(test_pred)
        
        # 保存特征重要性
        save_feature_importance(model, model_name, target, fold, logger)
    
    # 计算CV指标
    cv_metrics = regression_metrics(y, oof_predictions)
    cv_wmae = compute_single_wmae(y, oof_predictions, target, ranges, counts)
    
    # 尝试保存最佳模型
    saved, model_path = save_best_model_if_improved(best_fold_model, model_name, target, cv_wmae, logger)
    
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
        'fold_metrics': fold_metrics,
        'model_saved': saved,
        'model_path': model_path
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
    
    # 数据路径参数
    parser.add_argument('--train-path', required=True,
                       help='Path to training CSV file')
    parser.add_argument('--test-path', default=None,
                       help='Path to test CSV file (optional)')
    
    args = parser.parse_args()
    
    # 设置随机种子
    set_seed(args.seed)
    
    # 设置日志
    log_file = f"outputs/logs/{args.model}_{args.target}.log"
    logger = setup_logger(f"{args.model}_{args.target}", log_file)
    
    logger.info(f"Starting training: {args.model} for {args.target}")
    logger.info(f"Config: {args.config}")
    logger.info(f"Training data: {args.train_path}")
    logger.info(f"Test data: {args.test_path}")
    logger.info(f"Folds: {args.folds}, Seed: {args.seed}")
    
    try:
        # 加载配置
        config = load_json(args.config)
        logger.info("Config loaded successfully")
        
        # 执行训练
        results = run_cv(
            args.model, args.target, config, args.folds, args.seed, logger,
            args.train_path, args.test_path
        )
        
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