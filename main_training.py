"""
完整的训练、调优和预测流程
"""
import os
import sys
import pandas as pd
import numpy as np
import json
from pathlib import Path
from typing import List, Dict, Optional
import warnings
warnings.filterwarnings('ignore')

# 导入模块
from data import (get_train_test, add_extra_data, clean_smiles, filter_train_data,
                 replace_all_R_with_C, PolymerFeatureExtractor, prepare_ml_dataset)
from train import TraditionalMLPipeline
from model import MLModelConfig

def run_hyperparameter_optimization(targets: List[str], model_type: str = 'xgboost', 
                                   n_trials: int = 50) -> Dict:
    """
    使用Optuna进行超参数优化
    """
    try:
        import optuna
    except ImportError:
        print("Optuna not available. Using default parameters.")
        return {}
    
    print(f"Running hyperparameter optimization for {model_type}...")
    
    # 准备数据
    train, test, _ = get_train_test()
    train_extended = add_extra_data(train)
    train_extended['SMILES'] = train_extended['SMILES'].apply(replace_all_R_with_C)
    train_extended = clean_smiles(train_extended)
    train_filtered = filter_train_data(train_extended)
    
    feature_extractor = PolymerFeatureExtractor(use_mordred=False, use_rdkit=True)
    train_features, _ = prepare_ml_dataset(
        train_filtered, test, targets, feature_extractor,
        cache_train_path=f"train_features_{model_type}.pkl",
        force_rebuild=False
    )
    
    best_params = {}
    
    for target in targets:
        if target not in train_features.columns:
            continue
            
        print(f"Optimizing for {target}...")
        
        def objective(trial):
            if model_type == 'xgboost':
                params = {
                    'n_estimators': 1000,
                    'max_depth': trial.suggest_int('max_depth', 3, 10),
                    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                    'subsample': trial.suggest_float('subsample', 0.5, 1.0),
                    'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
                    'reg_lambda': trial.suggest_float('reg_lambda', 0.1, 10.0, log=True),
                    'gamma': trial.suggest_float('gamma', 0.0, 5.0),
                    'objective': 'reg:absoluteerror',
                    'eval_metric': 'mae',
                    'early_stopping_rounds': 50,
                    'random_state': 42
                }
            elif model_type == 'lightgbm':
                params = {
                    'n_estimators': 1000,
                    'max_depth': trial.suggest_int('max_depth', 3, 10),
                    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                    'subsample': trial.suggest_float('subsample', 0.5, 1.0),
                    'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
                    'reg_lambda': trial.suggest_float('reg_lambda', 0.1, 10.0, log=True),
                    'objective': 'mae',
                    'metric': 'mae',
                    'verbosity': -1,
                    'random_state': 42,
                    'force_col_wise': True
                }
            elif model_type == 'catboost':
                params = {
                    'iterations': 1000,
                    'depth': trial.suggest_int('depth', 3, 10),
                    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                    'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 0.1, 10.0, log=True),
                    'loss_function': 'MAE',
                    'eval_metric': 'MAE',
                    'random_seed': 42,
                    'verbose': False,
                    'early_stopping_rounds': 50
                }
            
            # 创建临时配置
            temp_config = MLModelConfig()
            if model_type == 'xgboost':
                temp_config.xgb_params[target] = params
            elif model_type == 'lightgbm':
                temp_config.lgb_params[target] = params
            elif model_type == 'catboost':
                temp_config.cat_params[target] = params
            
            # 训练模型
            pipeline = TraditionalMLPipeline(model_type=model_type, config=temp_config)
            
            # 获取有效数据
            valid_mask = train_features[target].notna()
            features_subset = train_features[valid_mask]
            
            X = pipeline.prepare_features(features_subset, targets, fit=True)
            y = features_subset[target]
            
            result = pipeline.train_target(X, y, target)
            return result['scores']['mae']  # 返回MAE作为优化目标
        
        # 运行优化
        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=n_trials)
        
        best_params[target] = study.best_params
        print(f"Best MAE for {target}: {study.best_value:.4f}")
        print(f"Best params: {study.best_params}")
    
    # 保存最优参数
    with open(f'best_params_{model_type}.json', 'w') as f:
        json.dump(best_params, f, indent=2)
    
    return best_params

def train_with_best_params(targets: List[str], model_types: List[str], 
                          use_optimization: bool = True) -> Dict[str, pd.DataFrame]:
    """
    使用最优参数训练模型
    """
    # 数据准备
    print("Preparing data...")
    train, test, _ = get_train_test()
    train_extended = add_extra_data(train)
    
    # 清理SMILES
    train_extended['SMILES'] = train_extended['SMILES'].apply(replace_all_R_with_C)
    test['SMILES'] = test['SMILES'].apply(replace_all_R_with_C)
    
    train_extended = clean_smiles(train_extended)
    test = clean_smiles(test)
    train_filtered = filter_train_data(train_extended)
    
    print(f"Training data: {train_filtered.shape}")
    print(f"Test data: {test.shape}")
    
    # 提取特征
    print("Extracting features...")
    feature_extractor = PolymerFeatureExtractor(use_mordred=True, use_rdkit=True)
    train_features, test_features = prepare_ml_dataset(
        train_filtered, test, targets, feature_extractor,
        force_rebuild=False
    )
    
    predictions = {}
    
    for model_type in model_types:
        print(f"\n{'='*60}")
        print(f"Training {model_type.upper()} models")
        print(f"{'='*60}")
        
        # 配置模型
        config = MLModelConfig()
        
        # 加载最优参数（如果存在）
        if use_optimization:
            params_file = f'best_params_{model_type}.json'
            if os.path.exists(params_file):
                print(f"Loading optimized parameters from {params_file}")
                with open(params_file, 'r') as f:
                    best_params = json.load(f)
                
                # 更新配置
                for target, params in best_params.items():
                    if model_type == 'xgboost':
                        config.xgb_params[target].update(params)
                    elif model_type == 'lightgbm':
                        config.lgb_params[target].update(params)
                    elif model_type == 'catboost':
                        config.cat_params[target].update(params)
            else:
                print(f"No optimized parameters found for {model_type}, using defaults")
        
        # 初始化管道
        pipeline = TraditionalMLPipeline(model_type=model_type, config=config)
        
        # 检查是否已有训练好的模型
        if os.path.exists(f"models/{model_type}") and not use_optimization:
            try:
                pipeline.load_models()
                if len(pipeline.models) > 0:
                    print(f"Loaded existing {model_type} models")
                    predictions[model_type] = pipeline.predict(test_features, targets)
                    continue
            except Exception as e:
                print(f"Failed to load {model_type} models: {e}")
        
        # 训练新模型
        pipeline.train_all_targets(train_features, targets)
        
        # 预测
        predictions[model_type] = pipeline.predict(test_features, targets)
        
        # 保存单独的预测结果
        predictions[model_type].to_csv(f"{model_type}_predictions.csv", index=False)
        
        # 绘制结果
        pipeline.plot_results()
    
    return predictions

def create_ensemble_predictions(predictions: Dict[str, pd.DataFrame], 
                               targets: List[str], 
                               weights: Optional[Dict[str, float]] = None) -> pd.DataFrame:
    """
    创建集成预测
    """
    if not predictions:
        raise ValueError("No predictions provided")
    
    # 获取测试ID
    test_ids = predictions[list(predictions.keys())[0]]['id']
    ensemble = pd.DataFrame({'id': test_ids})
    
    if weights is None:
        # 等权重集成
        weights = {model: 1.0 / len(predictions) for model in predictions.keys()}
    
    print(f"Creating ensemble with weights: {weights}")
    
    for target in targets:
        target_preds = []
        target_weights = []
        
        for model_name, pred_df in predictions.items():
            if target in pred_df.columns and not pred_df[target].isna().all():
                target_preds.append(pred_df[target] * weights.get(model_name, 0))
                target_weights.append(weights.get(model_name, 0))
        
        if target_preds:
            # 加权平均
            ensemble[target] = sum(target_preds) / sum(target_weights)
        else:
            print(f"Warning: No valid predictions for target {target}")
            ensemble[target] = np.nan
    
    return ensemble

def evaluate_predictions(predictions: Dict[str, pd.DataFrame], 
                        train_features: pd.DataFrame, 
                        targets: List[str]) -> pd.DataFrame:
    """
    评估预测结果（基于训练数据的统计分析）
    """
    results = []
    
    for model_name, pred_df in predictions.items():
        for target in targets:
            if target in pred_df.columns and target in train_features.columns:
                # 训练数据统计
                train_mean = train_features[target].mean()
                train_std = train_features[target].std()
                train_min = train_features[target].min()
                train_max = train_features[target].max()
                
                # 预测数据统计
                pred_mean = pred_df[target].mean()
                pred_std = pred_df[target].std()
                pred_min = pred_df[target].min()
                pred_max = pred_df[target].max()
                
                # 超出范围的预测数量
                out_of_range = ((pred_df[target] < train_min) | 
                               (pred_df[target] > train_max)).sum()
                
                results.append({
                    'model': model_name,
                    'target': target,
                    'train_mean': train_mean,
                    'pred_mean': pred_mean,
                    'train_std': train_std,
                    'pred_std': pred_std,
                    'train_range': train_max - train_min,
                    'pred_range': pred_max - pred_min,
                    'out_of_range_count': out_of_range,
                    'out_of_range_pct': out_of_range / len(pred_df) * 100
                })
    
    return pd.DataFrame(results)

def main():
    """主训练流程"""
    targets = ['Tg', 'FFV', 'Tc', 'Density', 'Rg']
    model_types = ['xgboost', 'lightgbm', 'catboost']
    
    # 解析命令行参数
    import argparse
    parser = argparse.ArgumentParser(description='Train polymer property prediction models')
    parser.add_argument('--optimize', action='store_true', 
                       help='Run hyperparameter optimization')
    parser.add_argument('--n_trials', type=int, default=50,
                       help='Number of optimization trials')
    parser.add_argument('--models', nargs='+', choices=model_types, 
                       default=model_types, help='Models to train')
    parser.add_argument('--targets', nargs='+', choices=targets,
                       default=targets, help='Targets to predict')
    
    args = parser.parse_args()
    
    # 创建必要的目录
    os.makedirs('models', exist_ok=True)
    os.makedirs('results', exist_ok=True)
    os.makedirs('predictions', exist_ok=True)
    
    # 超参数优化（可选）
    if args.optimize:
        print("Running hyperparameter optimization...")
        for model_type in args.models:
            run_hyperparameter_optimization(args.targets, model_type, args.n_trials)
        print("Optimization completed!")
    
    # 训练模型
    print("Training models with best parameters...")
    predictions = train_with_best_params(args.targets, args.models, args.optimize)
    
    # 准备训练数据用于评估
    train, test, _ = get_train_test()
    train_extended = add_extra_data(train)
    train_extended['SMILES'] = train_extended['SMILES'].apply(replace_all_R_with_C)
    train_extended = clean_smiles(train_extended)
    train_filtered = filter_train_data(train_extended)
    
    feature_extractor = PolymerFeatureExtractor(use_mordred=True, use_rdkit=True)
    train_features, _ = prepare_ml_dataset(
        train_filtered, test, args.targets, feature_extractor
    )
    
    # 评估预测结果
    evaluation = evaluate_predictions(predictions, train_features, args.targets)
    evaluation.to_csv('results/prediction_evaluation.csv', index=False)
    print("\nPrediction evaluation:")
    print(evaluation.round(3))
    
    # 创建集成预测
    if len(predictions) > 1:
        print("\nCreating ensemble predictions...")
        
        # 简单等权重集成
        ensemble_equal = create_ensemble_predictions(predictions, args.targets)
        ensemble_equal.to_csv('predictions/ensemble_equal_weight.csv', index=False)
        
        # 基于模型表现的加权集成（这里使用简单的启发式权重）
        # 在实际应用中，你可以基于交叉验证结果来计算权重
        model_weights = {
            'xgboost': 0.4,
            'lightgbm': 0.35,
            'catboost': 0.25
        }
        
        # 只使用实际训练的模型
        actual_weights = {k: v for k, v in model_weights.items() if k in predictions}
        weight_sum = sum(actual_weights.values())
        actual_weights = {k: v/weight_sum for k, v in actual_weights.items()}
        
        ensemble_weighted = create_ensemble_predictions(predictions, args.targets, actual_weights)
        ensemble_weighted.to_csv('predictions/ensemble_weighted.csv', index=False)
        
        print("Ensemble predictions saved!")
    
    print("\n" + "="*60)
    print("TRAINING COMPLETED!")
    print("="*60)
    print(f"Models trained: {list(predictions.keys())}")
    print(f"Targets predicted: {args.targets}")
    print("\nFiles generated:")
    for model in predictions.keys():
        print(f"- {model}_predictions.csv")
    if len(predictions) > 1:
        print("- predictions/ensemble_equal_weight.csv")
        print("- predictions/ensemble_weighted.csv")
    print("- results/prediction_evaluation.csv")
    print("- Cross-validation results in results/ directory")

if __name__ == "__main__":
    main()