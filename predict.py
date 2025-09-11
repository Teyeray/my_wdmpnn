"""
快速预测脚本 - 使用已训练的模型进行预测
"""
import os
import pandas as pd
import numpy as np
from typing import List, Optional

from data import (get_train_test, add_extra_data, clean_smiles, 
                 replace_all_R_with_C, PolymerFeatureExtractor, prepare_ml_dataset)
from train import TraditionalMLPipeline
from model import MLModelConfig

def quick_predict(model_types: List[str] = None, 
                 targets: List[str] = None,
                 output_file: str = "quick_predictions.csv") -> pd.DataFrame:
    """
    使用已训练的模型快速进行预测
    """
    if model_types is None:
        model_types = ['xgboost', 'lightgbm', 'catboost']
    
    if targets is None:
        targets = ['Tg', 'FFV', 'Tc', 'Density', 'Rg']
    
    print("Loading test data...")
    train, test, _ = get_train_test()
    
    # 数据清理
    test['SMILES'] = test['SMILES'].apply(replace_all_R_with_C)
    test = clean_smiles(test)
    
    # 特征提取
    print("Extracting features...")
    feature_extractor = PolymerFeatureExtractor(use_mordred=True, use_rdkit=True)
    
    # 使用缓存的特征（如果存在）
    try:
        test_features = pd.read_pickle("test_ml_features.pkl")
        print("Loaded cached test features")
    except:
        print("Extracting test features...")
        train_extended = add_extra_data(train)
        train_extended['SMILES'] = train_extended['SMILES'].apply(replace_all_R_with_C)
        train_extended = clean_smiles(train_extended)
        
        _, test_features = prepare_ml_dataset(
            train_extended, test, targets, feature_extractor
        )
    
    predictions = {}
    
    # 加载每个模型的预测
    for model_type in model_types:
        print(f"Loading {model_type} model...")
        
        try:
            config = MLModelConfig()
            pipeline = TraditionalMLPipeline(model_type=model_type, config=config)
            pipeline.load_models()
            
            if len(pipeline.models) > 0:
                pred = pipeline.predict(test_features, targets)
                predictions[model_type] = pred
                print(f"✓ {model_type} predictions loaded")
            else:
                print(f"✗ No {model_type} models found")
        
        except Exception as e:
            print(f"✗ Failed to load {model_type}: {e}")
    
    if not predictions:
        print("No models found! Please train models first.")
        return pd.DataFrame()
    
    # 创建集成预测
    print("Creating ensemble predictions...")
    test_ids = test_features['id']
    ensemble = pd.DataFrame({'id': test_ids})
    
    for target in targets:
        target_preds = []
        valid_models = []
        
        for model_name, pred_df in predictions.items():
            if target in pred_df.columns and not pred_df[target].isna().all():
                target_preds.append(pred_df[target])
                valid_models.append(model_name)
        
        if target_preds:
            ensemble[target] = pd.concat(target_preds, axis=1).mean(axis=1)
            print(f"✓ {target}: ensemble from {valid_models}")
        else:
            print(f"✗ {target}: no valid predictions")
            ensemble[target] = np.nan
    
    # 保存结果
    ensemble.to_csv(output_file, index=False)
    print(f"Predictions saved to {output_file}")
    
    return ensemble

def predict_from_smiles(smiles_list: List[str], 
                       model_types: List[str] = None,
                       targets: List[str] = None) -> pd.DataFrame:
    """
    从SMILES列表直接预测
    """
    if model_types is None:
        model_types = ['xgboost', 'lightgbm', 'catboost']
    
    if targets is None:
        targets = ['Tg', 'FFV', 'Tc', 'Density', 'Rg']
    
    print(f"Predicting properties for {len(smiles_list)} molecules...")
    
    # 创建临时DataFrame
    temp_df = pd.DataFrame({
        'id': range(len(smiles_list)),
        'SMILES': smiles_list
    })
    
    # 清理SMILES
    temp_df['SMILES'] = temp_df['SMILES'].apply(replace_all_R_with_C)
    
    # 提取特征
    print("Extracting molecular features...")
    feature_extractor = PolymerFeatureExtractor(use_mordred=True, use_rdkit=True)
    mol_features = feature_extractor.smiles_to_features(temp_df['SMILES'].tolist())
    
    test_features = pd.concat([
        temp_df[['id', 'SMILES']].reset_index(drop=True),
        mol_features.reset_index(drop=True)
    ], axis=1)
    
    predictions = {}
    
    # 加载模型并预测
    for model_type in model_types:
        print(f"Loading {model_type} model...")
        
        try:
            config = MLModelConfig()
            pipeline = TraditionalMLPipeline(model_type=model_type, config=config)
            pipeline.load_models()
            
            if len(pipeline.models) > 0:
                pred = pipeline.predict(test_features, targets)
                predictions[model_type] = pred
                print(f"✓ {model_type} predictions completed")
            else:
                print(f"✗ No {model_type} models found")
        
        except Exception as e:
            print(f"✗ Failed to load {model_type}: {e}")
    
    if not predictions:
        print("No models found! Please train models first.")
        return pd.DataFrame()
    
    # 创建集成预测
    ensemble = pd.DataFrame({'SMILES': smiles_list})
    
    for target in targets:
        target_preds = []
        for model_name, pred_df in predictions.items():
            if target in pred_df.columns and not pred_df[target].isna().all():
                target_preds.append(pred_df[target])
        
        if target_preds:
            ensemble[target] = pd.concat(target_preds, axis=1).mean(axis=1)
    
    return ensemble

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Quick prediction using trained models')
    parser.add_argument('--models', nargs='+', 
                       choices=['xgboost', 'lightgbm', 'catboost'],
                       default=['xgboost', 'lightgbm', 'catboost'],
                       help='Models to use for prediction')
    parser.add_argument('--targets', nargs='+',
                       choices=['Tg', 'FFV', 'Tc', 'Density', 'Rg'],
                       default=['Tg', 'FFV', 'Tc', 'Density', 'Rg'],
                       help='Targets to predict')
    parser.add_argument('--output', default='quick_predictions.csv',
                       help='Output file name')
    parser.add_argument('--smiles', nargs='+', 
                       help='SMILES strings to predict (alternative to test set)')
    
    args = parser.parse_args()
    
    if args.smiles:
        # 从命令行SMILES预测
        results = predict_from_smiles(args.smiles, args.models, args.targets)
        print("\nPrediction Results:")
        print(results.round(3))
        results.to_csv(args.output, index=False)
    else:
        # 从测试集预测
        results = quick_predict(args.models, args.targets, args.output)
        print(f"\nPredicted {len(results)} molecules")
        print("Sample predictions:")
        print(results.head().round(3))

if __name__ == "__main__":
    main()