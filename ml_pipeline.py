"""
Traditional ML training pipeline for XGBoost, LightGBM, and CatBoost
"""
import os
import pandas as pd
from pathlib import Path
from typing import List, Dict

# Import our modules
from data import (get_train_test, add_extra_data, clean_smiles, filter_train_data, 
                 replace_all_R_with_C, PolymerFeatureExtractor, prepare_ml_dataset)
from train import TraditionalMLPipeline
from model import MLModelConfig

def main():
    """Main pipeline for traditional ML models"""
    
    # Configuration
    targets = ['Tg', 'FFV', 'Tc', 'Density', 'Rg']
    models_to_train = ['xgboost', 'lightgbm', 'catboost']
    
    print("Loading and preprocessing data...")
    
    # Load and preprocess data (reuse existing functions)
    train, test, _ = get_train_test()
    train_extended = add_extra_data(train)
    
    # Clean SMILES
    train_extended['SMILES_raw'] = train_extended['SMILES']
    train_extended['SMILES'] = train_extended['SMILES'].apply(replace_all_R_with_C)
    test['SMILES'] = test['SMILES'].apply(replace_all_R_with_C)
    
    train_extended = clean_smiles(train_extended)
    test = clean_smiles(test)
    train_filtered = filter_train_data(train_extended)
    
    print(f"Final training data shape: {train_filtered.shape}")
    print(f"Test data shape: {test.shape}")
    
    # Extract molecular features
    print("Extracting molecular features...")
    feature_extractor = PolymerFeatureExtractor(use_mordred=True, use_rdkit=True)
    
    train_features, test_features = prepare_ml_dataset(
        train_filtered, test, targets, feature_extractor,
        force_rebuild=False  # Set to True to rebuild cache
    )
    
    # Train models
    config = MLModelConfig()
    all_predictions = []
    
    for model_type in models_to_train:
        print(f"\n{'='*50}")
        print(f"Training {model_type.upper()} models")
        print(f"{'='*50}")
        
        # Initialize pipeline
        pipeline = TraditionalMLPipeline(model_type=model_type, config=config)
        
        # Check if models exist
        if os.path.exists(f"models/{model_type}"):
            try:
                pipeline.load_models()
                if len(pipeline.models) > 0:
                    print(f"Loaded existing {model_type} models")
                    # Make predictions
                    predictions = pipeline.predict(test_features, targets)
                    predictions.to_csv(f"{model_type}_predictions.csv", index=False)
                    all_predictions.append(predictions)
                    continue
            except:
                print(f"Failed to load {model_type} models, training new ones...")
        
        # Train new models
        pipeline.train_all_targets(train_features, targets)
        
        # Make predictions
        predictions = pipeline.predict(test_features, targets)
        predictions.to_csv(f"{model_type}_predictions.csv", index=False)
        all_predictions.append(predictions)
        
        # Plot results
        pipeline.plot_results()
    
    # Create ensemble predictions
    if len(all_predictions) > 1:
        print("\nCreating ensemble predictions...")
        ensemble_pred = all_predictions[0][['id']].copy()
        
        for target in targets:
            target_preds = []
            for pred_df in all_predictions:
                if target in pred_df.columns:
                    target_preds.append(pred_df[target])
            
            if target_preds:
                ensemble_pred[target] = pd.concat(target_preds, axis=1).mean(axis=1)
        
        ensemble_pred.to_csv("ensemble_ml_predictions.csv", index=False)
        print("Ensemble predictions saved!")
    
    # Compare model performances
    print("\n" + "="*50)
    print("MODEL COMPARISON")
    print("="*50)
    
    for model_type in models_to_train:
        results_path = f"results/{model_type}_cv_results.csv"
        if os.path.exists(results_path):
            results = pd.read_csv(results_path, index_col=0)
            print(f"\n{model_type.upper()} Results:")
            print(results[['mae_mean', 'mae_std']].round(4))


if __name__ == "__main__":
    main()