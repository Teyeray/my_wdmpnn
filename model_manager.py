"""
模型管理工具 - 查看、比较和管理训练的模型
"""
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json

def list_available_models():
    """列出所有可用的模型"""
    models_dir = Path("models")
    if not models_dir.exists():
        print("No models directory found!")
        return
    
    available_models = {}
    
    for model_type in ['xgboost', 'lightgbm', 'catboost']:
        model_path = models_dir / model_type
        if model_path.exists():
            targets = []
            for target_dir in model_path.iterdir():
                if target_dir.is_dir() and not target_dir.name.endswith('.pkl'):
                    fold_models = list(target_dir.glob('model_fold_*.pkl'))
                    if fold_models:
                        targets.append({
                            'target': target_dir.name,
                            'n_folds': len(fold_models),
                            'has_feature_selection': (target_dir / 'selected_features.pkl').exists()
                        })
            
            if targets:
                available_models[model_type] = targets
    
    return available_models

def compare_model_performance():
    """比较不同模型的性能"""
    results_dir = Path("results")
    if not results_dir.exists():
        print("No results directory found!")
        return
    
    all_results = {}
    
    for model_type in ['xgboost', 'lightgbm', 'catboost']:
        results_file = results_dir / f"{model_type}_cv_results.csv"
        if results_file.exists():
            df = pd.read_csv(results_file, index_col=0)
            all_results[model_type] = df
    
    if not all_results:
        print("No CV results found!")
        return
    
    # 创建比较表
    comparison_data = []
    for model_type, results in all_results.items():
        for target in results.index:
            comparison_data.append({
                'Model': model_type,
                'Target': target,
                'MAE_mean': results.loc[target, 'mae_mean'],
                'MAE_std': results.loc[target, 'mae_std'],
                'RMSE_mean': results.loc[target, 'rmse_mean'],
                'R2_mean': results.loc[target, 'r2_mean']
            })
    
    comparison_df = pd.DataFrame(comparison_data)
    
    # 保存比较结果
    comparison_df.to_csv(results_dir / "model_comparison.csv", index=False)
    
    # 可视化比较
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # MAE比较
    mae_pivot = comparison_df.pivot(index='Target', columns='Model', values='MAE_mean')
    mae_pivot.plot(kind='bar', ax=axes[0,0], title='Mean Absolute Error by Model and Target')
    axes[0,0].set_ylabel('MAE')
    axes[0,0].tick_params(axis='x', rotation=45)
    
    # RMSE比较
    rmse_pivot = comparison_df.pivot(index='Target', columns='Model', values='RMSE_mean')
    rmse_pivot.plot(kind='bar', ax=axes[0,1], title='Root Mean Square Error by Model and Target')
    axes[0,1].set_ylabel('RMSE')
    axes[0,1].tick_params(axis='x', rotation=45)
    
    # R²比较
    r2_pivot = comparison_df.pivot(index='Target', columns='Model', values='R2_mean')
    r2_pivot.plot(kind='bar', ax=axes[1,0], title='R² Score by Model and Target')
    axes[1,0].set_ylabel('R²')
    axes[1,0].tick_params(axis='x', rotation=45)
    
    # 整体性能热图
    performance_matrix = comparison_df.groupby(['Model', 'Target'])['MAE_mean'].unstack()
    sns.heatmap(performance_matrix, annot=True, fmt='.3f', cmap='YlOrRd_r', 
                ax=axes[1,1], cbar_kws={'label': 'MAE'})
    axes[1,1].set_title('MAE Heatmap')
    
    plt.tight_layout()
    plt.savefig(results_dir / "model_comparison.png", dpi=300, bbox_inches='tight')
    plt.show()
    
    return comparison_df

def analyze_prediction_consistency():
    """分析不同模型预测的一致性"""
    prediction_files = []
    model_predictions = {}
    
    # 找到所有预测文件
    for model_type in ['xgboost', 'lightgbm', 'catboost']:
        pred_file = f"{model_type}_predictions.csv"
        if os.path.exists(pred_file):
            df = pd.read_csv(pred_file)
            model_predictions[model_type] = df
    
    if len(model_predictions) < 2:
        print("Need at least 2 model predictions to analyze consistency")
        return
    
    targets = ['Tg', 'FFV', 'Tc', 'Density', 'Rg']
    
    # 计算预测之间的相关性
    correlations = {}
    
    for target in targets:
        target_data = {}
        for model, pred_df in model_predictions.items():
            if target in pred_df.columns:
                target_data[model] = pred_df[target]
        
        if len(target_data) >= 2:
            target_df = pd.DataFrame(target_data)
            corr_matrix = target_df.corr()
            correlations[target] = corr_matrix
    
    # 可视化相关性
    if correlations:
        n_targets = len(correlations)
        fig, axes = plt.subplots(1, n_targets, figsize=(5*n_targets, 4))
        if n_targets == 1:
            axes = [axes]
        
        for i, (target, corr_matrix) in enumerate(correlations.items()):
            sns.heatmap(corr_matrix, annot=True, fmt='.3f', cmap='coolwarm',
                       center=0, square=True, ax=axes[i],
                       cbar_kws={'label': 'Correlation'})
            axes[i].set_title(f'{target} Prediction Correlations')
        
        plt.tight_layout()
        plt.savefig("results/prediction_correlations.png", dpi=300, bbox_inches='tight')
        plt.show()
    
    return correlations

def model_summary():
    """模型总览"""
    print("="*60)
    print("MODEL SUMMARY")
    print("="*60)
    
    # 列出可用模型
    available = list_available_models()
    if available:
        print("\nAvailable Models:")
        for model_type, targets in available.items():
            print(f"\n{model_type.upper()}:")
            for target_info in targets:
                feature_sel = "✓" if target_info['has_feature_selection'] else "✗"
                print(f"  - {target_info['target']}: {target_info['n_folds']} folds, "
                      f"feature selection: {feature_sel}")
    else:
        print("No trained models found!")
        return
    
    # 性能比较
    print(f"\n{'='*60}")
    print("PERFORMANCE COMPARISON")
    print("="*60)
    
    comparison = compare_model_performance()
    if comparison is not None:
        print("\nBest performing model for each target (by MAE):")
        best_models = comparison.loc[comparison.groupby('Target')['MAE_mean'].idxmin()]
        for _, row in best_models.iterrows():
            print(f"  - {row['Target']}: {row['Model']} (MAE: {row['MAE_mean']:.4f})")
    
    # 预测一致性
    print(f"\n{'='*60}")
    print("PREDICTION CONSISTENCY")
    print("="*60)
    
    correlations = analyze_prediction_consistency()
    if correlations:
        print("\nAverage correlation between models:")
        for target, corr_matrix in correlations.items():
            # 计算非对角线元素的平均值
            mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
            avg_corr = corr_matrix.values[mask].mean()
            print(f"  - {target}: {avg_corr:.3f}")

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Model management and analysis')
    parser.add_argument('--action', choices=['list', 'compare', 'consistency', 'summary'],
                       default='summary', help='Action to perform')
    
    args = parser.parse_args()
    
    if args.action == 'list':
        models = list_available_models()
        print(json.dumps(models, indent=2))
    elif args.action == 'compare':
        compare_model_performance()
    elif args.action == 'consistency':
        analyze_prediction_consistency()
    elif args.action == 'summary':
        model_summary()

if __name__ == "__main__":
    main()