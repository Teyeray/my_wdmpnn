#!/usr/bin/env python3
"""
主训练流水线示例
演示如何使用模块化组件进行完整的ML工作流
"""
import os
import argparse
import subprocess
from pathlib import Path

def run_command(cmd, description):
    """运行命令并处理错误"""
    print(f"\n=== {description} ===")
    print(f"Running: {' '.join(cmd)}")
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"❌ Failed: {description}")
        print(f"Error: {result.stderr}")
        return False
    else:
        print(f"✅ Success: {description}")
        print(result.stdout)
        return True

def main():
    parser = argparse.ArgumentParser(description='ML Pipeline Example')
    parser.add_argument('--target', required=True, 
                       choices=['Tg', 'Tc', 'Rg', 'FFV', 'Density'],
                       help='Target variable')
    parser.add_argument('--skip-tune', action='store_true',
                       help='Skip hyperparameter tuning')
    
    args = parser.parse_args()
    target = args.target
    
    print(f"🚀 Starting ML Pipeline for {target}")
    
    # 1. 单模型训练（使用基础配置）
    models = ['xgb', 'lgb', 'cat']
    
    for model in models:
        if not args.skip_tune:
            # 超参数优化
            tune_cmd = [
                'python', '-m', 'tools.tune',
                '--model', model,
                '--target', target,
                '--base-config', f'configs/{model}_base.json',
                '--n-trials', '50',
                '--folds', '5'
            ]
            
            if not run_command(tune_cmd, f"Hyperparameter tuning: {model}"):
                print(f"⚠️  Tuning failed for {model}, using base config")
                config_path = f'configs/{model}_base.json'
            else:
                config_path = f'configs/best_{model}_{target}.json'
        else:
            config_path = f'configs/{model}_base.json'
        
        # 使用最佳/基础配置训练
        train_cmd = [
            'python', '-m', 'utils.train_model',
            '--model', model,
            '--target', target,
            '--config', config_path,
            '--folds', '5'
        ]
        
        if not run_command(train_cmd, f"Training: {model}"):
            print(f"❌ Training failed for {model}")
            continue
    
    # 2. 集成方法
    ensemble_methods = [
        ('weighted', 'configs/ensemble_weighted.json'),
        ('stacking', 'configs/ensemble_stacking.json')
    ]
    
    for method, config in ensemble_methods:
        ensemble_cmd = [
            'python', '-m', 'tools.ensemble',
            '--target', target,
            '--models'] + models + [
            '--method', method,
            '--config', config
        ]
        
        run_command(ensemble_cmd, f"Ensemble: {method}")
    
    print(f"\n🎉 Pipeline completed for {target}!")
    print("📊 Check outputs/ directory for results:")
    print("  - outputs/oof/     : Out-of-fold predictions")
    print("  - outputs/preds/   : Test set predictions") 
    print("  - outputs/models/  : Trained models")
    print("  - outputs/logs/    : Training logs")
    print("  - outputs/meta/    : Optimization results")

if __name__ == "__main__":
    main()
