# my_wdmpnn
# 🧪 Polymer Property Prediction ML Framework

A modular machine learning framework for the Kaggle Polymer Property Prediction competition, supporting XGBoost, LightGBM, and CatBoost with automated hyperparameter tuning and ensemble methods.

## 📁 Project Structure

```
my_wdmpnn/
├── tools/                          # 核心模块
│   ├── __init__.py
│   ├── utils.py                    # 基础工具：JSON、日志、随机种子
│   ├── data.py                     # 数据处理和特征工程
│   ├── evaluate.py                 # 评估指标，包含竞赛wMAE
│   ├── train_model.py              # 单模型交叉验证训练
│   ├── ensemble.py                 # 模型集成（加权平均、Stacking）
│   └── tune.py                     # Optuna超参数优化
├── configs/                        # 配置文件
│   ├── xgb_base.json              # XGBoost基础配置
│   ├── lgb_base.json              # LightGBM基础配置  
│   ├── cat_base.json              # CatBoost基础配置
│   ├── ensemble_weighted.json      # 加权集成配置
│   ├── ensemble_stacking.json      # Stacking集成配置
│   └── search_space.json          # 超参数搜索空间
├── outputs/                        # 输出目录
│   ├── oof/                       # Out-of-fold预测
│   ├── preds/                     # 测试集预测
│   ├── models/                    # 保存的模型
│   ├── logs/                      # 训练日志
│   ├── meta/                      # 元数据（调优结果等）
│   └── submissions/               # Kaggle提交文件
├── datasets/                       # 数据集
├── main_pipeline.py               # 主流水线示例
└── README.md                      # 本文件
```

## 🚀 Quick Start

### 1. 单模型训练

```bash
# 使用基础配置训练XGBoost
python -m tools.train_model --model xgb --target Tg --config configs/xgb_base.json --train-path kaggle/input/neurips-open-polymer-prediction-2025/train.csv --test-path kaggle/input/neurips-open-polymer-prediction-2025/test.csv

# 训练所有模型
python -m tools.train_model --model xgb --target Tg --config configs/xgb_base.json --train-path kaggle/input/neurips-open-polymer-prediction-2025/train.csv --test-path kaggle/input/neurips-open-polymer-prediction-2025/test.csv
python -m tools.train_model --model lgb --target Tg --config configs/lgb_base.json --train-path kaggle/input/neurips-open-polymer-prediction-2025/train.csv --test-path kaggle/input/neurips-open-polymer-prediction-2025/test.csv
python -m tools.train_model --model cat --target Tg --config configs/cat_base.json --train-path kaggle/input/neurips-open-polymer-prediction-2025/train.csv --test-path kaggle/input/neurips-open-polymer-prediction-2025/test.csv

# 仅使用训练数据（无测试集预测）
python -m tools.train_model --model xgb --target Tg --config configs/xgb_base.json --train-path kaggle/input/neurips-open-polymer-prediction-2025/train.csv
```

### 2. 超参数优化

```bash
# 优化XGBoost超参数
python -m tools.tune --model xgb --target Tg --base-config configs/xgb_base.json --train-path kaggle/input/neurips-open-polymer-prediction-2025/train.csv --n-trials 100

# 使用自定义搜索空间
python -m tools.tune --model xgb --target Tg --base-config configs/xgb_base.json --search-space configs/search_space.json --train-path kaggle/input/neurips-open-polymer-prediction-2025/train.csv --n-trials 50

# 同时包含测试数据
python -m tools.tune --model xgb --target Tg --base-config configs/xgb_base.json --train-path kaggle/input/neurips-open-polymer-prediction-2025/train.csv --test-path kaggle/input/neurips-open-polymer-prediction-2025/test.csv --n-trials 100

# 使用优化后的配置训练
python -m tools.train_model --model xgb --target Tg --config configs/best_xgb_Tg.json --train-path kaggle/input/neurips-open-polymer-prediction-2025/train.csv --test-path kaggle/input/neurips-open-polymer-prediction-2025/test.csv
```

### 3. 模型集成

```bash
# 加权平均集成
python -m tools.ensemble --target Tg --models xgb lgb cat --method weighted --config configs/ensemble_weighted.json --train-path kaggle/input/neurips-open-polymer-prediction-2025/train.csv

# Stacking集成
python -m tools.ensemble --target Tg --models xgb lgb cat --method stacking --config configs/ensemble_stacking.json --train-path kaggle/input/neurips-open-polymer-prediction-2025/train.csv

# 自定义输出名称
python -m tools.ensemble --target Tg --models xgb lgb cat --method weighted --output-name final_ensemble --train-path kaggle/input/neurips-open-polymer-prediction-2025/train.csv

# 创建多目标提交文件（从已有的预测结果合并）
python -m tools.ensemble --target Tg --models xgb lgb cat --method stacking --train-path kaggle/input/neurips-open-polymer-prediction-2025/train.csv --create-multi-submission
```

### 4. 完整流水线

```bash
# 运行完整流水线（包含调优）
python main_pipeline.py --target Tg

# 跳过调优，仅使用基础配置
python main_pipeline.py --target Tg --skip-tune
```

## 📊 输出文件格式

### OOF预测 (`outputs/oof/`)
```csv
id,oof_pred
0,123.45
1,234.56
...
```

### 测试集预测 (`outputs/preds/`)
```csv
id,pred
0,123.45
1,234.56
...
```

### Kaggle提交文件 (`outputs/submissions/`)

**单目标提交文件：**
```csv
id,Tg
0,123.45
1,234.56
...
```

**多目标提交文件：**
```csv
id,Tg,Tc,Rg,FFV,Density
0,123.45,234.56,345.67,456.78,567.89
1,124.45,235.56,346.67,457.78,568.89
...
```

### 调优结果 (`outputs/meta/`)
```json
{
  "best_params": {...},
  "best_wmae": 0.123456,
  "n_trials": 100,
  "model_name": "xgb",
  "target": "Tg"
}
```

## 🎯 评估指标

框架使用竞赛指定的加权平均绝对误差（wMAE）：

```
w_i = (1/r_i) * (K * sqrt(1/n_i) / Z)
wMAE = mean_over_samples(sum_i w_i * |y_hat_i - y_i|)
```

其中：
- `r_i`: 第i个目标的值域范围
- `n_i`: 第i个目标的样本数量  
- `K`: 目标总数
- `Z`: 归一化因子

## ⚙️ 配置说明

### 模型配置结构
```json
{
  "params": {
    "n_estimators": 1000,
    "max_depth": 6,
    "learning_rate": 0.1,
    ...
  },
  "fit": {
    "early_stopping_rounds": 50,
    "verbose": false
  }
}
```

### 集成配置

**加权平均：**
```json
{
  "weights": {
    "xgb": 0.4,
    "lgb": 0.4, 
    "cat": 0.2
  }
}
```

**Stacking：**
```json
{
  "stacking": {
    "meta_model": "ridge",
    "alpha": 1.0,
    "cv_folds": 5,
    "random_state": 42
  }
}
```

## 🔧 高级用法

### 命令行参数详解

**训练模块参数：**
```bash
python -m tools.train_model --help
  --model {xgb,lgb,cat}     # 模型类型
  --target {Tg,Tc,Rg,FFV,Density}  # 目标变量
  --config CONFIG           # 配置文件路径
  --folds FOLDS            # 交叉验证折数
  --seed SEED              # 随机种子
  --train-path TRAIN_PATH   # 训练数据CSV文件路径
  --test-path TEST_PATH     # 测试数据CSV文件路径（可选）
```

**调优模块参数：**
```bash
python -m tools.tune --help
  --model {xgb,lgb,cat}     # 模型类型
  --target {Tg,Tc,Rg,FFV,Density}  # 目标变量
  --base-config CONFIG      # 基础配置文件
  --search-space SPACE      # 搜索空间文件
  --n-trials TRIALS        # 优化试验次数
  --study-name NAME         # Optuna study名称
  --timeout TIMEOUT         # 超时时间（秒）
  --train-path TRAIN_PATH   # 训练数据CSV文件路径
  --test-path TEST_PATH     # 测试数据CSV文件路径（可选）
```

**集成模块参数：**
```bash
python -m tools.ensemble --help
  --target {Tg,Tc,Rg,FFV,Density}  # 目标变量
  --models MODELS [MODELS ...]      # 要集成的模型列表
  --method {weighted,stacking}      # 集成方法
  --config CONFIG                   # 集成配置文件
  --output-name NAME                # 自定义输出名称
  --seed SEED                       # 随机种子
  --train-path TRAIN_PATH           # 训练数据CSV文件路径
  --create-multi-submission         # 创建多目标提交文件
```

### 自定义搜索空间

编辑 `configs/search_space.json` 来自定义超参数搜索范围：

```json
{
  "xgb": {
    "n_estimators": [100, 2000],
    "max_depth": [3, 10],
    "learning_rate": [0.01, 0.3],
    ...
  }
}
```

## 📝 日志和调试

- 所有操作都会生成详细日志到 `outputs/logs/`
- 使用 `--seed` 参数确保结果可复现
- 调优过程会保存试验历史到 `outputs/meta/tune_trials_*.csv`
- Optuna study会保存到数据库，支持断点续传

### 常用调试命令

```bash
# 查看训练日志
tail -f outputs/logs/xgb_Tg.log

# 查看调优进度
tail -f outputs/logs/tune_xgb_Tg.log

# 检查输出文件
ls outputs/oof/
ls outputs/preds/
ls outputs/models/
```

## 🏆 竞赛提交

### 生成提交文件
```bash
# 1. 训练所有模型
python -m tools.train_model --model xgb --target Tg --config configs/best_xgb_Tg.json --train-path kaggle/input/neurips-open-polymer-prediction-2025/train.csv --test-path kaggle/input/neurips-open-polymer-prediction-2025/test.csv
python -m tools.train_model --model lgb --target Tg --config configs/best_lgb_Tg.json --train-path kaggle/input/neurips-open-polymer-prediction-2025/train.csv --test-path kaggle/input/neurips-open-polymer-prediction-2025/test.csv
python -m tools.train_model --model cat --target Tg --config configs/best_cat_Tg.json --train-path kaggle/input/neurips-open-polymer-prediction-2025/train.csv --test-path kaggle/input/neurips-open-polymer-prediction-2025/test.csv

# 2. 生成集成预测和单目标提交文件
python -m tools.ensemble --target Tg --models xgb lgb cat --method stacking --output-name final --train-path kaggle/input/neurips-open-polymer-prediction-2025/train.csv

# 3. 生成多目标提交文件（需要先完成所有目标的集成）
python -m tools.ensemble --target Tg --models xgb lgb cat --method stacking --output-name final --train-path kaggle/input/neurips-open-polymer-prediction-2025/train.csv --create-multi-submission

# 4. 检查提交文件
head outputs/submissions/final_Tg_submission.csv
head outputs/submissions/final_multi_target_submission.csv
```

### 完整的多目标工作流
```bash
#!/bin/bash
targets=("Tg" "Tc" "Rg" "FFV" "Density")
models=("xgb" "lgb" "cat")
train_path="kaggle/input/neurips-open-polymer-prediction-2025/train.csv"
test_path="kaggle/input/neurips-open-polymer-prediction-2025/test.csv"

# 1. 训练所有模型和目标
for target in "${targets[@]}"; do
    for model in "${models[@]}"; do
        echo "Training $model for $target"
        python -m tools.train_model --model $model --target $target --config configs/${model}_base.json --train-path $train_path --test-path $test_path
    done
    
    # 2. 为每个目标创建集成
    echo "Ensembling for $target"
    python -m tools.ensemble --target $target --models xgb lgb cat --method stacking --output-name final --train-path $train_path
done

# 3. 创建最终的多目标提交文件
echo "Creating multi-target submission file"
python -m tools.ensemble --target Tg --models xgb lgb cat --method stacking --output-name final --train-path $train_path --create-multi-submission

echo "✅ Complete pipeline finished!"
echo "📁 Submission file: outputs/submissions/final_multi_target_submission.csv"
```

## 🐛 故障排除

### 常见问题和解决方案

**1. 导入错误**
```bash
# 确保当前目录正确
cd /path/to/my_wdmpnn

# 检查Python路径
python -c "import sys; print(sys.path)"

# 安装依赖
pip install -r requirements.txt
```

**2. 数据相关错误**
```bash
# XGBoost inf/nan错误
# 解决方案：数据预处理会自动清理极值，检查数据清理函数

# 数据文件路径错误
# 解决方案：确保使用正确的CSV文件路径
python -m tools.train_model --model xgb --target Tg --config configs/xgb_base.json --train-path path/to/train.csv
```

**3. 内存和性能问题**
```bash
# 减少内存使用
# 解决方案：使用较小的数据集或减少交叉验证折数
python -m tools.train_model --model xgb --target Tg --config configs/xgb_base.json --train-path train.csv --folds 3

# 加速训练
python -m tools.tune --model xgb --target Tg --base-config configs/xgb_base.json --train-path train.csv --n-trials 20  # 减少试验次数
```

**4. 结果不一致**
```bash
# 确保使用相同的随机种子
python -m tools.train_model --model xgb --target Tg --config configs/xgb_base.json --train-path train.csv --seed 42

# 检查配置文件是否相同
diff configs/xgb_base.json configs/best_xgb_Tg.json
```

**5. 文件路径问题**
```bash
# 检查输出目录是否存在
ls -la outputs/

# 创建缺失目录
mkdir -p outputs/{oof,preds,models,logs,meta}

# 检查配置文件
ls -la configs/
```

### 调试技巧

```bash
# 使用小数据集快速测试
python -m tools.train_model --model xgb --target Tg --config configs/xgb_base.json --train-path train.csv --folds 2

# 查看详细日志
tail -f outputs/logs/xgb_Tg.log

# 检查数据文件
head kaggle/input/neurips-open-polymer-prediction-2025/train.csv
wc -l kaggle/input/neurips-open-polymer-prediction-2025/train.csv
```

## 📄 许可证

MIT License
