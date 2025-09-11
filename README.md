# my_wdmpnn
# 🧪 Polymer Property Prediction ML Framework

A modular machine learning framework for the Kaggle Polymer Property Prediction competition, supporting XGBoost, LightGBM, and CatBoost with automated hyperparameter tuning and ensemble methods.

## 📁 Project Structure

```
my_wdmpnn/
├── utils/                          # 核心模块
│   ├── __init__.py
│   ├── utils.py                    # 基础工具：JSON、日志、随机种子
│   ├── data.py                     # 数据处理（已存在）
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
│   └── meta/                      # 元数据（调优结果等）
├── datasets/                       # 数据集
├── main_pipeline.py               # 主流水线示例
└── README.md                      # 本文件
```

## 🚀 Quick Start

### 1. 单模型训练

```bash
# 使用基础配置训练XGBoost
python -m utils.train_model --model xgb --target Tg --config configs/xgb_base.json --folds 5

# 训练所有模型
for model in xgb lgb cat; do
    python -m utils.train_model --model $model --target Tg --config configs/${model}_base.json --folds 5
done
```

### 2. 超参数优化

```bash
# 优化XGBoost超参数
python -m utils.tune --model xgb --target Tg --base-config configs/xgb_base.json --n-trials 100

# 使用优化后的配置训练
python -m utils.train_model --model xgb --target Tg --config configs/best_xgb_Tg.json --folds 5
```

### 3. 模型集成

```bash
# 加权平均集成
python -m utils.ensemble --target Tg --models xgb lgb cat --method weighted --config configs/ensemble_weighted.json

# Stacking集成
python -m utils.ensemble --target Tg --models xgb lgb cat --method stacking --config configs/ensemble_stacking.json
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

### 命令行参数

所有模块都支持丰富的命令行参数：

```bash
# 查看完整参数列表
python -m utils.train_model --help
python -m utils.tune --help
python -m utils.ensemble --help
```

### 自定义评估指标

修改 `utils/evaluate.py` 中的 `get_default_ranges()` 和 `get_default_counts()` 函数来调整wMAE计算参数。

## 📝 日志和调试

- 所有操作都会生成详细日志到 `outputs/logs/`
- 使用 `--seed` 参数确保结果可复现
- 调优过程会保存试验历史到 `outputs/meta/tune_trials_*.csv`

## 🏆 竞赛提交

1. 确保有测试集预测文件：`outputs/preds/ensemble_*.csv`
2. 根据竞赛要求格式化提交文件
3. 选择验证集wMAE最低的集成方法

## 🐛 故障排除

- **内存不足**：减少 `n_estimators` 或使用更少的特征
- **训练太慢**：减少 `n_trials` 或使用更小的搜索空间
- **结果不一致**：确保使用相同的 `--seed` 参数
- **导入错误**：确保安装所需依赖：`pip install -r requirements.txt`

## 📄 许可证

MIT License
