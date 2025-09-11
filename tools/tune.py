"""
基于Optuna的超参数优化脚本
通过子进程调用train_model.py进行模型训练
"""

import os
import argparse
import subprocess
import tempfile
import json
import optuna
from pathlib import Path
import pandas as pd
import re

from tools.utils import set_seed, load_json, save_json, setup_logger, ensure_dir

# 默认搜索空间
DEFAULT_SEARCH_SPACES = {
    "xgb": {
        "n_estimators": [100, 2000],
        "max_depth": [3, 10],
        "learning_rate": [0.01, 0.3],
        "subsample": [0.6, 1.0],
        "colsample_bytree": [0.6, 1.0],
        "reg_alpha": [0, 10],
        "reg_lambda": [1, 10],
        "min_child_weight": [1, 10],
        "gamma": [0, 5],
    },
    "lgb": {
        "n_estimators": [100, 2000],
        "max_depth": [3, 10],
        "learning_rate": [0.01, 0.3],
        "subsample": [0.6, 1.0],
        "colsample_bytree": [0.6, 1.0],
        "reg_alpha": [0, 10],
        "reg_lambda": [1, 10],
        "min_child_samples": [10, 100],
        "min_child_weight": [1e-5, 1e-1],
        "num_leaves": [10, 300],
    },
    "cat": {
        "iterations": [100, 2000],
        "depth": [3, 10],
        "learning_rate": [0.01, 0.3],
        "subsample": [0.6, 1.0],
        "colsample_bylevel": [0.6, 1.0],
        "reg_lambda": [1, 10],
        "min_child_samples": [1, 20],
        "random_strength": [0, 10],
        "bagging_temperature": [0, 1],
    },
}


def get_search_space(model_name: str, custom_space: dict = None):
    """获取搜索空间"""
    if custom_space and model_name in custom_space:
        return custom_space[model_name]
    return DEFAULT_SEARCH_SPACES.get(model_name, {})


def suggest_params(trial: optuna.trial.Trial, model_name: str, search_space: dict):
    """根据搜索空间建议参数"""
    params = {}

    for param_name, param_range in search_space.items():
        if isinstance(param_range, list) and len(param_range) == 2:
            if isinstance(param_range[0], int) and isinstance(param_range[1], int):
                # 整数参数
                params[param_name] = trial.suggest_int(
                    param_name, param_range[0], param_range[1]
                )
            elif isinstance(param_range[0], float) or isinstance(param_range[1], float):
                # 浮点数参数
                params[param_name] = trial.suggest_float(
                    param_name, param_range[0], param_range[1]
                )
        elif isinstance(param_range, list):
            # 分类参数
            params[param_name] = trial.suggest_categorical(param_name, param_range)

    return params


def create_temp_config(model_name: str, params: dict, base_config: dict):
    """创建临时配置文件"""
    config = base_config.copy()
    config["params"] = params

    # 创建临时文件
    temp_fd, temp_path = tempfile.mkstemp(suffix=".json", prefix=f"{model_name}_")

    try:
        with open(temp_path, "w") as f:
            json.dump(config, f, indent=2)
        os.close(temp_fd)  # 关闭文件描述符
        return temp_path
    except:
        os.close(temp_fd)
        raise


def run_training(
    model_name: str,
    target: str,
    config_path: str,
    folds: int,
    seed: int,
    logger,
    train_path: str,
    test_path: str = None,
    timeout: int = 36000,
) -> str:
    """通过子进程运行训练"""
    cmd = [
        "python",
        "-m",
        "tools.train_model",
        "--model",
        model_name,
        "--target",
        target,
        "--config",
        config_path,
        "--folds",
        str(folds),
        "--seed",
        str(seed),
        "--train-path",
        train_path,
    ]

    # 添加测试数据路径（如果提供）
    if test_path:
        cmd.extend(["--test-path", test_path])

    logger.debug(f"Running command: {' '.join(cmd)}")

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,  # 1小时超时
            cwd=os.getcwd(),
        )

        if result.returncode != 0:
            logger.error(f"Training failed with return code {result.returncode}")
            logger.error(f"STDERR: {result.stderr}")
            raise RuntimeError(f"Training subprocess failed: {result.stderr}")

        return result.stdout

    except subprocess.TimeoutExpired:
        logger.error("Training timeout")
        raise RuntimeError("Training timeout")
    except Exception as e:
        logger.error(f"Subprocess error: {str(e)}")
        raise


def parse_result(output: str, logger):
    """从输出中解析结果"""
    lines = output.strip().split("\n")

    for line in lines:
        if "[RESULT]" in line:
            # 使用正则表达式提取wMAE值
            wmae_match = re.search(r"CV_wMAE=([0-9.]+)", line)
            if wmae_match:
                wmae = float(wmae_match.group(1))
                logger.debug(f"Parsed wMAE: {wmae}")
                return wmae

    logger.error("Failed to parse wMAE from output")
    logger.error(f"Output: {output}")
    raise ValueError("Could not parse training result")


class OptimizationObjective:
    """Optuna优化目标函数"""

    def __init__(
        self,
        model_name: str,
        target: str,
        search_space: dict,
        base_config: dict,
        folds: int,
        seed: int,
        logger,
        train_path: str,
        test_path: str = None,
    ):
        self.model_name = model_name
        self.target = target
        self.search_space = search_space
        self.base_config = base_config
        self.folds = folds
        self.seed = seed
        self.logger = logger
        self.trial_count = -1
        self.train_path = train_path
        self.test_path = test_path

    def __call__(self, trial: optuna.trial.Trial):
        self.trial_count += 1
        self.logger.info(f"Trial {self.trial_count}: {trial.number}")

        # 建议参数
        params = suggest_params(trial, self.model_name, self.search_space)
        self.logger.info(f"Suggested params: {params}")

        # 创建临时配置文件
        temp_config_path = create_temp_config(self.model_name, params, self.base_config)

        try:
            # 运行训练
            output = run_training(
                self.model_name,
                self.target,
                temp_config_path,
                self.folds,
                self.seed,
                self.logger,
                self.train_path,
                self.test_path,
            )

            # 解析结果
            wmae = parse_result(output, self.logger)

            self.logger.info(f"Trial {trial.number} result: wMAE={wmae:.6f}")
            return wmae

        finally:
            # 清理临时文件
            if os.path.exists(temp_config_path):
                os.remove(temp_config_path)


def save_optimization_results(
    study: optuna.study.Study, model_name: str, target: str, logger
):
    """保存优化结果"""
    ensure_dir("outputs/meta")

    # 最佳参数
    best_params = study.best_params
    best_value = study.best_value

    results = {
        "best_params": best_params,
        "best_wmae": best_value,
        "n_trials": len(study.trials),
        "model_name": model_name,
        "target": target,
    }

    # 保存结果
    result_path = f"outputs/meta/tune_{model_name}_{target}.json"
    save_json(results, result_path)
    logger.info(f"Optimization results saved: {result_path}")

    # 详细试验历史
    trials_df = study.trials_dataframe()
    trials_path = f"outputs/meta/tune_trials_{model_name}_{target}.csv"
    trials_df.to_csv(trials_path, index=False)
    logger.info(f"Trials history saved: {trials_path}")

    return results


def create_best_config(
    best_params: dict, base_config: dict, model_name: str, target: str, logger
):
    """创建最佳参数配置文件"""
    ensure_dir("configs")

    best_config = base_config.copy()
    best_config["params"] = best_params

    config_path = f"configs/best_{model_name}_{target}.json"
    save_json(best_config, config_path)
    logger.info(f"Best config saved: {config_path}")

    return config_path


def main():
    parser = argparse.ArgumentParser(
        description="Hyperparameter optimization with Optuna"
    )
    parser.add_argument(
        "--model", required=True, choices=["xgb", "lgb", "cat"], help="Model type"
    )
    parser.add_argument(
        "--target",
        required=True,
        choices=["Tg", "Tc", "Rg", "FFV", "Density"],
        help="Target variable",
    )
    parser.add_argument(
        "--base-config", required=True, help="Base JSON config file path"
    )
    parser.add_argument("--search-space", help="Custom search space JSON file")
    parser.add_argument(
        "--n-trials", type=int, default=100, help="Number of optimization trials"
    )
    parser.add_argument("--folds", type=int, default=5, help="Number of CV folds")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--study-name", help="Custom study name")
    parser.add_argument("--timeout", type=int, help="Optimization timeout in seconds")

    # 数据路径参数
    parser.add_argument('--train-path', required=True,
                       help='Path to training CSV file')
    parser.add_argument('--test-path', default=None,
                       help='Path to test CSV file (optional)')

    args = parser.parse_args()

    # 设置随机种子
    set_seed(args.seed)

    # 设置日志
    study_name = args.study_name or f"tune_{args.model}_{args.target}"
    log_file = f"outputs/logs/{study_name}.log"
    logger = setup_logger(study_name, log_file)

    logger.info(f"Starting hyperparameter optimization")
    logger.info(f"Model: {args.model}, Target: {args.target}")
    logger.info(f"Trials: {args.n_trials}, Folds: {args.folds}")
    logger.info(f"Training data: {args.train_path}")
    logger.info(f"Test data: {args.test_path}")

    try:
        # 加载基础配置
        base_config = load_json(args.base_config)
        logger.info(f"Base config loaded: {args.base_config}")

        # 加载搜索空间
        custom_search_space = {}
        if args.search_space:
            custom_search_space = load_json(args.search_space)
            logger.info(f"Custom search space loaded: {args.search_space}")

        search_space = get_search_space(args.model, custom_search_space)
        logger.info(f"Search space: {search_space}")

        # 创建Optuna study
        study = optuna.create_study(
            direction="minimize", study_name=study_name  # 最小化wMAE
        )

        # 创建目标函数
        objective = OptimizationObjective(
            args.model,
            args.target,
            search_space,
            base_config,
            args.folds,
            args.seed,
            logger,
            args.train_path,
            args.test_path,
        )

        # 运行优化
        study.optimize(objective, n_trials=args.n_trials, timeout=args.timeout)

        # 保存结果
        results = save_optimization_results(study, args.model, args.target, logger)

        # 创建最佳配置文件
        best_config_path = create_best_config(
            study.best_params, base_config, args.model, args.target, logger
        )

        # 输出最终结果
        logger.info("Optimization completed!")
        logger.info(f"Best wMAE: {study.best_value:.6f}")
        logger.info(f"Best params: {study.best_params}")
        logger.info(f"Best config saved: {best_config_path}")

        # 标准输出（供外部脚本解析）
        print(
            f"[TUNE_RESULT] Model={args.model.upper()} | Target={args.target} | "
            f"Best_wMAE={study.best_value:.6f} | Trials={len(study.trials)} | "
            f"Best_config={best_config_path}"
        )

    except Exception as e:
        logger.error(f"Optimization failed: {str(e)}")
        raise


if __name__ == "__main__":
    main()
