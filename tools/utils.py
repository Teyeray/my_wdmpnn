"""
基础工具函数：JSON处理、随机种子、日志、路径管理
"""
import os
import json
import random
import logging
import numpy as np
from pathlib import Path


def load_json(path: str) -> dict:
    """加载JSON配置文件"""
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def save_json(obj: dict, path: str) -> None:
    """保存JSON文件"""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def set_seed(seed: int) -> None:
    """设置全局随机种子"""
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    # 设置各ML库的随机种子
    try:
        import xgboost
        # XGBoost通过参数设置
    except ImportError:
        pass
    
    try:
        import lightgbm
        # LightGBM通过参数设置
    except ImportError:
        pass
    
    try:
        import catboost
        # CatBoost通过参数设置
    except ImportError:
        pass


def ensure_dir(path: str) -> None:
    """确保目录存在"""
    os.makedirs(path, exist_ok=True)


def setup_logger(name: str, log_file: str = None) -> logging.Logger:
    """设置统一日志格式"""
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    
    # 避免重复添加handler
    if logger.handlers:
        return logger
    
    # 控制台输出
    console_handler = logging.StreamHandler()
    console_formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%H:%M:%S'
    )
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    
    # 文件输出
    if log_file:
        ensure_dir(os.path.dirname(log_file))
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
    
    return logger


def get_logger(name: str = __name__) -> logging.Logger:
    """获取logger实例"""
    return logging.getLogger(name)
