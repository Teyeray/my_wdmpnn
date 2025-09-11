import ast
import math
import torch
import pickle
import logging
import numpy as np
import os, re, json
import pandas as pd
from rdkit import Chem
from torch import Tensor
from pathlib import Path
from collections import Counter
from rich.progress import Progress
from torch.utils.data import Dataset
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from rdkit.ML.Descriptors import MoleculeDescriptors
from sklearn.model_selection import train_test_split
from rdkit.Chem import Descriptors, rdMolDescriptors, AllChem
from rdkit.Chem.rdMolDescriptors import CalcNumRotatableBonds
from typing import List, Tuple, Optional, Dict, Union, Iterable
from mordred import Calculator, descriptors as mordred_descriptors

# Configure logger
def setup_logger(name: str = "data_processor", level: int = logging.INFO) -> logging.Logger:
    """Setup logger with consistent formatting"""
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%H:%M:%S'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    
    return logger

# Global logger instance
logger = setup_logger()


def save_columns_to_json(df: pd.DataFrame, name: str):
    """Save DataFrame columns to JSON file"""
    cols = df.columns.tolist()
    os.makedirs("datasets", exist_ok=True)
    with open(f"datasets/{name}.json", "w", encoding="utf-8") as f:
        json.dump(cols, f, ensure_ascii=False, indent=2)
    logger.info(f"Saved datasets/{name}.json with {len(cols)} columns")
    return cols


def load_feature_cache(cache_path):
    if os.path.exists(cache_path):
        with open(cache_path, "rb") as f:
            return pickle.load(f)
    return {}


def save_feature_cache(cache_path, cache_dict):
    if not os.path.exists(os.path.dirname(cache_path)):
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    with open(cache_path, "wb") as f:
        pickle.dump(cache_dict, f)


# Preliminary processing of the raw data
def get_data_paths(print_paths: bool = False) -> dict:
    BASE_PATH = Path("kaggle/input/neurips-open-polymer-prediction-2025")
    EXTRA_BASE = Path("kaggle/input/smiles-extra-data")
    TC_BASE = Path("kaggle/input/tc-smiles")

    paths = {
        "train_csv": Path(BASE_PATH / "train.csv"),
        "test_csv": Path(BASE_PATH / "test.csv"),
        "sample_submission": Path(BASE_PATH / "sample_submission.csv"),
        "tc_smiles": Path(TC_BASE / "Tc_SMILES.csv"),
        "sed_bigsmiles": Path(EXTRA_BASE / "JCIM_sup_bigsmiles.csv"),
        "sed_tg3": Path(EXTRA_BASE / "data_tg3.xlsx"),
        "sed_dnst1": Path(EXTRA_BASE / "data_dnst1.xlsx"),
        "dataset4": Path(BASE_PATH / "train_supplement" / "dataset4.csv"),
        "dataset1": Path(BASE_PATH / "train_supplement" / "dataset1.csv"),
        "dataset2": Path(BASE_PATH / "train_supplement" / "dataset2.csv"),
        "dataset3": Path(BASE_PATH / "train_supplement" / "dataset3.csv"),
    }
    for key, path in paths.items():
        if not path.is_absolute():
            paths[key] = path.resolve()
        if paths[key].exists():
            if print_paths:
                logger.info(f"{key} found at {path} [OK]")
        else:
            logger.error(f"{key} not found at {path} [ERROR]")
    return paths


def get_train_test():
    P = get_data_paths(print_paths=True)
    train = pd.read_csv(P["train_csv"])
    test = pd.read_csv(P["test_csv"])
    sub = pd.read_csv(P["sample_submission"])
    return train, test, sub


def combine_data(
    train: pd.DataFrame,
    extra: Union[str, Path, pd.DataFrame],
    target: str,
    source_name: str,
) -> pd.DataFrame:
    if isinstance(extra, pd.DataFrame):
        df_extra = extra.rename(columns={source_name: target}).copy()
        logger.info("=" * 60)
        logger.info(
            f"[START] Adding extra data targeting '{source_name}' to train target '{target}'."
        )
    else:
        if not Path(extra).exists():
            logger.error(
                f"Extra data file '{extra}' does not exist. Returning original train data."
            )
            return train
        df_extra = pd.read_csv(extra).rename(columns={source_name: target}).copy()
        logger.info("=" * 60)
        logger.info(
            f"[START] Adding extra data from '{extra}' targeting '{source_name}' to train target '{target}'."
        )

    logger.info(f"Train data shape: {train.shape}")
    df_train = train.copy()

    logger.info(
        f"Extra data shape: {df_extra.shape}, columns: {df_extra.columns.tolist()}"
    )
    df_extra = df_extra[["SMILES", target]].dropna(subset=["SMILES", target])
    df_extra = df_extra.groupby("SMILES", as_index=False).mean()

    # Find common SMILES
    common = set(df_train["SMILES"]) & set(df_extra["SMILES"])
    logger.info(f"Found {len(common)} overlapping SMILES.")

    if common:
        for smi in common:
            # Check if train's target is NaN
            train_target_value = df_train.loc[df_train["SMILES"] == smi, target]
            if train_target_value.isna().all():
                # If train's target is NaN, use extra's value
                extra_value = df_extra.loc[df_extra["SMILES"] == smi, target].values[0]
                df_train.loc[df_train["SMILES"] == smi, target] = extra_value
                logger.info(
                    f"[ADD] Updated target for SMILES '{smi}' and value '{extra_value}' from extra data target '{target}'."
                )
            else:
                # Otherwise, drop the SMILES from extra
                df_extra.drop(df_extra[df_extra["SMILES"] == smi].index, inplace=True)
                # logger.debug(f"Dropped SMILES '{smi}' from extra data.")

    # Merge train and extra
    df_combined = pd.concat([df_train, df_extra], ignore_index=True)
    logger.info(f"Combined data shape: {df_combined.shape}")
    return df_combined


def make_smile_canonical(smile: str):
    try:
        mol = Chem.MolFromSmiles(smile)
        if mol is None:
            return np.nan
        return Chem.MolToSmiles(mol, canonical=True)
    except Exception:
        return np.nan


def add_extra_data(train: pd.DataFrame) -> pd.DataFrame:
    P = get_data_paths()
    # read extra data
    train = train.copy()

    train = combine_data(train, P["tc_smiles"], target="Tc", source_name="TC_mean")

    train = combine_data(train, P["sed_bigsmiles"], target="Tg", source_name="Tg (C)")

    tg_excel_data = pd.read_excel(P["sed_tg3"]).assign(
        Tg_K=lambda df: df["Tg [K]"] - 273.15
    )
    train = combine_data(train, tg_excel_data, target="Tg", source_name="Tg_K")

    train = combine_data(train, P["dataset3"], target="Tg", source_name="dataset3")

    density_data = (
        pd.read_excel(P["sed_dnst1"])
        .rename(columns={"density(g/cm3)": "Density"})
        .assign(
            Density=lambda df: pd.to_numeric(df["Density"], errors="coerce") - 0.118
        )
    )
    train = combine_data(train, density_data, target="Density", source_name="Density")

    train = combine_data(train, P["dataset1"], target="Tc", source_name="TC_mean")

    train = combine_data(train, P["dataset4"], target="FFV", source_name="FFV")

    return train


def _compute_all_string_features(smiles: str) -> dict:
    """提取 SMILES 字符串的字符/化学符号/占位符特征（不依赖 RDKit）"""
    if not isinstance(smiles, str):
        smiles = str(smiles)
    feats = {}

    # 基础统计
    feats["smiles_length"] = len(smiles)
    feats["capital_letters"] = sum(c.isupper() for c in smiles)
    feats["lowercase_letters"] = sum(c.islower() for c in smiles)
    feats["digits"] = sum(c.isdigit() for c in smiles)

    # 符号统计
    feats["parentheses"] = smiles.count("(") + smiles.count(")")
    feats["brackets"] = smiles.count("[") + smiles.count("]")
    feats["braces"] = smiles.count("{") + smiles.count("}")
    feats["equals"] = smiles.count("=")
    feats["hashes"] = smiles.count("#")
    feats["colons"] = smiles.count(":")
    feats["ats"] = smiles.count("@")
    feats["slashes"] = smiles.count("/") + smiles.count("\\")
    feats["plus_minus"] = smiles.count("+") + smiles.count("-")

    # 元素计数
    feats["C_count"] = smiles.count("C") + smiles.count("c")
    feats["O_count"] = smiles.count("O") + smiles.count("o")
    feats["N_count"] = smiles.count("N") + smiles.count("n")
    feats["S_count"] = smiles.count("S") + smiles.count("s")
    feats["P_count"] = smiles.count("P") + smiles.count("p")
    feats["F_count"] = smiles.count("F") + smiles.count("f")
    feats["Cl_count"] = smiles.count("Cl") + smiles.count("cl")
    feats["Br_count"] = smiles.count("Br") + smiles.count("br")
    feats["I_count"] = smiles.count("I") + smiles.count("i")

    # 结构模式
    feats["has_ring"] = int(any(d in smiles for d in "123456789"))
    feats["has_double_bond"] = int("=" in smiles)
    feats["has_triple_bond"] = int("#" in smiles)
    feats["has_aromatic"] = int(any(c in smiles for c in "cnos"))

    # 元素比例
    feats["O_to_C_ratio"] = feats["O_count"] / (feats["C_count"] + 1e-5)
    feats["N_to_C_ratio"] = feats["N_count"] / (feats["C_count"] + 1e-5)
    feats["heteroatom_ratio"] = (
        feats["O_count"] + feats["N_count"] + feats["S_count"] + feats["P_count"]
    ) / (feats["C_count"] + 1e-5)

    # 占位符特征
    feats["star_count"] = smiles.count("*")
    feats["R_placeholder_count"] = len(re.findall(r"\[R[0-9']*\]", smiles))
    feats["any_placeholder"] = int(
        (feats["star_count"] > 0) or (feats["R_placeholder_count"] > 0)
    )

    return feats


def clean_smiles(df: pd.DataFrame) -> pd.DataFrame:
    """Clean and canonicalize SMILES strings, then group duplicates"""
    df = df.copy()
    df["SMILES"] = df["SMILES"].apply(make_smile_canonical)
    df = df.dropna(subset=["SMILES"]).reset_index(drop=True)

    gb = df.groupby("SMILES", as_index=False)
    logger.info(f"We have {len(df)} entries, {len(gb)} are unique SMILES.")
    df = gb.mean(numeric_only=True)
    logger.info(f"After grouping by SMILES and averaging, we have {len(df)} entries.")
    return df


def _generate_rdkit_features(smiles_str: str) -> np.ndarray:
    """
    生成RDKit描述符和Morgan指纹

    Args:
        smiles_str: SMILES字符串

    Returns:
        包含所有RDKit描述符和Morgan指纹的numpy数组
    """
    mol = Chem.MolFromSmiles(smiles_str)

    # 获取所有可用的RDKit描述符
    desc_list = [d[0] for d in Descriptors._descList]
    calculator = MoleculeDescriptors.MolecularDescriptorCalculator(desc_list)

    morgan_fp_size = 1024

    if mol is None:
        return np.full(len(desc_list) + morgan_fp_size, np.nan)

    # 计算分子描述符
    descriptors = np.array(calculator.CalcDescriptors(mol))

    # 计算Morgan指纹
    mfp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=morgan_fp_size)
    mfp_array = np.array(list(mfp.ToBitString())).astype(int)

    # 连接描述符和指纹
    return np.concatenate([descriptors, mfp_array])


def _filter_dataset(
    df: pd.DataFrame, column: str, lower_bound: float, upper_bound: float
) -> pd.DataFrame:
    """Filter dataset by column value range while preserving NaN values"""
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame.")
    logger.info(
        f"Original data shape: {df.shape}; non-null '{column}': {df[column].notna().sum()}"
    )
    # Keep rows that are within the range, or those where the column is missing (skip filtering if missing)
    mask_in_range = (df[column] >= lower_bound) & (df[column] <= upper_bound)
    mask_keep = df[column].isna() | mask_in_range
    filtered_df = df[mask_keep].copy()
    dropped = len(df) - len(filtered_df)
    logger.info(
        f"Dropped {dropped} rows from '{column}' (kept NaN and values in [{lower_bound}, {upper_bound}])."
    )
    logger.info(f"Filtered data shape: {filtered_df.shape}")
    return filtered_df


def filter_train_data(df: pd.DataFrame) -> pd.DataFrame:
    # filter according to the specified ranges, keeping NaNs
    tg_filtered = _filter_dataset(df, "Tg", -223, 480)
    tc_filtered = _filter_dataset(tg_filtered, "Tc", 0, 0.54)
    rg_filtered = _filter_dataset(tc_filtered, "Rg", 0, 33)
    ffv_filtered = _filter_dataset(rg_filtered, "FFV", 0.1, 0.6)
    de_filtered = _filter_dataset(ffv_filtered, "Density", 0, 1.76)
    return de_filtered


def _replace_all_R_with_C(smi: str) -> str:
    # Replace any R placeholders with C:
    #  - bracketed R-groups like [R], [R'], [R1] -> C
    #  - any remaining uppercase 'R' anywhere -> C
    # Does not touch lowercase letters (e.g. 'r') or other characters.
    s = str(smi)
    # 1) replace bracketed R-groups
    s = re.sub(r"\[R[^\]]*\]", "C", s)
    # 2) replace any remaining uppercase R
    s = re.sub(r"R", "C", s)
    return s


def replace_all_R_with_C(df: pd.DataFrame, smiles_col: str = "SMILES") -> pd.DataFrame:
    df = df.copy()
    df[smiles_col] = df[smiles_col].apply(_replace_all_R_with_C)
    return df


def add_smiles_string_features(
    df: pd.DataFrame, smiles_col: str = "SMILES"
) -> pd.DataFrame:
    """Add SMILES string-based features to DataFrame"""
    df = df.copy()

    logger.info(f"Adding SMILES string features to {len(df)} molecules...")

    smiles_features = []
    with Progress() as progress:
        task = progress.add_task("Extracting SMILES string features", total=len(df))
        for smiles in df[smiles_col]:
            features = _compute_all_string_features(smiles)
            # Add prefix to distinguish from other features
            features = {f"smiles_string_{k}": v for k, v in features.items()}
            smiles_features.append(features)
            progress.update(task, advance=1)

    # Convert to DataFrame and merge
    smiles_df = pd.DataFrame(smiles_features)
    result_df = pd.concat(
        [df.reset_index(drop=True), smiles_df.reset_index(drop=True)], axis=1
    )

    logger.info(f"Added {len(smiles_df.columns)} SMILES string features")
    return result_df


def add_rdkit_features(
    df, smiles_col="SMILES", cache_path="datasets/cache/rdkit_features.pkl"
):
    """Add RDKit-based features to DataFrame with progress tracking and caching"""
    logger.info("Starting RDKit feature extraction...")
    
    desc_list_names = [d[0] for d in Descriptors._descList]
    fp_morgan_cols = [f"rdkit_mfp_{i}" for i in range(1024)]  # 修改这里
    feature_columns = [
        f"rdkit_{name}" for name in desc_list_names
    ] + fp_morgan_cols  # 修改这里

    cache = load_feature_cache(cache_path)
    new_cache = {}
    features_list = []

    logger.info(f"Processing {len(df)} molecules for RDKit features...")
    
    with Progress() as progress:
        task = progress.add_task("计算RDKit特征", total=len(df))
        for idx, smiles in enumerate(df[smiles_col]):
            if idx % 1000 == 0:
                logger.info(f"Processing RDKit features: {idx+1}/{len(df)} ({(idx+1)/len(df)*100:.1f}%)")
                
            if pd.isna(smiles):
                features = np.full(len(feature_columns), np.nan)
            elif smiles in cache:
                features = cache[smiles]
            else:
                features = _generate_rdkit_features(smiles)
                new_cache[smiles] = features
            features_list.append(features)
            progress.update(task, advance=1)

    # 更新缓存
    cache.update(new_cache)
    save_feature_cache(cache_path, cache)
    logger.info(f"Updated cache with {len(new_cache)} new features")

    features_df = pd.DataFrame(features_list, columns=feature_columns)
    features_df.replace([np.inf, -np.inf], np.nan, inplace=True)
    # features_df.fillna(features_df.mean(), inplace=True)

    result_df = pd.concat(
        [df.reset_index(drop=True), features_df.reset_index(drop=True)], axis=1
    )
    logger.info(f"Added {len(feature_columns)} RDKit features to DataFrame")
    return result_df


def align_test_features_with_train(
    test_df: pd.DataFrame,
    train_df: pd.DataFrame,
    target_cols: List[str],
    id_col: str = "id",
) -> pd.DataFrame:
    """
    确保测试集具有与训练集相同的特征列

    Args:
        test_df: 测试集DataFrame
        train_df: 训练集DataFrame
        target_cols: 目标列名列表
        id_col: ID列名

    Returns:
        对齐后的测试集DataFrame
    """
    test_df = test_df.copy()

    # 获取训练集的特征列（排除目标列）
    train_feature_cols = [col for col in train_df.columns if col not in target_cols]

    # 测试集应该有的列：id + 所有训练集特征列（除了id，如果训练集有的话）
    expected_test_cols = [id_col] + [col for col in train_feature_cols if col != id_col]

    # 检查缺失的列
    missing_cols = [col for col in expected_test_cols if col not in test_df.columns]
    if missing_cols:
        logger.info(f"Adding missing columns to test set: {missing_cols}")
        for col in missing_cols:
            test_df[col] = np.nan

    # 检查多余的列
    extra_cols = [col for col in test_df.columns if col not in expected_test_cols]
    if extra_cols:
        logger.info(
            f"Removing extra columns from test set: {extra_cols[:10]}{', ...' if len(extra_cols) > 10 else ''}"
        )
        test_df = test_df.drop(columns=extra_cols)

    # 确保列的顺序一致
    test_df = test_df[expected_test_cols]

    logger.info(f"Test features aligned: {test_df.shape}")
    return test_df


def add_mordred_features(
    df, smiles_col="SMILES", cache_path="datasets/cache/mordred_features.pkl"
):
    mordred_calc = Calculator(mordred_descriptors, ignore_3D=True)
    cache = load_feature_cache(cache_path)
    new_cache = {}
    features_list = []

    # 获取所有Mordred描述符名，并加前缀
    mordred_desc_names = [str(d) for d in mordred_calc.descriptors]
    feature_columns = [f"mordred_{name}" for name in mordred_desc_names]

    with Progress() as progress:
        task = progress.add_task("计算Mordred特征", total=len(df))
        for smiles in df[smiles_col]:
            if pd.isna(smiles):
                features = np.full(len(feature_columns), np.nan)
            elif smiles in cache:
                features = cache[smiles]
            else:
                mol = Chem.MolFromSmiles(smiles)
                if mol is None:
                    features = np.full(len(feature_columns), np.nan)
                else:
                    try:
                        vals = mordred_calc(mol)
                        # 只保留数值型特征
                        vals = [
                            (
                                vals[d]
                                if isinstance(
                                    vals[d], (int, float, np.integer, np.floating)
                                )
                                else np.nan
                            )
                            for d in mordred_calc.descriptors
                        ]
                        features = np.array(vals)
                    except Exception:
                        features = np.full(len(feature_columns), np.nan)
                new_cache[smiles] = features
            features_list.append(features)
            progress.update(task, advance=1)

    # 更新缓存
    cache.update(new_cache)
    save_feature_cache(cache_path, cache)

    features_df = pd.DataFrame(features_list, columns=feature_columns)
    features_df.replace([np.inf, -np.inf], np.nan, inplace=True)
    # features_df.fillna(features_df.median(), inplace=True)

    result_df = pd.concat(
        [df.reset_index(drop=True), features_df.reset_index(drop=True)], axis=1
    )
    logger.info(f"Added {len(feature_columns)} Mordred features to DataFrame")
    return result_df


def remove_highly_correlated_features(
    df: pd.DataFrame, threshold: float = 0.95, exclude_cols: List[str] = None
) -> pd.DataFrame:
    """Remove one of each pair of highly correlated features"""
    if exclude_cols is None:
        exclude_cols = ["SMILES", "id"]

    df = df.copy()

    # Get feature columns (exclude non-feature columns)
    feature_cols = [col for col in df.columns if col not in exclude_cols]

    # Select only numeric columns
    numeric_cols = df[feature_cols].select_dtypes(include=[np.number]).columns.tolist()

    if len(numeric_cols) < 2:
        logger.warning("Not enough numeric features for correlation analysis")
        return df

    logger.info(f"Analyzing correlations for {len(numeric_cols)} numeric features...")

    # Calculate correlation matrix with progress bar
    with Progress() as progress:
        task = progress.add_task("计算相关性矩阵", total=1)
        corr_matrix = df[numeric_cols].corr().abs()
        progress.update(task, advance=1)

    # Find pairs of highly correlated features with progress bar
    high_corr_pairs = []
    total_pairs = len(corr_matrix.columns) * (len(corr_matrix.columns) - 1) // 2

    with Progress() as progress:
        task = progress.add_task("查找高相关性特征对", total=total_pairs)
        for i in range(len(corr_matrix.columns)):
            for j in range(i + 1, len(corr_matrix.columns)):
                if corr_matrix.iloc[i, j] >= threshold:
                    col1 = corr_matrix.columns[i]
                    col2 = corr_matrix.columns[j]
                    high_corr_pairs.append((col1, col2, corr_matrix.iloc[i, j]))
                progress.update(task, advance=1)

    logger.info(
        f"Found {len(high_corr_pairs)} highly correlated pairs (correlation >= {threshold})"
    )

    # Decide which features to remove
    features_to_remove = set()

    with Progress() as progress:
        task = progress.add_task("决定移除特征", total=len(high_corr_pairs))
        for col1, col2, corr_val in high_corr_pairs:
            if col1 not in features_to_remove and col2 not in features_to_remove:
                # Prefer to keep features with less missing values
                col1_missing = df[col1].isna().sum()
                col2_missing = df[col2].isna().sum()

                if col1_missing > col2_missing:
                    features_to_remove.add(col1)
                elif col2_missing > col1_missing:
                    features_to_remove.add(col2)
                else:
                    # If equal missing values, remove the one that comes later alphabetically
                    features_to_remove.add(max(col1, col2))
            progress.update(task, advance=1)

    # Remove the features
    if features_to_remove:
        df = df.drop(columns=list(features_to_remove))
        logger.info(f"Removed {len(features_to_remove)} highly correlated features")
    else:
        logger.info("No features removed")

    return df


def drop_high_missing_and_impute_median(
    df: pd.DataFrame,
    threshold: float = 0.5,
    exclude_cols: Optional[List[str]] = None,
    verbose: bool = True,
    drop: bool = True,
    variance_threshold: float = 0.01,
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Drop columns whose missing rate > threshold and low variance features, then fill remaining numeric NaNs with column median.

    Args:
        df: input DataFrame (will not be modified in-place).
        threshold: drop column if fraction of NaN > threshold (0..1).
        exclude_cols: list of column names never to drop (e.g. ["id","SMILES"]).
        verbose: print summary when True.
        drop: if True, drop high-missing and low-variance columns; if False, only impute.
        variance_threshold: variance threshold for removing low-variance features.

    Returns:
        (cleaned_df, all_dropped_columns)
    """
    if exclude_cols is None:
        exclude_cols = []
    exclude_cols = set(exclude_cols)
    df_copy = df.copy()

    # missing fraction per column
    na_frac = df_copy.isna().mean()

    all_dropped_cols = []

    # Step 1: Drop high missing columns (respect exclude list and drop flag)
    if drop:
        drop_cols = [
            c for c, f in na_frac.items() if (f > threshold and c not in exclude_cols)
        ]
        if drop_cols:
            df_copy = df_copy.drop(columns=drop_cols)
            all_dropped_cols.extend(drop_cols)
            if verbose:
                logger.info(
                    f"Dropped {len(drop_cols)} high-missing columns (>{threshold*100:.1f}% missing)"
                )

        # Step 2: Drop low variance features
        numeric_cols = df_copy.select_dtypes(include=[np.number]).columns.tolist()
        feature_cols = [col for col in numeric_cols if col not in exclude_cols]

        if feature_cols:
            variances = df_copy[feature_cols].var()
            low_var_cols = variances[variances < variance_threshold].index.tolist()

            if low_var_cols:
                df_copy = df_copy.drop(columns=low_var_cols)
                all_dropped_cols.extend(low_var_cols)
                if verbose:
                    logger.info(
                        f"Dropped {len(low_var_cols)} low-variance columns (<{variance_threshold} variance)"
                    )
    else:
        if verbose:
            logger.info("Skipping column dropping (drop=False)")

    # Step 3: Impute numeric columns by median (leave non-numeric as-is)
    numeric_cols = df_copy.select_dtypes(include=[np.number]).columns.tolist()
    if numeric_cols:
        medians = df_copy[numeric_cols].median()
        df_copy[numeric_cols] = df_copy[numeric_cols].fillna(medians)

    if verbose:
        kept = df_copy.shape[1]
        total = len(na_frac)
        action = "dropped and imputed" if drop else "only imputed"
        logger.info(
            f"drop_high_missing_and_impute_median ({action}): dropped {len(all_dropped_cols)} / {total} cols, kept {kept}"
        )
        if all_dropped_cols and len(all_dropped_cols) <= 20:
            logger.info(f"  dropped columns: {all_dropped_cols}")
        elif all_dropped_cols:
            logger.info(f"  dropped example (first 20): {all_dropped_cols[:20]}")

    return df_copy, all_dropped_cols


def print_feature_statistics(
    df: pd.DataFrame, prefix_list: List[str] = ["rdkit_", "mordred_", "smiles_string_"]
):
    stats = {}
    for prefix in prefix_list:
        cols = [c for c in df.columns if c.startswith(prefix)]
        stats[prefix] = {
            "count": len(cols),
            "mean_nan_ratio": df[cols].isna().mean().mean() if cols else None,
            "median_nan_ratio": df[cols].isna().mean().median() if cols else None,
        }
    logger.info("=== Feature Statistics ===")
    for prefix, info in stats.items():
        logger.info(
            f"{prefix}: {info['count']} columns, avg missing: {info['mean_nan_ratio']:.4f}, median missing: {info['median_nan_ratio']:.4f}"
            if info["count"]
            else f"{prefix}: 0 columns"
        )
    logger.info("=" * 30)
    return


def process_train_test_data(
    save_files: bool = False,
    use_string: bool = True,
    use_rdkit: bool = True,
    use_mordred: bool = True,
    columns: List[str] = None,
    save_tail: str = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    完整的训练和测试数据处理流程，确保特征一致性

    Args:
        save_files: 是否保存文件
        use_string: 是否使用SMILES字符串特征
        use_rdkit: 是否使用RDKit特征
        use_mordred: 是否使用Mordred特征
        columns: 指定要保留的列列表，如果提供则跳过相关性分析

    Returns:
        处理后的训练集和测试集
    """
    train_df, test_df, sub = get_train_test()
    train, test = train_df.copy(), test_df.copy()
    logger.info(
        f"Train shape: {train.shape}, Test shape: {test.shape}, Sub shape: {sub.shape}"
    )

    # 处理训练集
    train = add_extra_data(train)
    logger.info(f"After adding extra data, Train shape: {train.shape}")

    train = clean_smiles(train)
    logger.info(f"After cleaning SMILES, Train shape: {train.shape}")

    train = filter_train_data(train)
    logger.info(f"After filtering, Train shape: {train.shape}")

    if use_string:
        train = add_smiles_string_features(train, smiles_col="SMILES")
        logger.info(f"After adding SMILES features, Train shape: {train.shape}")

    train = replace_all_R_with_C(train, smiles_col="SMILES")
    logger.info(f"After replacing R with C, Train shape: {train.shape}")

    if use_rdkit:
        train = add_rdkit_features(train, smiles_col="SMILES")
        logger.info(f"After adding RDKit features, Train shape: {train.shape}")

    if use_mordred:
        train = add_mordred_features(train, smiles_col="SMILES")
        logger.info(f"After adding Mordred features, Train shape: {train.shape}")

    target = ["Tg", "Tc", "Rg", "FFV", "Density"]

    train, dropped_cols = drop_high_missing_and_impute_median(
        train,
        threshold=0.5,
        exclude_cols=["id", "SMILES"] + target,
        drop=True,
        variance_threshold=0.01,
    )

    # 如果指定了columns，直接使用指定的列
    if columns is not None:
        logger.info(f"Using specified columns: {len(columns)} columns")
        # 检查指定的列是否都存在
        available_cols = [col for col in columns if col in train.columns]
        missing_cols = [col for col in columns if col not in train.columns]

        if missing_cols:
            logger.warning(f"Following columns not found in train data: {missing_cols}")

        train = train[available_cols].copy()
        logger.info(f"After selecting specified columns, Train shape: {train.shape}")
    else:
        # 原有的相关性分析逻辑
        train = remove_highly_correlated_features(
            train, threshold=0.95, exclude_cols=["id", "SMILES"] + target
        )
        logger.info(f"After removing highly correlated features, Train shape: {train.shape}")

    # 处理测试集
    if use_string:
        test = add_smiles_string_features(test, smiles_col="SMILES")
        logger.info(f"After adding SMILES features, Test shape: {test.shape}")

    test = replace_all_R_with_C(test, smiles_col="SMILES")
    logger.info(f"After replacing R with C, Test shape: {test.shape}")

    if use_rdkit:
        test = add_rdkit_features(test, smiles_col="SMILES")
        logger.info(f"After adding RDKit features, Test shape: {test.shape}")

    if use_mordred:
        test = add_mordred_features(test, smiles_col="SMILES")
        logger.info(f"After adding Mordred features, Test shape: {test.shape}")

    test, _ = drop_high_missing_and_impute_median(
        test, threshold=0.5, exclude_cols=["id", "SMILES"] + target, drop=False
    )
    # 对齐测试集特征与训练集
    test = align_test_features_with_train(test, train, target_cols=target)

    # 验证特征一致性
    train_features = [col for col in train.columns if col not in target + ["id"]]
    test_features = [col for col in test.columns if col not in ["id"]]

    if set(train_features) == set(test_features):
        logger.info("✓ Feature consistency check passed")
        logger.info(f"  Train features: {len(train_features)}")
        logger.info(f"  Test features: {len(test_features)}")
    else:
        logger.error("✗ Feature consistency check failed!")
        train_only = set(train_features) - set(test_features)
        test_only = set(test_features) - set(train_features)
        if train_only:
            logger.error(f"  Features only in train: {list(train_only)[:10]}...")
        if test_only:
            logger.error(f"  Features only in test: {list(test_only)[:10]}...")

    if save_files:
        # 要求调用者提供非空 save_tail；函数内部校验并防止覆盖已有文件
        if not save_tail or not str(save_tail).strip():
            raise ValueError(
                "save_files=True requires a non-empty save_tail (base filename)."
            )

        base = str(save_tail).strip()
        if base.lower().endswith(".csv"):
            base = base[:-4]

        path = "datasets/"
        os.makedirs(path, exist_ok=True)

        train_path = os.path.join(path, f"train_orig_{base}.csv")
        test_path = os.path.join(path, f"test_orig_{base}.csv")

        # 防止覆盖已存在文件
        if os.path.exists(train_path) or os.path.exists(test_path):
            raise FileExistsError(
                f"Target files already exist: {train_path} or {test_path}. Choose a different save_tail."
            )

        train.to_csv(train_path, index=False)
        test.to_csv(test_path, index=False)
        logger.info(f"Cleaned data saved to:\n  {train_path}\n  {test_path}")
        logger.info(
            f'[FINISH] Completed, data saved in "{path}" directory as "{base}_train.csv" and "{base}_test.csv"'
        )

    print_feature_statistics(train)
    return train, test


def create_datasets(
    train_df: pd.DataFrame = None,
    save_dir: str = "datasets/target_datasets/",
    csv_path: str = "datasets/cleaned_train.csv",
    targets: Union[str, List[str]] = None,
) -> Dict[str, pd.DataFrame]:
    """
    创建目标特异性数据集

    Args:
        train_df: 完整的训练数据（可选）
        save_dir: 保存目录
        csv_path: 如果train_df为None，从此路径读取数据
        targets: 目标列表，可以是单个目标或目标列表。如果为None，使用默认的所有目标

    Returns:
        每个目标的数据集字典
    """
    import os

    logger.info(f"[START] Creating target {targets} datasets...")
    os.makedirs(save_dir, exist_ok=True)

    # 如果没有提供train_df，从CSV读取
    if train_df is None:
        if os.path.exists(csv_path):
            logger.info(f"Loading train data from {csv_path}")
            train_df = pd.read_csv(csv_path)
        else:
            raise FileNotFoundError(
                f"CSV file not found at {csv_path}. Please provide train_df or ensure the CSV exists."
            )

    # 处理targets参数
    if targets is None:
        targets = ["Tg", "FFV", "Tc", "Density", "Rg"]
    elif isinstance(targets, str):
        targets = [targets]
    elif isinstance(targets, list) and len(targets) > 1:
        # 修改这里：明确指定创建多目标数据集的行为
        logger.info(f"Creating multi-target dataset for targets: {targets}")

        # 检查目标列是否存在
        missing_targets = [t for t in targets if t not in train_df.columns]
        if missing_targets:
            logger.warning(f"Targets {missing_targets} not found in DataFrame columns.")
            targets = [t for t in targets if t in train_df.columns]

        if not targets:
            logger.error("No valid targets found.")
            return {}

        # 统计每个目标的非空样本数
        for target in targets:
            non_null_count = train_df[target].notna().sum()
            logger.info(f"Target '{target}': {non_null_count} non-null samples")

        # 创建多目标数据集（只保留所有目标都有值的样本）
        multi_target_name = "_".join(targets)
        multi_target_mask = train_df[targets].notna().all(axis=1)
        multi_target_df = train_df[multi_target_mask].copy()

        logger.info(f"Samples with all targets {targets} non-null: {multi_target_mask.sum()}")

        target_datasets = {multi_target_name: multi_target_df}
        multi_target_df.to_csv(f"{save_dir}train_{multi_target_name}.csv", index=False)
        logger.info(
            f"Created multi-target dataset '{multi_target_name}' with {len(multi_target_df)} samples"
        )
        print_feature_statistics(multi_target_df)

        return target_datasets

    # 如果是单个目标或默认行为，创建独立的数据集
    target_datasets = {}

    for target in targets:
        # 检查目标列是否存在
        if target not in train_df.columns:
            logger.warning(
                f"Target '{target}' not found in DataFrame columns. Skipping."
            )
            continue

        # 选择有该目标值的行
        target_df = train_df[train_df[target].notna()].copy()
        # Remove other target columns so the resulting dataset contains only this target
        props = ["Tg", "Tc", "Rg", "FFV", "Density"]
        other_targets = [p for p in props if p != target and p in target_df.columns]
        if other_targets:
            target_df = target_df.drop(columns=other_targets)
            logger.info(f"Removed other target columns for '{target}': {other_targets}")

        # 保存
        target_datasets[target] = target_df
        target_df.to_csv(f"{save_dir}train_{target}.csv", index=False)
        logger.info(f"Created {target} dataset with {len(target_df)} samples")
        print_feature_statistics(target_df)

    return target_datasets


if __name__ == "__main__":
    """
        # 基于化学知识的标签特异性特征
    label_features = {
        'Tg': [
            'rdkit_MolLogP', 'rdkit_NumRotatableBonds', 'rdkit_TPSA',
            'rdkit_NumAromaticRings', 'rdkit_FractionCSP3', 'rdkit_BalabanJ',
            'rdkit_HallKierAlpha', 'rdkit_branching_index'
        ],
        'FFV': [
            'rdkit_MolWt', 'rdkit_TPSA', 'rdkit_NumRotatableBonds',
            'rdkit_MolMR', 'rdkit_LabuteASA', 'rdkit_molecular_compactness',
            'rdkit_NHOHCount', 'rdkit_NumHDonors'
        ],
        'Tc': [
            'rdkit_MolLogP', 'rdkit_MolWt', 'rdkit_NumHDonors', 'rdkit_NumHAcceptors',
            'rdkit_TPSA', 'rdkit_NumValenceElectrons'
        ],
        'Density': [
            'rdkit_MolWt', 'rdkit_MolMR', 'rdkit_FractionCSP3', 'rdkit_NumHeteroatoms',
            'rdkit_NumFluorine', 'rdkit_NumChlorine', 'rdkit_molecular_compactness'
        ],
        'Rg': [
            'rdkit_MolWt', 'rdkit_NumRotatableBonds', 'rdkit_BalabanJ',
            'rdkit_Kappa2', 'rdkit_branching_index', 'rdkit_HallKierAlpha'
        ]
    }

    """
    # PROPERTIES = ["Tg", "Tc", "Rg", "FFV", "Density"]
    # with open("datasets/used_columns.json", "r", encoding="utf-8") as f:
    #     columns = json.load(f)

    # train, test = process_train_test_data(
    #     use_mordred=True,
    #     use_rdkit=True,
    #     save_files=True,
    #     columns=columns,
    #     save_tail="testing1",
    # )

    # save_columns_to_json(train, name="used_columns")

    # # create target datasets (multi-target if multiple targets specified)

    # for pro in PROPERTIES:
    #     datasets = create_datasets(targets=pro, csv_path="datasets/train_orig_1.csv", save_dir="datasets/target_datasets/")
