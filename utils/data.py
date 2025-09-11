import math
import torch
import pickle
import os, re
import numpy as np
import pandas as pd
from tqdm import tqdm
from rdkit import Chem
from torch import Tensor
from pathlib import Path
from collections import Counter
from torch.utils.data import Dataset
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from rdkit.ML.Descriptors import MoleculeDescriptors
from sklearn.model_selection import train_test_split
from rdkit.Chem import Descriptors, rdMolDescriptors, AllChem
from rdkit.Chem.rdMolDescriptors import CalcNumRotatableBonds
from typing import List, Tuple, Optional, Dict, Union, Iterable
from mordred import Calculator, descriptors as mordred_descriptors


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
                print(f"{key} found at {path} [OK]")
        else:
            print(f"{key} not found at {path} [ERROR]")
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
        print(f"---" * 20)
        print(
            f"[START] Adding extra data targeting '{source_name}' to train target '{target}'."
        )
    else:
        if not Path(extra).exists():
            print(
                f"[ERROR] Extra data file '{extra}' does not exist. Returning original train data."
            )
            return train
        df_extra = pd.read_csv(extra).rename(columns={source_name: target}).copy()
        print(f"---" * 20)
        print(
            f"[START] Adding extra data from '{extra}' targeting '{source_name}' to train target '{target}'."
        )

    print(f"[INFO] Train data shape: {train.shape}")
    df_train = train.copy()

    print(
        f"[INFO] Extra data shape: {df_extra.shape}, columns: {df_extra.columns.tolist()}"
    )
    df_extra = df_extra[["SMILES", target]].dropna(subset=["SMILES", target])
    df_extra = df_extra.groupby("SMILES", as_index=False).mean()

    # Find common SMILES
    common = set(df_train["SMILES"]) & set(df_extra["SMILES"])
    print(f"[INFO] Found {len(common)} overlapping SMILES.")

    if common:
        for smi in common:
            # Check if train's target is NaN
            train_target_value = df_train.loc[df_train["SMILES"] == smi, target]
            if train_target_value.isna().all():
                # If train's target is NaN, use extra's value
                extra_value = df_extra.loc[df_extra["SMILES"] == smi, target].values[0]
                df_train.loc[df_train["SMILES"] == smi, target] = extra_value
                print(
                    f"[ADD] Updated target for SMILES '{smi}' and value '{extra_value}' from extra data target '{target}'."
                )
            else:
                # Otherwise, drop the SMILES from extra
                df_extra.drop(df_extra[df_extra["SMILES"] == smi].index, inplace=True)
                # print(f"[INFO] Dropped SMILES '{smi}' from extra data.")

    # Merge train and extra
    df_combined = pd.concat([df_train, df_extra], ignore_index=True)
    print(f"[INFO] Combined data shape: {df_combined.shape}")
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
    feats['smiles_length'] = len(smiles)
    feats['capital_letters'] = sum(c.isupper() for c in smiles)
    feats['lowercase_letters'] = sum(c.islower() for c in smiles)
    feats['digits'] = sum(c.isdigit() for c in smiles)

    # 符号统计
    feats['parentheses'] = smiles.count('(') + smiles.count(')')
    feats['brackets'] = smiles.count('[') + smiles.count(']')
    feats['braces'] = smiles.count('{') + smiles.count('}')
    feats['equals'] = smiles.count('=')
    feats['hashes'] = smiles.count('#')
    feats['colons'] = smiles.count(':')
    feats['ats'] = smiles.count('@')
    feats['slashes'] = smiles.count('/') + smiles.count('\\')
    feats['plus_minus'] = smiles.count('+') + smiles.count('-')

    # 元素计数
    feats['C_count'] = smiles.count('C') + smiles.count('c')
    feats['O_count'] = smiles.count('O') + smiles.count('o')
    feats['N_count'] = smiles.count('N') + smiles.count('n')
    feats['S_count'] = smiles.count('S') + smiles.count('s')
    feats['P_count'] = smiles.count('P') + smiles.count('p')
    feats['F_count'] = smiles.count('F') + smiles.count('f')
    feats['Cl_count'] = smiles.count('Cl') + smiles.count('cl')
    feats['Br_count'] = smiles.count('Br') + smiles.count('br')
    feats['I_count'] = smiles.count('I') + smiles.count('i')

    # 结构模式
    feats['has_ring'] = int(any(d in smiles for d in '123456789'))
    feats['has_double_bond'] = int('=' in smiles)
    feats['has_triple_bond'] = int('#' in smiles)
    feats['has_aromatic'] = int(any(c in smiles for c in 'cnos'))

    # 元素比例
    feats['O_to_C_ratio'] = feats['O_count'] / (feats['C_count'] + 1e-5)
    feats['N_to_C_ratio'] = feats['N_count'] / (feats['C_count'] + 1e-5)
    feats['heteroatom_ratio'] = (
        feats['O_count'] + feats['N_count'] + feats['S_count'] + feats['P_count']
    ) / (feats['C_count'] + 1e-5)

    # 占位符特征
    feats['star_count'] = smiles.count('*')
    feats['R_placeholder_count'] = len(re.findall(r"\[R[0-9']*\]", smiles))
    feats['any_placeholder'] = int((feats['star_count'] > 0) or (feats['R_placeholder_count'] > 0))

    return feats

def add_smiles_string_features(df: pd.DataFrame, smiles_col: str = 'SMILES') -> pd.DataFrame:
    """Add SMILES string-based features to DataFrame"""
    df = df.copy()
    
    print(f"Adding SMILES string features to {len(df)} molecules...")
    
    smiles_features = []
    for smiles in tqdm(df[smiles_col], desc="Extracting SMILES string features"):
        features = _compute_all_string_features(smiles)
        # Add prefix to distinguish from other features
        features = {f'smiles_string_{k}': v for k, v in features.items()}
        smiles_features.append(features)
    
    # Convert to DataFrame and merge
    smiles_df = pd.DataFrame(smiles_features)
    result_df = pd.concat([df.reset_index(drop=True), smiles_df.reset_index(drop=True)], axis=1)
    
    print(f"Added {len(smiles_df.columns)} SMILES string features")
    return result_df

def clean_smiles(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["SMILES"] = df["SMILES"].apply(make_smile_canonical)
    df = df.dropna(subset=["SMILES"]).reset_index(drop=True)

    gb = df.groupby("SMILES", as_index=False)
    print(f"[INFO] We have {len(df)} entries, {len(gb)} are unique SMILES.")
    df = gb.mean(numeric_only=True)
    print(f"[INFO] After grouping by SMILES and averaging, we have {len(df)} entries.")
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
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame.")
    print(
        f"[INFO] Original data shape: {df.shape}; non-null '{column}': {df[column].notna().sum()}"
    )
    # Keep rows that are within the range, or those where the column is missing (skip filtering if missing)
    mask_in_range = (df[column] >= lower_bound) & (df[column] <= upper_bound)
    mask_keep = df[column].isna() | mask_in_range
    filtered_df = df[mask_keep].copy()
    dropped = len(df) - len(filtered_df)
    print(
        f"[INFO] Dropped {dropped} rows from '{column}' (kept NaN and values in [{lower_bound}, {upper_bound}])."
    )
    print(f"[INFO] Filtered data shape: {filtered_df.shape}")
    return filtered_df


def filter_train_data(df: pd.DataFrame) -> pd.DataFrame:
    # filter according to the specified ranges, keeping NaNs
    tg_filtered = _filter_dataset(df, "Tg", -223, 480)
    tc_filtered = _filter_dataset(tg_filtered, "Tc", 0, 0.54)
    rg_filtered = _filter_dataset(tc_filtered, "Rg", 0, 33)
    ffv_filtered = _filter_dataset(rg_filtered, "FFV", 0.1, 0.6)
    de_filtered = _filter_dataset(ffv_filtered, "Density", 0, 1.76)
    return de_filtered


def replace_all_R_with_C(smi: str) -> str:
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

def remove_highly_correlated_features(df: pd.DataFrame, threshold: float = 0.95, 
                                    exclude_cols: List[str] = None) -> pd.DataFrame:
    """Remove one of each pair of highly correlated features"""
    if exclude_cols is None:
        exclude_cols = ['SMILES', 'id']
    
    df = df.copy()
    
    # Get feature columns (exclude non-feature columns)
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    
    # Select only numeric columns
    numeric_cols = df[feature_cols].select_dtypes(include=[np.number]).columns.tolist()
    
    if len(numeric_cols) < 2:
        print("Not enough numeric features for correlation analysis")
        return df
    
    print(f"Analyzing correlations for {len(numeric_cols)} numeric features...")
    
    # Calculate correlation matrix
    corr_matrix = df[numeric_cols].corr().abs()
    
    # Find pairs of highly correlated features
    high_corr_pairs = []
    for i in range(len(corr_matrix.columns)):
        for j in range(i+1, len(corr_matrix.columns)):
            if corr_matrix.iloc[i, j] >= threshold:
                col1 = corr_matrix.columns[i]
                col2 = corr_matrix.columns[j]
                high_corr_pairs.append((col1, col2, corr_matrix.iloc[i, j]))
    
    print(f"Found {len(high_corr_pairs)} highly correlated pairs (correlation >= {threshold})")
    
    # Decide which features to remove
    features_to_remove = set()
    
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
    
    # Remove the features
    if features_to_remove:
        df = df.drop(columns=list(features_to_remove))
        print(f"Removed {len(features_to_remove)} highly correlated features")
    else:
        print("No features removed")
    
    return df

def load_feature_cache(cache_path):
    if os.path.exists(cache_path):
        with open(cache_path, 'rb') as f:
            return pickle.load(f)
    return {}

def save_feature_cache(cache_path, cache_dict):
    with open(cache_path, 'wb') as f:
        pickle.dump(cache_dict, f)

def add_rdkit_features(df, smiles_col='SMILES', cache_path='dataset/cache/rdkit_features.pkl'):
    desc_list_names = [d[0] for d in Descriptors._descList]
    fp_morgan_cols = [f'mfp_{i}' for i in range(1024)]
    feature_columns = [f'rdkit_{name}' for name in desc_list_names] + fp_morgan_cols

    cache = load_feature_cache(cache_path)
    new_cache = {}
    features_list = []

    for smiles in tqdm(df[smiles_col], desc="计算RDKit特征"):
        if pd.isna(smiles):
            features = np.full(len(feature_columns), np.nan)
        elif smiles in cache:
            features = cache[smiles]
        else:
            features = _generate_rdkit_features(smiles)
            new_cache[smiles] = features
        features_list.append(features)

    # 更新缓存
    cache.update(new_cache)
    save_feature_cache(cache_path, cache)

    features_df = pd.DataFrame(features_list, columns=feature_columns)
    features_df.replace([np.inf, -np.inf], np.nan, inplace=True)
    features_df.fillna(features_df.mean(), inplace=True)
    result_df = pd.concat([df.reset_index(drop=True), features_df.reset_index(drop=True)], axis=1)
    print(f"添加了 {len(feature_columns)} 个RDKit特征")
    return result_df

def align_test_features_with_train(test_df: pd.DataFrame, train_df: pd.DataFrame, 
                                  target_cols: List[str], id_col: str = 'id') -> pd.DataFrame:
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
        print(f"Adding missing columns to test set: {missing_cols}")
        for col in missing_cols:
            test_df[col] = np.nan
    
    # 检查多余的列
    extra_cols = [col for col in test_df.columns if col not in expected_test_cols]
    if extra_cols:
        print(f"Removing extra columns from test set: {extra_cols}")
        test_df = test_df.drop(columns=extra_cols)
    
    # 确保列的顺序一致
    test_df = test_df[expected_test_cols]
    
    print(f"Test features aligned: {test_df.shape}")
    return test_df


def process_train_test_data(
        save_files: bool = False,
        use_mordred: bool = True,
        use_rdkit: bool = True,
        ) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    完整的训练和测试数据处理流程，确保特征一致性
    
    Returns:
        处理后的训练集和测试集
    """
    train, test, sub = get_train_test()
    print(f"Train shape: {train.shape}, Test shape: {test.shape}, Sub shape: {sub.shape}")

    # 处理训练集
    train = add_extra_data(train)
    print(f"After adding extra data, Train shape: {train.shape}")

    train = clean_smiles(train)
    print(f"After cleaning SMILES, Train shape: {train.shape}")

    train = filter_train_data(train)
    print(f"After filtering, Train shape: {train.shape}")

    train = add_smiles_string_features(train, smiles_col='SMILES')
    print(f"After adding SMILES features, Train shape: {train.shape}")

    # 添加RDKit特征
    if use_rdkit:
        train = add_rdkit_features(train, smiles_col='SMILES')
        print(f"After adding RDKit features, Train shape: {train.shape}")

    target = ['Tg', 'Tc', 'Rg', 'FFV', 'Density']
    train = remove_highly_correlated_features(train, threshold=0.95, exclude_cols=['id','SMILES'] + target)
    print(f"After removing highly correlated features, Train shape: {train.shape}")

    # 处理测试集（基础处理）
    test = test.copy()
    test["SMILES"] = test["SMILES"].apply(make_smile_canonical)
    test = test.dropna(subset=["SMILES"]).reset_index(drop=True)
    print(f"After cleaning SMILES, Test shape: {test.shape}")
    
    test = add_smiles_string_features(test, smiles_col='SMILES')
    print(f"After adding SMILES features, Test shape: {test.shape}")

    # 添加RDKit特征到测试集
    if use_rdkit:
        test = add_rdkit_features(test, smiles_col='SMILES')
        print(f"After adding RDKit features, Test shape: {test.shape}")

    # 对齐测试集特征与训练集
    test = align_test_features_with_train(test, train, target_cols=target)
    
    # 验证特征一致性
    train_features = [col for col in train.columns if col not in target + ['id']]
    test_features = [col for col in test.columns if col not in ['id']]
    
    if set(train_features) == set(test_features):
        print("✓ Feature consistency check passed")
        print(f"  Train features: {len(train_features)}")
        print(f"  Test features: {len(test_features)}")
    else:
        print("✗ Feature consistency check failed!")
        train_only = set(train_features) - set(test_features)
        test_only = set(test_features) - set(train_features)
        if train_only:
            print(f"  Features only in train: {list(train_only)[:]}...")
        if test_only:
            print(f"  Features only in test: {list(test_only)[:]}...")

    if save_files:
        path = 'datasets/'
        if not os.path.exists(path):
            os.makedirs(path)
        train.to_csv(f"{path}cleaned_train.csv", index=False)
        test.to_csv(f"{path}cleaned_test.csv", index=False)
        print("Cleaned data saved to CSV files.")

    return train, test

def create_target_specific_datasets(train_df: pd.DataFrame, save_dir: str = "target_datasets/") -> Dict[str, pd.DataFrame]:
    """
    为每个目标变量创建单独的数据集（模仿参考代码的方法）
    
    Args:
        train_df: 完整的训练数据
        save_dir: 保存目录
        
    Returns:
        每个目标的数据集字典
    """
    import os
    os.makedirs(save_dir, exist_ok=True)
    
    targets = ['Tg', 'FFV', 'Tc', 'Density', 'Rg']
    target_datasets = {}
    
    for target in targets:
        # 选择有该目标值的行
        target_df = train_df[train_df[target].notna()].copy()
        
        # 保存
        target_datasets[target] = target_df
        target_df.to_csv(f"{save_dir}train_{target}.csv", index=False)
        print(f"Created {target} dataset with {len(target_df)} samples")
    
    return target_datasets

if __name__ == "__main__":
    '''
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
    
    '''
    train, test = process_train_test_data(save_files=True)