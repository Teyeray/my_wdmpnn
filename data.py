import torch
import os, re
import numpy as np
import pandas as pd
from rdkit import Chem
from torch import Tensor
from pathlib import Path
from collections import Counter
from rdkit.Chem import rdPartialCharges
from typing import List, Tuple, Optional, Dict, Union, Iterable

#Preliminary processing of the raw data
def get_data_paths(print_paths: bool = False) -> dict:
    BASE_PATH = Path("kaggle/input/neurips-open-polymer-prediction-2025")
    EXTRA_BASE = Path("kaggle/input/smiles-extra-data")
    TC_BASE = Path("kaggle/input/tc-smiles")

    paths = {
        "train_csv": Path(
            BASE_PATH / "train.csv"
        ),

        "test_csv": Path(BASE_PATH / "test.csv"),

        "sample_submission": Path(
            BASE_PATH / "sample_submission.csv"
        ),

        "tc_smiles": Path(TC_BASE / "Tc_SMILES.csv"),

        "sed_bigsmiles": Path(EXTRA_BASE / "JCIM_sup_bigsmiles.csv"),

        "sed_tg3": Path(EXTRA_BASE / "data_tg3.xlsx"),

        "sed_dnst1": Path(EXTRA_BASE / "data_dnst1.xlsx"),

        "dataset4": Path(
            BASE_PATH / "train_supplement" / "dataset4.csv"
        ),

        "dataset1": Path(
            BASE_PATH / "train_supplement" / "dataset1.csv"
        ),

        "dataset2": Path(
            BASE_PATH / "train_supplement" / "dataset2.csv"
        ),

        "dataset3": Path(
            BASE_PATH / "train_supplement" / "dataset3.csv"
        ),
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
        print(f'---' * 20)
        print(f"[START] Adding extra data targeting '{source_name}' to train target '{target}'.")
    else:
        if not Path(extra).exists():
            print(f"[ERROR] Extra data file '{extra}' does not exist. Returning original train data.")
            return train
        df_extra = pd.read_csv(extra).rename(columns={source_name: target}).copy()
        print(f'---' * 20)
        print(f"[START] Adding extra data from '{extra}' targeting '{source_name}' to train target '{target}'.")

    print(f"[INFO] Train data shape: {train.shape}")
    df_train = train.copy()

    print(f"[INFO] Extra data shape: {df_extra.shape}, columns: {df_extra.columns.tolist()}")
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
                print(f"[ADD] Updated target for SMILES '{smi}' and value '{extra_value}' from extra data target '{target}'.")
            else:
                # Otherwise, drop the SMILES from extra
                df_extra.drop(df_extra[df_extra["SMILES"] == smi].index, inplace=True)
                #print(f"[INFO] Dropped SMILES '{smi}' from extra data.")

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

def add_extra_data(
    train: pd.DataFrame
    )-> pd.DataFrame:
    P = get_data_paths()
    # read extra data
    train = train.copy()

    train = combine_data(train, P['tc_smiles'], target='Tc', source_name='TC_mean')

    train = combine_data(train,P['sed_bigsmiles'], target='Tg', source_name='Tg (C)')

    tg_excel_data = pd.read_excel(P['sed_tg3']).assign(Tg_K=lambda df: df['Tg [K]'] - 273.15)
    train = combine_data(train, tg_excel_data, target='Tg', source_name='Tg_K')

    train = combine_data(train, P['dataset3'], target='Tg', source_name='dataset3')

    density_data = pd.read_excel(P['sed_dnst1']).rename(columns={'density(g/cm3)':'Density'}).assign(Density=lambda df: pd.to_numeric(df['Density'], errors='coerce') - 0.118)
    train = combine_data(train,density_data,target='Density', source_name='Density')

    train = combine_data(train, P['dataset1'], target='Tc', source_name='TC_mean')

    train = combine_data(train, P['dataset4'], target='FFV', source_name='FFV')

    return train

def clean_smiles(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["SMILES"] = df["SMILES"].apply(make_smile_canonical)
    df = df.dropna(subset=["SMILES"]).reset_index(drop=True)

    gb = df.groupby("SMILES", as_index=False)
    print(f"[INFO] We have {len(df)} entries, {len(gb)} are unique SMILES.")
    df = gb.mean(numeric_only=True)
    print(f"[INFO] After grouping by SMILES and averaging, we have {len(df)} entries.")
    return df

def _filter_dataset(df: pd.DataFrame, column: str, lower_bound: float, upper_bound: float) -> pd.DataFrame:
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame.")
    print(f"[INFO] Original data shape: {df.shape}; non-null '{column}': {df[column].notna().sum()}")
    #Keep rows that are within the range, or those where the column is missing (skip filtering if missing)
    mask_in_range = (df[column] >= lower_bound) & (df[column] <= upper_bound)
    mask_keep = df[column].isna() | mask_in_range
    filtered_df = df[mask_keep].copy()
    dropped = len(df) - len(filtered_df)
    print(f"[INFO] Dropped {dropped} rows from '{column}' (kept NaN and values in [{lower_bound}, {upper_bound}]).")
    print(f"[INFO] Filtered data shape: {filtered_df.shape}")
    return filtered_df

def filter_train_data(df: pd.DataFrame) -> pd.DataFrame:
    #filter according to the specified ranges, keeping NaNs
    tg_filtered = _filter_dataset(df, 'Tg', -223, 480)
    tc_filtered = _filter_dataset(tg_filtered, 'Tc', 0, 0.54)
    rg_filtered = _filter_dataset(tc_filtered, 'Rg', 0, 33)
    ffv_filtered = _filter_dataset(rg_filtered, 'FFV', 0.1, 0.6)
    de_filtered = _filter_dataset(ffv_filtered, 'Density', 0, 1.76)
    return de_filtered

def replace_all_R_with_C(smi: str) -> str:
    #Replace any R placeholders with C:
    #  - bracketed R-groups like [R], [R'], [R1] -> C
    #  - any remaining uppercase 'R' anywhere -> C
    #Does not touch lowercase letters (e.g. 'r') or other characters.
    s = str(smi)
    # 1) replace bracketed R-groups
    s = re.sub(r'\[R[^\]]*\]', 'C', s)
    # 2) replace any remaining uppercase R
    s = re.sub(r'R', 'C', s)
    return s

def count_smiles_symbols(smiles_iter: Iterable[str]) -> Dict[str, int]:
    """
    Count atom element symbols appearing in an iterable of SMILES.
    - smiles_iter: iterable of SMILES strings (can be a pandas Series)
    - returns: dict symbol -> count
    Invalid SMILES are skipped.
    """
    cnt = Counter()
    for smi in smiles_iter:
        if smi is None:
            continue
        try:
            smi = str(smi)
        except Exception:
            continue
        if smi == "" or smi.lower() == "nan":
            continue
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        for a in mol.GetAtoms():
            try:
                sym = a.GetSymbol()
            except Exception:
                continue
            cnt[sym] += 1
    return dict(cnt)

def count_symbols_in_df(df: pd.DataFrame, col: str = "SMILES") -> Dict[str, int]:
    """
    Convenience wrapper: count symbols in DataFrame column `col`.
    """
    if col not in df.columns:
        raise ValueError(f"Column '{col}' not found in DataFrame.")
    return count_smiles_symbols(df[col].dropna().astype(str))

def z_to_group_period(z: int) -> Tuple[int, int, bool]:
    """
    返回: (period, group, is_fblock)
    group 为 1..18；f 区元素（Ce-Lu、Th-Lr）返回 is_fblock=True，group=None
    规则按照 IUPAC 长式周期表直觉定位；La/Ac 置于 group=3。
    """
    if z < 1 or z > 118:
        raise ValueError("Z must be in [1,118]")
    # period
    if z <= 2:            period = 1
    elif z <= 10:         period = 2
    elif z <= 18:         period = 3
    elif z <= 36:         period = 4
    elif z <= 54:         period = 5
    elif z <= 86:         period = 6
    else:                 period = 7

    group, is_f = None, False
    if period == 1:
        group = 1 if z == 1 else 18
    elif period == 2:
        # Li..Ne 的族序列
        group = [1,2,13,14,15,16,17,18][z - 3]
    elif period == 3:
        group = [1,2,13,14,15,16,17,18][z - 11]
    elif period == 4:
        group = (z - 19) + 1           # 19..36 -> 1..18
    elif period == 5:
        group = (z - 37) + 1
    elif period == 6:
        if z == 55: group = 1          # Cs
        elif z == 56: group = 2        # Ba
        elif z == 57: group = 3        # La
        elif 58 <= z <= 71:
            is_f = True                # Ce..Lu
        elif 72 <= z <= 80:
            group = z - 68             # 72..80 -> 4..12
        elif 81 <= z <= 86:
            group = z - 68             # 81..86 -> 13..18
    elif period == 7:
        if z == 87: group = 1          # Fr
        elif z == 88: group = 2        # Ra
        elif z == 89: group = 3        # Ac
        elif 90 <= z <= 103:
            is_f = True                # Th..Lr
        elif 104 <= z <= 112:
            group = z - 100            # 104..112 -> 4..12
        elif 113 <= z <= 118:
            group = z - 100            # 113..118 -> 13..18
    return period, group, is_f

def z_to_grid_xy(z: int) -> Tuple[float, float]:
    """
    网格坐标: x=group, y=period.
    - 正常块: (x, y) = (group, period)
    - f 区:   放在“下挂行”，x=3.5..16.5，y=8(镧系)或9(锕系)
    可按需做归一化：(x-1)/17, (y-1)/6
    """
    period, group, is_f = z_to_group_period(z)
    if not is_f:
        x = float(group)
        y = float(period)
    else:
        # f-block 连续铺开到 3.5..16.5，保持与长式视觉列对应
        if 58 <= z <= 71:
            x = 3.5 + (z - 58)         # Ce..Lu -> 3.5..16.5
            y = 8.0                    # 镧系“下挂”
        elif 90 <= z <= 103:
            x = 3.5 + (z - 90)         # Th..Lr -> 3.5..16.5
            y = 9.0                    # 锕系“下挂”
        else:
            # 理论上不会到这里
            x, y = 3.5, 8.0
    return x, y

def periodic_distance_to_C(z: int) -> Tuple[float, float]:
    """
    返回 (raw_distance, normalized_distance)：
      - raw_distance: 在 z_to_grid_xy 网格上的欧氏距离 (x,y) 到 Carbon(Z=6)
      - normalized_distance: 除以理论最大距离，归一化到 ~[0,1]
    """
    cx, cy = z_to_grid_xy(6)
    x, y = z_to_grid_xy(z)
    raw = math.hypot(x - cx, y - cy)
    # 估计最大可能距离（网格 x 最小=1, 最大≈16.5; y 最小=1, 最大≈9）
    max_x = 16.5
    max_y = 9.0
    max_dist = math.hypot(max_x - 1.0, max_y - 1.0)
    norm = raw / max_dist if max_dist > 0 else 0.0
    return float(raw), float(norm)

#Generate data for WdMPNN model training
def make_node_features(atom) -> List[float]:
    try:
        mass = float(atom.GetMass())
        atom_map_num = float(atom.GetAtomMapNum())
        is_aromatic = float(int(atom.GetIsAromatic()))
        formal_charge = float(atom.GetFormalCharge())
        atomic_num = float(atom.GetAtomicNum())
        chiral_tag = float(int(atom.GetChiralTag()))
        # RDKit HybridizationType 可以直接 int()，作为可重复编码
        try:
            hybridization = float(int(atom.GetHybridization()))
        except Exception:
            hybridization = 0.0
        degree = float(atom.GetDegree())
        total_h = float(atom.GetTotalNumHs())
        is_in_ring = float(int(atom.IsInRing()))
        mass = float(atom.GetMass())
        # Gasteiger 电荷（如果之前计算过）
        try:
            gcharge = float(atom.GetProp("_GasteigerCharge"))
        except Exception:
            gcharge = 0.0
        if atomic_num:
            _, distance_norm = periodic_distance_to_C(atomic_num)
        else:
            distance_norm = 0.0
        return [
            mass,
            atom_map_num,
            is_aromatic,
            formal_charge,
            atomic_num,
            chiral_tag,
            hybridization,
            degree,
            total_h,
            is_in_ring,
            mass,
            gcharge,
            distance_norm,
        ]
    except Exception as e:
        # 保证函数不会返回非数值，调用者可捕获异常
        raise RuntimeError(f"make_node_features failed: {e}")

if __name__ == "__main__":
    train, test, sub = get_train_test()
    train_added = add_extra_data(train)
    train_added.rename(columns={'SMILES':'SMILES_raw'}, inplace=True)
    train_added['SMILES'] = train_added['SMILES_raw'].apply(replace_all_R_with_C)
    train_cleaned = clean_smiles(train_added)
    train_filtered = filter_train_data(train_cleaned)