import math
import torch
import pickle
import os, re
import numpy as np
import pandas as pd
from rdkit import Chem
from torch import Tensor
from pathlib import Path
from collections import Counter
from torch.utils.data import Dataset
from torch_geometric.data import Data
from torch_geometric.datasets import QM9
from torch_geometric.loader import DataLoader
from sklearn.model_selection import train_test_split
from typing import List, Tuple, Optional, Dict, Union, Iterable


def load_qm9(batch_size=64, num_workers=0, root="kaggle/input/my-qm9/qm9"):
    dataset = QM9(root=root)
    idx = list(range(len(dataset)))
    train_idx, val_idx = train_test_split(idx, test_size=0.1, random_state=42)

    train_ds = dataset[train_idx]
    val_ds = dataset[val_idx]

    #train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    #val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    QM9_TASKS = [
        "mu", "alpha", "homo", "lumo", "gap", "r2", "zpve",
        "U0", "U", "H", "G", "Cv",
        "u0_atom", "u_atom", "h_atom", "g_atom",
        "A", "B", "C",
    ]

    # 转换成 DataFrame 方便统计 n_dict / r_dict
    y = dataset._data.y.numpy()
    df = pd.DataFrame(y, columns=QM9_TASKS)

    return train_ds, val_ds, df, QM9_TASKS, dataset
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


def clean_smiles(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["SMILES"] = df["SMILES"].apply(make_smile_canonical)
    df = df.dropna(subset=["SMILES"]).reset_index(drop=True)

    gb = df.groupby("SMILES", as_index=False)
    print(f"[INFO] We have {len(df)} entries, {len(gb)} are unique SMILES.")
    df = gb.mean(numeric_only=True)
    print(f"[INFO] After grouping by SMILES and averaging, we have {len(df)} entries.")
    return df


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
    if z <= 2:
        period = 1
    elif z <= 10:
        period = 2
    elif z <= 18:
        period = 3
    elif z <= 36:
        period = 4
    elif z <= 54:
        period = 5
    elif z <= 86:
        period = 6
    else:
        period = 7

    group, is_f = None, False
    if period == 1:
        group = 1 if z == 1 else 18
    elif period == 2:
        # Li..Ne 的族序列
        group = [1, 2, 13, 14, 15, 16, 17, 18][z - 3]
    elif period == 3:
        group = [1, 2, 13, 14, 15, 16, 17, 18][z - 11]
    elif period == 4:
        group = (z - 19) + 1  # 19..36 -> 1..18
    elif period == 5:
        group = (z - 37) + 1
    elif period == 6:
        if z == 55:
            group = 1  # Cs
        elif z == 56:
            group = 2  # Ba
        elif z == 57:
            group = 3  # La
        elif 58 <= z <= 71:
            is_f = True  # Ce..Lu
        elif 72 <= z <= 80:
            group = z - 68  # 72..80 -> 4..12
        elif 81 <= z <= 86:
            group = z - 68  # 81..86 -> 13..18
    elif period == 7:
        if z == 87:
            group = 1  # Fr
        elif z == 88:
            group = 2  # Ra
        elif z == 89:
            group = 3  # Ac
        elif 90 <= z <= 103:
            is_f = True  # Th..Lr
        elif 104 <= z <= 112:
            group = z - 100  # 104..112 -> 4..12
        elif 113 <= z <= 118:
            group = z - 100  # 113..118 -> 13..18
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
            x = 3.5 + (z - 58)  # Ce..Lu -> 3.5..16.5
            y = 8.0  # 镧系“下挂”
        elif 90 <= z <= 103:
            x = 3.5 + (z - 90)  # Th..Lr -> 3.5..16.5
            y = 9.0  # 锕系“下挂”
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


# 你的数据频率
element_freq = {
    "*": 18588,
    "C": 251669,
    "N": 16027,
    "O": 33799,
    "F": 6666,
    "S": 2059,
    "Cl": 564,
    "Si": 631,
    "Na": 7,
    "H": 112,
    "P": 323,
    "Br": 255,
    "Ge": 5,
    "Se": 7,
    "Sn": 7,
    "I": 13,
    "Cd": 1,
    "B": 2,
    "Te": 1,
    "Ca": 1,
}


def freq_normalized_atomic_num(atom):
    """
    基于元素频率归一化原子序数
    输入: rdkit Atom 对象
    输出: 归一化后的值 (float)，频率越低权重越高
    """
    # 直接获取元素符号
    element_symbol = atom.GetSymbol()
    atomic_num = atom.GetAtomicNum()
    # 特殊处理星号元素和原子序数为0的情况
    if element_symbol == "*" or atomic_num == 0:
        # 星号元素通常表示未知或通配符原子
        # 给予一个中间值，或者根据频率调整
        freq = element_freq.get("*", 1)
        weight = 1 / np.log(freq + 1)
        return 0.5 * weight  # 返回中间值并加权

    # 获取频率，如果不在字典中，频率为1（最稀有）
    freq = element_freq.get(element_symbol, 1)

    # 基础归一化：原子序数缩放到[0,1]
    base_norm = (atomic_num - 1) / (53 - 1)  # 基于I=53

    # 频率权重：频率越低，权重越高
    weight = 1 / np.log(freq + 1)
    # print(f"Element: {element_symbol}, base_norm:{base_norm}, weighted_norm:{base_norm*weight}")
    return base_norm * weight


# Generate data for WdMPNN model training
def make_node_features(atom) -> List[float]:
    """
    从 rdkit Atom 生成数值特征向量，确保所有输出为 float 且 hybridization 做 one-hot。
    输出顺序：
      [mass, atom_map_num, is_aromatic, formal_charge, atomic_num, chiral_tag,
       hyb_0, ..., hyb_8, degree, total_h, is_in_ring, distance_norm]
    """
    try:
        # 为每个特征计算添加独立的异常处理
        try:
            atomic_mass = float(atom.GetMass())
            mass = (atomic_mass - 1.008) / (126.90 - 1.008)
        except Exception:
            mass = "wrongmass"

        is_aromatic = 1.0 if bool(atom.GetIsAromatic()) else 0.0

        try:
            formal_charge = float(atom.GetFormalCharge())
        except Exception:
            formal_charge = "wrongformal_charge"

        try:
            chiral_tag = float(int(atom.GetChiralTag()))
        except Exception:
            chiral_tag = "wrongchiral_tag"

        # Hybridization one-hot over a fixed ordered list
        try:
            hyb = atom.GetHybridization()
        except Exception:
            hyb = None
        hyb_types = [
            Chem.rdchem.HybridizationType.UNSPECIFIED,
            Chem.rdchem.HybridizationType.OTHER,
            Chem.rdchem.HybridizationType.S,
            Chem.rdchem.HybridizationType.SP,
            Chem.rdchem.HybridizationType.SP2,
            Chem.rdchem.HybridizationType.SP2D,
            Chem.rdchem.HybridizationType.SP3,
            Chem.rdchem.HybridizationType.SP3D,
            Chem.rdchem.HybridizationType.SP3D2,
        ]

        # 为 hybridization one-hot 也添加异常处理
        try:
            hybridization_oh = [1.0 if hyb == t else 0.0 for t in hyb_types]
        except Exception:
            hybridization_oh = ["wronghybridization"] * len(hyb_types)

        try:
            degree = float(atom.GetDegree())
        except Exception:
            degree = "wrongdegree"

        try:
            total_h = float(atom.GetTotalNumHs())
        except Exception:
            total_h = "wrongtotal_h"

        is_in_ring = 1.0 if atom.IsInRing() else 0.0

        # Use integer atomic number for periodic_distance_to_C

        try:
            atomic_num_int = int(atom.GetAtomicNum())
            if atomic_num_int > 0 and not isinstance(atomic_num_int, str):
                _, distance_norm = periodic_distance_to_C(atomic_num_int)
                distance_norm = float(distance_norm)
            else:
                distance_norm = 0.0
        except Exception:
            distance_norm = "wrongdistance_norm"

        try:
            # 直接传入atom对象进行归一化
            atomic_num_norm = freq_normalized_atomic_num(atom)
        except Exception:
            atomic_num_norm = "wrongatomic_num_norm"

        feats: List[float] = [
            mass,
            is_aromatic,
            formal_charge,
            # atomic_num,
            chiral_tag,
        ]
        feats += hybridization_oh
        feats += [
            degree,
            total_h,
            is_in_ring,
            distance_norm,
            atomic_num_norm,
        ]

        # 检查是否有错误特征，如果有则抛出详细异常
        error_features = []
        for i, feat in enumerate(feats):
            if isinstance(feat, str) and feat.startswith("wrong"):
                error_features.append((i, feat))

        if error_features:
            error_msg = f"Features with errors: {error_features}"
            raise ValueError(error_msg)

        # 确保所有条目都是浮点数
        feats = [float(x) for x in feats]
        return feats

    except Exception as e:
        raise RuntimeError(f"make_node_features failed: {e}")


def make_edge_features(bond) -> List[float]:
    """
    从 rdkit Bond 构造数值特征向量（返回 List[float]）。
    包含（顺序）:
      - bond_type_as_double (float)
      - is_aromatic (0/1)
      - is_conjugated (0/1)
      - is_in_ring (0/1)
      - stereo (int -> float)
      - valence_contrib (float, 若不可用则 0.0)
      - begin_atomic_num (int -> float)
      - end_atomic_num (int -> float)
      - bond_idx (int -> float)
    函数尽量容错，遇到异常时使用默认值。
    """
    try:
        bt = float(bond.GetBondTypeAsDouble())
    except Exception:
        bt = "wrong" + "bond_type_as_double"

    try:
        is_aromatic = float(int(bond.GetIsAromatic()))
        if is_aromatic:
            is_aromatic = 1.0
        else:
            is_aromatic = 0.0
    except Exception:
        is_aromatic = "wrong" + "is_aromatic"

    try:
        is_conjugated = float(int(bond.GetIsConjugated()))
    except Exception:
        is_conjugated = "wrong" + "is_conjugated"

    try:
        is_in_ring = float(int(bond.IsInRing()))
    except Exception:
        is_in_ring = "wrong" + "is_in_ring"

    try:
        ba = bond.GetBeginAtom()
        ea = bond.GetEndAtom()
        begin_z = float(ba.GetAtomicNum()) if ba is not None else 0.0
        end_z = float(ea.GetAtomicNum()) if ea is not None else 0.0
    except Exception:
        begin_z = "wrong" + "begin_z"
        end_z = "wrong" + "end_z"

    try:
        bond_idx = float(bond.GetIdx())
    except Exception:
        bond_idx = "wrong" + "bond_idx"
    try:
        bond_stereo = float(bond.GetStereo())
    except Exception:
        bond_stereo = "wrong" + "bond_stereo"
    return [
        bt,
        is_aromatic,
        is_conjugated,
        is_in_ring,
        begin_z,
        end_z,
        bond_idx,
        bond_stereo,
    ]


def _mol_to_pyg_data(mol: Chem.Mol, smiles: str, target=None) -> Optional[Data]:
    # convert rdkit Mol -> torch_geometric Data object

    # node features: dim=18
    node_feats: List[list] = []
    for atom in mol.GetAtoms():
        node_feats.append(make_node_features(atom))
    x = (
        torch.tensor(node_feats, dtype=torch.float32)
        if node_feats
        else torch.empty((0, 18), dtype=torch.float32)
    )

    # edge features: dim=8
    src = []
    dst = []
    edge_attrs = []

    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        ef = make_edge_features(bond)

        # add both directions
        src += [i, j]
        dst += [j, i]
        edge_attrs += [ef, ef]
    if len(src) == 0:
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.empty((0, 8), dtype=torch.float32)
    else:
        edge_index = torch.tensor([src, dst], dtype=torch.long)
        edge_attr = torch.tensor(edge_attrs, dtype=torch.float32)

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    data.smiles = str(smiles)
    # --- 支持 multi-target: scalar 或 iterable -> 1D float tensor ---
    if target is not None:
        try:
            arr = np.atleast_1d(np.asarray(target, dtype=float))
            data.y = torch.tensor(arr, dtype=torch.float32).view(1, -1)
        except Exception:
            data.y = torch.full((1, 1), float("nan"), dtype=torch.float32)
    return data


class PygGraphDataset(Dataset):
    """
    Simple PyG-compatible dataset that yields torch_geometric.data.Data objects.
    - Accepts an iterable of SMILES and optional targets (same length).
    - Optional caching via cache_path (pickle).
    Usage:
      ds = PygGraphDataset(smiles_list, targets=targets_list, cache_path='cache.pkl')
      from torch_geometric.loader import DataLoader
      loader = DataLoader(ds, batch_size=32, shuffle=True)
    """

    def __init__(
        self,
        smiles: Iterable[str],
        targets: Optional[Iterable] = None,
        cache_path: Optional[str] = None,
        force_rebuild: bool = False,
    ):
        self.smiles = list(smiles)
        self.targets = (
            list(targets) if targets is not None else [None] * len(self.smiles)
        )
        self.cache_path = cache_path

        if cache_path and os.path.exists(cache_path) and not force_rebuild:
            with open(cache_path, "rb") as f:
                self.data_list = pickle.load(f)
        else:
            self.data_list: List[Data] = []
            for s, y in zip(self.smiles, self.targets):
                try:
                    mol = Chem.MolFromSmiles(str(s)) if s is not None else None
                except Exception:
                    mol = None
                data = _mol_to_pyg_data(mol, s, y)
                if data is None:
                    # skip invalid SMILES
                    continue
                self.data_list.append(data)
            if cache_path:
                with open(cache_path, "wb") as f:
                    pickle.dump(self.data_list, f)

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        return self.data_list[idx]


def build_pyg_dataset(
    smiles: Iterable[str],
    targets: Optional[Iterable] = None,
    cache_path: Optional[str] = None,
    force_rebuild: bool = False,
):
    """
    Convenience builder.
    """
    return PygGraphDataset(
        smiles, targets=targets, cache_path=cache_path, force_rebuild=force_rebuild
    )

def load_polymer(
    tasks: List[str] = None,
    batch_size: int = 64,
    split_ratio: float = 0.9,
    cache_path: str = "train_polymer.pkl",
    force_rebuild: bool = False,
) -> Tuple[DataLoader, DataLoader, pd.DataFrame, List[str]]:
    """
    加载并预处理 Polymer 数据，返回 DataLoader。

    Args:
        tasks (List[str], optional): 要预测的目标属性。
            默认 ["Tg", "FFV", "Tc", "Density", "Rg"]。
        batch_size (int, optional): DataLoader 的 batch size，默认 64。
        split_ratio (float, optional): 训练集划分比例 (0~1)，默认 0.9。
        cache_path (str, optional): 数据缓存路径，默认 "train_polymer.pkl"。
        force_rebuild (bool, optional): 是否强制重建缓存，默认 False。

    Returns:
        train_loader (DataLoader): 训练集 loader
        val_loader (DataLoader): 验证集 loader
        train_filtered (pd.DataFrame): 清洗后的 DataFrame
        tasks (List[str]): 实际使用的任务列表
    """
    if tasks is None:
        tasks = ["Tg", "FFV", "Tc", "Density", "Rg"]

    # === 1) 读取并清洗 DataFrame ===
    train, _, _ = get_train_test()
    train = add_extra_data(train)
    train.rename(columns={"SMILES": "SMILES_raw"}, inplace=True)
    train["SMILES"] = train["SMILES_raw"].apply(replace_all_R_with_C)
    train = clean_smiles(train)
    train_filtered = filter_train_data(train)

    # === 2) 构建 PyG dataset ===
    ds = build_pyg_dataset(
        train_filtered["SMILES"],
        targets=train_filtered[tasks].values,
        cache_path=cache_path,
        force_rebuild=force_rebuild,
    )

    # === 3) 划分 train/val ===
    train_size = int(split_ratio * len(ds))
    val_size = len(ds) - train_size
    train_ds, val_ds = torch.utils.data.random_split(ds, [train_size, val_size])

    # === 4) DataLoader ===
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, train_filtered, tasks

# 统计
def attribute_stats_from_smiles(
    smiles_iter: Iterable[str],
    feature_source: str = "node",  # "node" or "edge"
    attribute: Union[str, int] = "mass",
    return_per_molecule: bool = False,
) -> Dict:
    """
    对一组 SMILES 统计单一属性（来自 make_node_features 或 make_edge_features）。
    - smiles_iter: iterable of SMILES strings
    - feature_source: "node" or "edge"
    - attribute: 属性名（str）或索引（int）。支持以下内置名字：
        node: "mass","is_aromatic","formal_charge","atomic_num","chiral_tag",
              "hyb_UNSPECIFIED","hyb_OTHER","hyb_S","hyb_SP","hyb_SP2","hyb_SP2D","hyb_SP3","hyb_SP3D","hyb_SP3D2",
              "degree","total_h","is_in_ring","distance_norm","atomic_num_norm"
        edge: "bond_type","is_aromatic","is_conjugated","is_in_ring",
              "begin_z","end_z","bond_idx","bond_stereo"
    - return_per_molecule: 若 True 返回每个分子的详细值列表（mean + 原始值列表）
    返回字典 keys: total_smiles, processed_smiles, skipped_smiles, total_items,
                    mean, std, missing_count, per_molecule (optional)
    """
    # name -> index映射
    node_names = [
        "mass",
        "is_aromatic",
        "formal_charge",
        "atomic_num",
        "chiral_tag",
        "hyb_UNSPECIFIED",
        "hyb_OTHER",
        "hyb_S",
        "hyb_SP",
        "hyb_SP2",
        "hyb_SP2D",
        "hyb_SP3",
        "hyb_SP3D",
        "hyb_SP3D2",
        "degree",
        "total_h",
        "is_in_ring",
        "distance_norm",
        "atomic_num_norm",
    ]
    edge_names = [
        "bond_type",
        "is_aromatic",
        "is_conjugated",
        "is_in_ring",
        "begin_z",
        "end_z",
        "bond_idx",
        "bond_stereo",
    ]

    if feature_source not in ("node", "edge"):
        raise ValueError("feature_source must be 'node' or 'edge'")

    # resolve attribute -> index
    if isinstance(attribute, int):
        attr_idx = int(attribute)
    else:
        name = str(attribute)
        if feature_source == "node":
            if name not in node_names:
                raise KeyError(f"Unknown node attribute '{name}'")
            attr_idx = node_names.index(name)
        else:
            if name not in edge_names:
                raise KeyError(f"Unknown edge attribute '{name}'")
            attr_idx = edge_names.index(name)

    smiles_list = list(smiles_iter)
    processed = 0
    skipped = 0
    total_items = 0
    missing_count = 0
    all_vals = []
    per_molecule = []

    for smi in smiles_list:
        if smi is None:
            skipped += 1
            continue
        try:
            s = str(smi)
        except Exception:
            skipped += 1
            continue
        mol = Chem.MolFromSmiles(s)
        if mol is None:
            skipped += 1
            continue
        processed += 1

        vals = []
        try:
            # nodes
            if feature_source == "node":
                for atom in mol.GetAtoms():
                    try:
                        feats = make_node_features(atom)
                        # guard index
                        if attr_idx < 0 or attr_idx >= len(feats):
                            raise IndexError(
                                "attribute index out of range for node features"
                            )
                        v = float(feats[attr_idx])
                    except Exception:
                        v = float("nan")
                        missing_count += 1
                    vals.append(v)
            # edges
            else:
                for bond in mol.GetBonds():
                    try:
                        feats = make_edge_features(bond)
                        if attr_idx < 0 or attr_idx >= len(feats):
                            raise IndexError(
                                "attribute index out of range for edge features"
                            )
                        v = float(feats[attr_idx])
                    except Exception:
                        v = float("nan")
                        missing_count += 1
                    vals.append(v)
        except Exception:
            # unexpected per-molecule error -> skip molecule
            skipped += 1
            continue

        # aggregate per-molecule
        arr = np.array(vals, dtype=float) if vals else np.array([], dtype=float)
        mean_val = float(np.nanmean(arr)) if arr.size > 0 else float("nan")
        std_val = float(np.nanstd(arr)) if arr.size > 0 else float("nan")

        per_molecule.append(
            {
                "smiles": s,
                "num_items": int(arr.size),
                "mean": mean_val,
                "std": std_val,
                "values": arr.tolist(),
            }
        )

        total_items += int(arr.size)
        # accumulate global values
        if arr.size > 0:
            # extend all_vals with arr (keeping NaNs to compute nanmean/std)
            all_vals.append(arr)

    # flatten all_vals to one array
    if all_vals:
        flat = np.concatenate(all_vals).astype(float)
        global_mean = float(np.nanmean(flat)) if flat.size > 0 else float("nan")
        global_std = float(np.nanstd(flat)) if flat.size > 0 else float("nan")
    else:
        flat = np.array([], dtype=float)
        global_mean = float("nan")
        global_std = float("nan")

    result = {
        "total_smiles": len(smiles_list),
        "processed_smiles": processed,
        "skipped_smiles": skipped,
        "total_items": int(total_items),
        "mean": global_mean,
        "std": global_std,
        "missing_count": int(missing_count),
    }
    if return_per_molecule:
        result["per_molecule"] = per_molecule
    return result


if __name__ == "__main__":
    train, test, sub = get_train_test()
    train_added = add_extra_data(train)
    train_added.rename(columns={"SMILES": "SMILES_raw"}, inplace=True)
    train_added["SMILES"] = train_added["SMILES_raw"].apply(replace_all_R_with_C)
    train_cleaned = clean_smiles(train_added)
    train_filtered = filter_train_data(train_cleaned)
    ds = build_pyg_dataset(
        train_filtered["SMILES"],
        targets=train_filtered[["Tg","Tc","Density"]].values,  # 假设多目标
        cache_path="train_multi.pkl",
    )
    print("Dataset size:", len(ds))
    for i in range(3):
        d = ds[i]
        print("SMILES:", d.smiles)
        print("x shape (nodes × feats):", d.x.shape)
        print("edge_index shape:", d.edge_index.shape)
        print("edge_attr shape:", d.edge_attr.shape)
        print("y:", d.y)
        print("="*40)
    ys = torch.stack([d.y for d in ds])
    print("y shape:", ys.shape)

    # 每列缺失情况
    nan_per_col = torch.isnan(ys).sum(dim=0)
    print("Missing count per target:", nan_per_col.tolist())
