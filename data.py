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
from torch_geometric.datasets import QM9
from torch_geometric.loader import DataLoader
from rdkit.Chem import Descriptors, rdMolDescriptors
from sklearn.model_selection import train_test_split
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


class PolymerFeatureExtractor:
    """Extract molecular features for traditional ML models"""

    def __init__(self, use_mordred=True, use_rdkit=True):
        self.use_mordred = use_mordred
        self.use_rdkit = use_rdkit
        self.mordred_calc = None
        if self.use_mordred:
            self.mordred_calc = Calculator(mordred_descriptors, ignore_3D=True)

    def extract_rdkit_descriptors(self, mol) -> Dict[str, float]:
        """Extract RDKit molecular descriptors"""
        if mol is None:
            return {}

        descriptors = {}

        # Basic descriptors
        descriptor_funcs = [
            ("MolWt", Descriptors.MolWt),
            ("MolLogP", Descriptors.MolLogP),
            ("TPSA", Descriptors.TPSA),
            ("NumRotatableBonds", Descriptors.NumRotatableBonds),
            ("NumHBD", Descriptors.NumHDonors),
            ("NumHBA", Descriptors.NumHAcceptors),
            ("NumAromaticRings", Descriptors.NumAromaticRings),
            ("NumSaturatedRings", Descriptors.NumSaturatedRings),
            ("NumHeteroatoms", Descriptors.NumHeteroatoms),
            ("BalabanJ", Descriptors.BalabanJ),
            ("Kappa1", Descriptors.Kappa1),
            ("Kappa2", Descriptors.Kappa2),
            ("Kappa3", Descriptors.Kappa3),
            ("LabuteASA", Descriptors.LabuteASA),
            ("FractionCSP3", Descriptors.FractionCSP3),
        ]

        for name, func in descriptor_funcs:
            try:
                descriptors[name] = float(func(mol))
            except:
                descriptors[name] = np.nan

        # Additional custom descriptors
        try:
            descriptors.update(
                {
                    "NumHeavyAtoms": mol.GetNumHeavyAtoms(),
                    "NumCarbons": sum(
                        1 for atom in mol.GetAtoms() if atom.GetAtomicNum() == 6
                    ),
                    "NumNitrogens": sum(
                        1 for atom in mol.GetAtoms() if atom.GetAtomicNum() == 7
                    ),
                    "NumOxygens": sum(
                        1 for atom in mol.GetAtoms() if atom.GetAtomicNum() == 8
                    ),
                    "NumFluorines": sum(
                        1 for atom in mol.GetAtoms() if atom.GetAtomicNum() == 9
                    ),
                    "NumSulfurs": sum(
                        1 for atom in mol.GetAtoms() if atom.GetAtomicNum() == 16
                    ),
                    "NumChlorines": sum(
                        1 for atom in mol.GetAtoms() if atom.GetAtomicNum() == 17
                    ),
                }
            )
        except:
            descriptors.update(
                {
                    "NumHeavyAtoms": np.nan,
                    "NumCarbons": np.nan,
                    "NumNitrogens": np.nan,
                    "NumOxygens": np.nan,
                    "NumFluorines": np.nan,
                    "NumSulfurs": np.nan,
                    "NumChlorines": np.nan,
                }
            )

        return descriptors

    def extract_mordred_descriptors(self, mol) -> Dict[str, float]:
        """Extract Mordred molecular descriptors"""
        if not self.mordred_calc or mol is None:
            return {}

        try:
            desc_dict = self.mordred_calc(mol)
            result = {}
            for key, value in desc_dict.items():
                try:
                    if pd.isna(value) or np.isinf(value):
                        result[str(key)] = np.nan
                    else:
                        result[str(key)] = float(value)
                except:
                    result[str(key)] = np.nan
            return result
        except:
            return {}

    def smiles_to_features(self, smiles_list: List[str]) -> pd.DataFrame:
        """Convert SMILES to feature matrix"""
        features_list = []

        for smiles in tqdm(smiles_list, desc="Extracting molecular features"):
            cleaned_smiles = make_smile_canonical(smiles) if smiles else None
            mol = Chem.MolFromSmiles(cleaned_smiles) if cleaned_smiles else None

            features = {}

            # RDKit descriptors
            if self.use_rdkit:
                rdkit_features = self.extract_rdkit_descriptors(mol)
                features.update(rdkit_features)

            # Mordred descriptors
            if self.use_mordred:
                mordred_features = self.extract_mordred_descriptors(mol)
                features.update(mordred_features)

            features_list.append(features)

        # Convert to DataFrame
        df = pd.DataFrame(features_list)

        # Handle missing values and infinities
        df = df.replace([np.inf, -np.inf], np.nan)

        # Drop columns with all NaN values
        df = df.dropna(axis=1, how="all")

        return df


def prepare_ml_dataset(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    targets: List[str],
    feature_extractor: PolymerFeatureExtractor = None,
    cache_train_path: str = "train_ml_features.pkl",
    cache_test_path: str = "test_ml_features.pkl",
    force_rebuild: bool = False,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Prepare feature matrices for traditional ML models

    Returns:
        train_features: DataFrame with features + targets + id/SMILES
        test_features: DataFrame with features + id/SMILES
    """
    if feature_extractor is None:
        feature_extractor = PolymerFeatureExtractor()

    # Extract training features
    if not force_rebuild and os.path.exists(cache_train_path):
        print(f"Loading cached training features from {cache_train_path}")
        train_features = pd.read_pickle(cache_train_path)
    else:
        print("Extracting training features...")
        train_mol_features = feature_extractor.smiles_to_features(
            train_df["SMILES"].tolist()
        )
        train_features = pd.concat(
            [
                train_df[["SMILES"] + targets].reset_index(drop=True),
                train_mol_features.reset_index(drop=True),
            ],
            axis=1,
        )
        train_features.to_pickle(cache_train_path)

    # Extract test features
    if not force_rebuild and os.path.exists(cache_test_path):
        print(f"Loading cached test features from {cache_test_path}")
        test_features = pd.read_pickle(cache_test_path)
    else:
        print("Extracting test features...")
        test_mol_features = feature_extractor.smiles_to_features(
            test_df["SMILES"].tolist()
        )
        test_features = pd.concat(
            [
                test_df[["id", "SMILES"]].reset_index(drop=True),
                test_mol_features.reset_index(drop=True),
            ],
            axis=1,
        )
        test_features.to_pickle(cache_test_path)

    print(f"Training features shape: {train_features.shape}")
    print(f"Test features shape: {test_features.shape}")

    return train_features, test_features
