import numpy as np
import pandas as pd
import pickle
import os
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
import warnings

# Machine Learning
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.feature_selection import SelectFromModel, VarianceThreshold
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
from xgboost import XGBRegressor

# Chemistry
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors
from rdkit.Chem.rdMolDescriptors import CalcNumRotatableBonds
try:
    from mordred import Calculator, descriptors as mordred_descriptors
    MORDRED_AVAILABLE = True
except ImportError:
    MORDRED_AVAILABLE = False
    print("Warning: Mordred not available. Molecular descriptors will be limited.")

# Utils
import joblib
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns


class XGBoostConfig:
    """Configuration class for XGBoost pipeline"""
    def __init__(self):
        # Data processing
        self.use_external_data = True
        self.use_augmentation = False
        self.random_state = 42
        self.test_size = 0.2
        
        # Feature engineering
        self.use_mordred = MORDRED_AVAILABLE
        self.use_rdkit_descriptors = True
        self.use_fingerprints = False  # Can be added later
        
        # Feature selection
        self.use_variance_threshold = True
        self.variance_threshold = 0.01
        self.use_correlation_filter = True
        self.correlation_threshold = 0.95
        self.use_feature_selection = True
        
        # Scaling
        self.use_robust_scaler = True  # Use RobustScaler instead of StandardScaler
        
        # Model parameters by task
        self.xgb_params = {
            'Tg': {
                'n_estimators': 3000,
                'max_depth': 5,
                'learning_rate': 0.01,
                'subsample': 0.8,
                'colsample_bytree': 1.0,
                'reg_lambda': 7.0,
                'gamma': 0.1,
                'objective': 'reg:absoluteerror',
                'eval_metric': 'mae',
                'early_stopping_rounds': 50,
                'random_state': 42
            },
            'FFV': {
                'n_estimators': 3000,
                'max_depth': 7,
                'learning_rate': 0.06,
                'subsample': 0.6,
                'colsample_bytree': 0.8,
                'reg_lambda': 2.0,
                'gamma': 0.0,
                'objective': 'reg:absoluteerror',
                'eval_metric': 'mae',
                'early_stopping_rounds': 50,
                'random_state': 42
            },
            'Tc': {
                'n_estimators': 3000,
                'max_depth': 4,
                'learning_rate': 0.01,
                'subsample': 0.6,
                'colsample_bytree': 0.8,
                'reg_lambda': 7.0,
                'gamma': 0.0,
                'objective': 'reg:absoluteerror',
                'eval_metric': 'mae',
                'early_stopping_rounds': 50,
                'random_state': 42
            },
            'Density': {
                'n_estimators': 3000,
                'max_depth': 5,
                'learning_rate': 0.06,
                'subsample': 0.8,
                'colsample_bytree': 1.0,
                'reg_lambda': 3.0,
                'gamma': 0.0,
                'objective': 'reg:absoluteerror',
                'eval_metric': 'mae',
                'early_stopping_rounds': 50,
                'random_state': 42
            },
            'Rg': {
                'n_estimators': 3000,
                'max_depth': 4,
                'learning_rate': 0.06,
                'subsample': 0.6,
                'colsample_bytree': 1.0,
                'reg_lambda': 10.0,
                'gamma': 0.1,
                'objective': 'reg:absoluteerror',
                'eval_metric': 'mae',
                'early_stopping_rounds': 50,
                'random_state': 42
            }
        }
        
        # Cross-validation
        self.n_folds = 5
        self.use_stratified_cv = True


class PolymerFeatureExtractor:
    """Extract molecular features for polymer property prediction"""
    
    def __init__(self, config: XGBoostConfig):
        self.config = config
        self.mordred_calc = None
        if config.use_mordred and MORDRED_AVAILABLE:
            self.mordred_calc = Calculator(mordred_descriptors, ignore_3D=True)
    
    def clean_smiles(self, smiles: str) -> Optional[str]:
        """Clean and canonicalize SMILES"""
        if not isinstance(smiles, str) or len(smiles) == 0:
            return None
        
        # Remove problematic polymer notation
        bad_patterns = ['[R]', '[R1]', '[R2]', '[R3]', '[R4]', '[R5]', 
                       "[R']", '[R"]', 'R1', 'R2', 'R3', 'R4', 'R5']
        
        for pattern in bad_patterns:
            if pattern in smiles:
                return None
        
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is not None:
                return Chem.MolToSmiles(mol, canonical=True)
            return None
        except:
            return None
    
    def extract_rdkit_descriptors(self, mol) -> Dict[str, float]:
        """Extract RDKit molecular descriptors"""
        descriptors = {}
        
        # Basic descriptors
        descriptor_funcs = [
            ('MolWt', Descriptors.MolWt),
            ('MolLogP', Descriptors.MolLogP),
            ('TPSA', Descriptors.TPSA),
            ('NumRotatableBonds', Descriptors.NumRotatableBonds),
            ('NumHBD', Descriptors.NumHDonors),
            ('NumHBA', Descriptors.NumHAcceptors),
            ('NumAromaticRings', Descriptors.NumAromaticRings),
            ('NumSaturatedRings', Descriptors.NumSaturatedRings),
            ('NumHeteroatoms', Descriptors.NumHeteroatoms),
            ('BalabanJ', Descriptors.BalabanJ),
            ('Kappa1', Descriptors.Kappa1),
            ('Kappa2', Descriptors.Kappa2),
            ('Kappa3', Descriptors.Kappa3),
            ('LabuteASA', Descriptors.LabuteASA),
            ('FractionCSP3', Descriptors.FractionCSP3),
        ]
        
        for name, func in descriptor_funcs:
            try:
                descriptors[name] = float(func(mol))
            except:
                descriptors[name] = np.nan
        
        # Additional custom descriptors
        try:
            descriptors['NumHeavyAtoms'] = mol.GetNumHeavyAtoms()
            descriptors['NumCarbons'] = sum(1 for atom in mol.GetAtoms() if atom.GetAtomicNum() == 6)
            descriptors['NumNitrogens'] = sum(1 for atom in mol.GetAtoms() if atom.GetAtomicNum() == 7)
            descriptors['NumOxygens'] = sum(1 for atom in mol.GetAtoms() if atom.GetAtomicNum() == 8)
            descriptors['NumFluorines'] = sum(1 for atom in mol.GetAtoms() if atom.GetAtomicNum() == 9)
            descriptors['NumSulfurs'] = sum(1 for atom in mol.GetAtoms() if atom.GetAtomicNum() == 16)
            descriptors['NumChlorines'] = sum(1 for atom in mol.GetAtoms() if atom.GetAtomicNum() == 17)
        except:
            descriptors.update({
                'NumHeavyAtoms': np.nan, 'NumCarbons': np.nan, 'NumNitrogens': np.nan,
                'NumOxygens': np.nan, 'NumFluorines': np.nan, 'NumSulfurs': np.nan,
                'NumChlorines': np.nan
            })
        
        return descriptors
    
    def extract_mordred_descriptors(self, mol) -> Dict[str, float]:
        """Extract Mordred molecular descriptors"""
        if not self.mordred_calc:
            return {}
        
        try:
            desc_dict = self.mordred_calc(mol)
            # Convert to regular dict and handle missing values
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
        
        for smiles in tqdm(smiles_list, desc="Extracting features"):
            mol = Chem.MolFromSmiles(smiles) if smiles else None
            
            if mol is None:
                features_list.append({})
                continue
            
            features = {}
            
            # RDKit descriptors
            if self.config.use_rdkit_descriptors:
                rdkit_features = self.extract_rdkit_descriptors(mol)
                features.update(rdkit_features)
            
            # Mordred descriptors
            if self.config.use_mordred:
                mordred_features = self.extract_mordred_descriptors(mol)
                features.update(mordred_features)
            
            features_list.append(features)
        
        # Convert to DataFrame
        df = pd.DataFrame(features_list)
        
        # Handle missing values and infinities
        df = df.replace([np.inf, -np.inf], np.nan)
        
        # Drop columns with all NaN values
        df = df.dropna(axis=1, how='all')
        
        return df


class XGBoostPipeline:
    """Complete XGBoost pipeline for polymer property prediction"""
    
    def __init__(self, config: Optional[XGBoostConfig] = None):
        self.config = config or XGBoostConfig()
        self.feature_extractor = PolymerFeatureExtractor(self.config)
        self.scalers = {}
        self.feature_selectors = {}
        self.models = {}
        self.cv_results = {}
        
        # Create output directory
        os.makedirs("models/xgboost", exist_ok=True)
        os.makedirs("results", exist_ok=True)
    
    def load_data(self, train_path: str, test_path: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Load and clean training and test data"""
        print("Loading data...")
        
        train_df = pd.read_csv(train_path)
        test_df = pd.read_csv(test_path)
        
        print(f"Loaded {len(train_df)} training samples and {len(test_df)} test samples")
        
        # Clean SMILES
        train_df['SMILES'] = train_df['SMILES'].apply(self.feature_extractor.clean_smiles)
        test_df['SMILES'] = test_df['SMILES'].apply(self.feature_extractor.clean_smiles)
        
        # Remove invalid SMILES
        train_df = train_df[train_df['SMILES'].notna()].reset_index(drop=True)
        test_df = test_df[test_df['SMILES'].notna()].reset_index(drop=True)
        
        print(f"After cleaning: {len(train_df)} training samples and {len(test_df)} test samples")
        
        return train_df, test_df
    
    def prepare_features(self, smiles_list: List[str], fit: bool = False) -> pd.DataFrame:
        """Extract and process features from SMILES"""
        # Extract molecular features
        features_df = self.feature_extractor.smiles_to_features(smiles_list)
        
        if len(features_df) == 0:
            return features_df
        
        # Handle missing values
        numeric_features = features_df.select_dtypes(include=[np.number])
        
        if fit:
            # Fit preprocessing steps
            
            # Variance threshold
            if self.config.use_variance_threshold:
                self.variance_selector = VarianceThreshold(threshold=self.config.variance_threshold)
                numeric_features = pd.DataFrame(
                    self.variance_selector.fit_transform(numeric_features),
                    columns=numeric_features.columns[self.variance_selector.get_support()],
                    index=numeric_features.index
                )
                print(f"Variance threshold removed {len(features_df.columns) - len(numeric_features.columns)} features")
            
            # Correlation filter
            if self.config.use_correlation_filter:
                corr_matrix = numeric_features.corr().abs()
                upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
                high_corr_features = [column for column in upper_tri.columns if any(upper_tri[column] > self.config.correlation_threshold)]
                numeric_features = numeric_features.drop(columns=high_corr_features)
                print(f"Correlation filter removed {len(high_corr_features)} features")
            
            # Fill missing values with median
            self.feature_medians = numeric_features.median()
            numeric_features = numeric_features.fillna(self.feature_medians)
            
            # Fit scaler
            if self.config.use_robust_scaler:
                self.feature_scaler = RobustScaler()
            else:
                self.feature_scaler = StandardScaler()
            
            scaled_features = self.feature_scaler.fit_transform(numeric_features)
            features_df = pd.DataFrame(scaled_features, columns=numeric_features.columns, index=numeric_features.index)
        
        else:
            # Transform using fitted preprocessors
            if self.config.use_variance_threshold and hasattr(self, 'variance_selector'):
                numeric_features = pd.DataFrame(
                    self.variance_selector.transform(numeric_features),
                    columns=numeric_features.columns[self.variance_selector.get_support()],
                    index=numeric_features.index
                )
            
            if self.config.use_correlation_filter:
                # Use the same columns as training
                available_cols = [col for col in self.feature_scaler.feature_names_in_ if col in numeric_features.columns]
                numeric_features = numeric_features[available_cols]
            
            # Fill missing values
            if hasattr(self, 'feature_medians'):
                numeric_features = numeric_features.fillna(self.feature_medians)
            
            # Scale features
            if hasattr(self, 'feature_scaler'):
                scaled_features = self.feature_scaler.transform(numeric_features)
                features_df = pd.DataFrame(scaled_features, columns=numeric_features.columns, index=numeric_features.index)
        
        return features_df
    
    def train_target(self, X: pd.DataFrame, y: pd.Series, target: str) -> Dict:
        """Train XGBoost model for a specific target"""
        print(f"\nTraining XGBoost for {target}...")
        
        # Get target-specific parameters
        params = self.config.xgb_params.get(target, self.config.xgb_params['Tg'])
        
        # Set up cross-validation
        if self.config.use_stratified_cv:
            # Create bins for stratification
            y_bins = pd.qcut(y, q=min(self.config.n_folds, len(y.unique())), 
                           labels=False, duplicates='drop')
            kf = StratifiedKFold(n_splits=self.config.n_folds, shuffle=True, 
                               random_state=self.config.random_state)
            splits = list(kf.split(X, y_bins))
        else:
            kf = KFold(n_splits=self.config.n_folds, shuffle=True, 
                      random_state=self.config.random_state)
            splits = list(kf.split(X, y))
        
        # Store results
        fold_scores = []
        models = []
        
        # Cross-validation loop
        for fold, (train_idx, val_idx) in enumerate(splits):
            print(f"  Fold {fold + 1}/{self.config.n_folds}")
            
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
            
            # Train model
            model = XGBRegressor(**params, n_jobs=-1, verbosity=0)
            model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                verbose=False
            )
            
            # Validate
            y_pred = model.predict(X_val)
            mae = mean_absolute_error(y_val, y_pred)
            rmse = np.sqrt(mean_squared_error(y_val, y_pred))
            r2 = r2_score(y_val, y_pred)
            
            fold_scores.append({'mae': mae, 'rmse': rmse, 'r2': r2})
            models.append(model)
            
            print(f"    MAE: {mae:.4f}, RMSE: {rmse:.4f}, R²: {r2:.4f}")
        
        # Calculate average scores
        avg_scores = {
            'mae': np.mean([s['mae'] for s in fold_scores]),
            'rmse': np.mean([s['rmse'] for s in fold_scores]),
            'r2': np.mean([s['r2'] for s in fold_scores]),
            'mae_std': np.std([s['mae'] for s in fold_scores]),
            'rmse_std': np.std([s['rmse'] for s in fold_scores]),
            'r2_std': np.std([s['r2'] for s in fold_scores])
        }
        
        print(f"  Average - MAE: {avg_scores['mae']:.4f}±{avg_scores['mae_std']:.4f}, "
              f"RMSE: {avg_scores['rmse']:.4f}±{avg_scores['rmse_std']:.4f}, "
              f"R²: {avg_scores['r2']:.4f}±{avg_scores['r2_std']:.4f}")
        
        # Feature selection based on ensemble
        if self.config.use_feature_selection:
            feature_importance = np.mean([model.feature_importances_ for model in models], axis=0)
            selector = SelectFromModel(
                RandomForestRegressor(n_estimators=100, random_state=self.config.random_state),
                threshold='median',
                prefit=False
            )
            # Fit on full data with feature importance as weights
            dummy_rf = RandomForestRegressor(n_estimators=10, random_state=self.config.random_state)
            dummy_rf.fit(X, y)
            dummy_rf.feature_importances_ = feature_importance
            selector.estimator_ = dummy_rf
            selector.threshold_ = np.median(feature_importance)
            
            selected_features = X.columns[feature_importance >= selector.threshold_]
            print(f"  Selected {len(selected_features)}/{len(X.columns)} features")
            
            # Retrain models with selected features
            if len(selected_features) < len(X.columns):
                X_selected = X[selected_features]
                models = []
                
                for fold, (train_idx, val_idx) in enumerate(splits):
                    X_train, X_val = X_selected.iloc[train_idx], X_selected.iloc[val_idx]
                    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
                    
                    model = XGBRegressor(**params, n_jobs=-1, verbosity=0)
                    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
                    models.append(model)
                
                self.feature_selectors[target] = selected_features
        
        return {
            'models': models,
            'scores': avg_scores,
            'fold_scores': fold_scores,
            'params': params
        }
    
    def train_all_targets(self, train_df: pd.DataFrame, targets: List[str]):
        """Train models for all targets"""
        print("Extracting features from training data...")
        X = self.prepare_features(train_df['SMILES'].tolist(), fit=True)
        
        print(f"Feature matrix shape: {X.shape}")
        
        for target in targets:
            if target not in train_df.columns:
                print(f"Warning: Target {target} not found in training data")
                continue
            
            # Get valid samples for this target
            valid_mask = train_df[target].notna()
            if valid_mask.sum() < 10:
                print(f"Warning: Not enough valid samples for {target} ({valid_mask.sum()})")
                continue
            
            X_target = X[valid_mask].copy()
            y_target = train_df[target][valid_mask].copy()
            
            # Train model
            result = self.train_target(X_target, y_target, target)
            self.models[target] = result['models']
            self.cv_results[target] = result
        
        # Save models and preprocessors
        self.save_models()
    
    def predict(self, test_df: pd.DataFrame, targets: List[str]) -> pd.DataFrame:
        """Make predictions on test data"""
        print("Extracting features from test data...")
        X_test = self.prepare_features(test_df['SMILES'].tolist(), fit=False)
        
        predictions = {'id': test_df['id']}
        
        for target in targets:
            if target not in self.models:
                print(f"Warning: No model found for {target}")
                predictions[target] = np.nan
                continue
            
            X_target = X_test.copy()
            
            # Apply feature selection if used
            if target in self.feature_selectors:
                available_features = [f for f in self.feature_selectors[target] if f in X_target.columns]
                X_target = X_target[available_features]
            
            # Ensemble prediction
            ensemble_preds = []
            for model in self.models[target]:
                pred = model.predict(X_target)
                ensemble_preds.append(pred)
            
            # Average predictions
            predictions[target] = np.mean(ensemble_preds, axis=0)
        
        return pd.DataFrame(predictions)
    
    def save_models(self):
        """Save trained models and preprocessors"""
        # Save main objects
        joblib.dump(self.feature_scaler, "models/xgboost/feature_scaler.pkl")
        joblib.dump(self.feature_medians, "models/xgboost/feature_medians.pkl")
        
        if hasattr(self, 'variance_selector'):
            joblib.dump(self.variance_selector, "models/xgboost/variance_selector.pkl")
        
        # Save models for each target
        for target, models in self.models.items():
            target_dir = f"models/xgboost/{target}"
            os.makedirs(target_dir, exist_ok=True)
            
            for i, model in enumerate(models):
                joblib.dump(model, f"{target_dir}/model_fold_{i}.pkl")
            
            if target in self.feature_selectors:
                joblib.dump(self.feature_selectors[target], f"{target_dir}/selected_features.pkl")
        
        # Save CV results
        pd.DataFrame({
            target: {
                'mae_mean': results['scores']['mae'],
                'mae_std': results['scores']['mae_std'],
                'rmse_mean': results['scores']['rmse'],
                'rmse_std': results['scores']['rmse_std'],
                'r2_mean': results['scores']['r2'],
                'r2_std': results['scores']['r2_std']
            }
            for target, results in self.cv_results.items()
        }).T.to_csv("results/xgboost_cv_results.csv")
        
        print("Models and results saved!")
    
    def load_models(self):
        """Load trained models and preprocessors"""
        self.feature_scaler = joblib.load("models/xgboost/feature_scaler.pkl")
        self.feature_medians = joblib.load("models/xgboost/feature_medians.pkl")
        
        if os.path.exists("models/xgboost/variance_selector.pkl"):
            self.variance_selector = joblib.load("models/xgboost/variance_selector.pkl")
        
        # Load models for each target
        self.models = {}
        self.feature_selectors = {}
        
        for target_dir in os.listdir("models/xgboost"):
            if os.path.isdir(f"models/xgboost/{target_dir}") and target_dir not in ['__pycache__']:
                target = target_dir
                models = []
                
                # Load all fold models
                fold_files = [f for f in os.listdir(f"models/xgboost/{target}") if f.startswith('model_fold_')]
                fold_files.sort()
                
                for fold_file in fold_files:
                    model = joblib.load(f"models/xgboost/{target}/{fold_file}")
                    models.append(model)
                
                self.models[target] = models
                
                # Load feature selection if exists
                if os.path.exists(f"models/xgboost/{target}/selected_features.pkl"):
                    self.feature_selectors[target] = joblib.load(f"models/xgboost/{target}/selected_features.pkl")
        
        print(f"Loaded models for targets: {list(self.models.keys())}")
    
    def plot_results(self):
        """Plot cross-validation results"""
        if not self.cv_results:
            print("No results to plot")
            return
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        targets = list(self.cv_results.keys())
        metrics = ['mae', 'rmse', 'r2']
        metric_names = ['MAE', 'RMSE', 'R²']
        
        for i, (metric, metric_name) in enumerate(zip(metrics, metric_names)):
            means = [self.cv_results[target]['scores'][metric] for target in targets]
            stds = [self.cv_results[target]['scores'][f'{metric}_std'] for target in targets]
            
            axes[i].bar(targets, means, yerr=stds, capsize=5, alpha=0.7)
            axes[i].set_title(f'Cross-Validation {metric_name}')
            axes[i].set_ylabel(metric_name)
            axes[i].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig("results/xgboost_cv_results.png", dpi=300, bbox_inches='tight')
        plt.show()


def main():
    """Main pipeline execution"""
    # Configuration
    config = XGBoostConfig()
    
    # Initialize pipeline
    pipeline = XGBoostPipeline(config)
    
    # Define paths (adjust these to your data)
    train_path = "train.csv"  # Your training data
    test_path = "test.csv"    # Your test data
    targets = ['Tg', 'FFV', 'Tc', 'Density', 'Rg']
    
    # Check if models exist
    if os.path.exists("models/xgboost/feature_scaler.pkl"):
        print("Loading existing models...")
        pipeline.load_models()
        
        # Load test data
        _, test_df = pipeline.load_data(train_path, test_path)
        
        # Make predictions
        predictions = pipeline.predict(test_df, targets)
        predictions.to_csv("xgboost_predictions.csv", index=False)
        print("Predictions saved to xgboost_predictions.csv")
    
    else:
        print("Training new models...")
        
        # Load data
        train_df, test_df = pipeline.load_data(train_path, test_path)
        
        # Train models
        pipeline.train_all_targets(train_df, targets)
        
        # Make predictions
        predictions = pipeline.predict(test_df, targets)
        predictions.to_csv("xgboost_predictions.csv", index=False)
        print("Predictions saved to xgboost_predictions.csv")
        
        # Plot results
        pipeline.plot_results()


if __name__ == "__main__":
    main()