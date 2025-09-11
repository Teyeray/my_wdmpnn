import os
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from typing import Dict, List, Optional
from torch.optim.lr_scheduler import CosineAnnealingLR

# ------------------ wMAE Loss ------------------
class WMAELoss(torch.nn.Module):
    def __init__(self, tasks: List[str], n_dict: Dict[str, int], r_dict: Dict[str, float]):
        super().__init__()
        self.tasks = tasks
        self.weights = self._compute_task_weights(n_dict, r_dict, tasks)

    def _compute_task_weights(self, n_dict, r_dict, tasks):
        K = len(tasks)
        inv_sqrt_ns = [(1.0 / np.sqrt(max(n_dict[t], 1))) for t in tasks]
        denom = sum(inv_sqrt_ns)
        weights = {}
        for i, t in enumerate(tasks):
            ri = max(r_dict[t], 1e-12)
            wi = (1.0 / ri) * (K * inv_sqrt_ns[i] / denom)
            weights[t] = wi
        return weights

    def forward(self, outputs, targets, return_per_task=False):
        total = None
        count = 0
        per_task = {}
        for t in self.tasks:
            yhat, y = outputs[t], targets[t]
            mask = torch.isfinite(y)   # 只在非 NaN 样本上算
            if mask.sum() == 0:
                continue
            err = (yhat[mask] - y[mask]).abs().mean()
            weighted = self.weights[t] * err
            total = weighted if total is None else total + weighted
            count += 1
            per_task[t] = err.item()

        if total is None:
            loss = torch.tensor(0.0, requires_grad=True)
        else:
            loss = total / max(count, 1)

        if return_per_task:
            return loss, per_task
        return loss


def compute_task_stats(df: pd.DataFrame, tasks: List[str]):
    """
    计算 n_dict 和 r_dict
    - n_dict[t]: 非缺失样本数
    - r_dict[t]: 取值范围 (max-min)
    """
    n_dict, r_dict = {}, {}
    for t in tasks:
        vals = df[t].dropna().values
        n_dict[t] = len(vals)
        if len(vals) > 0:
            r_dict[t] = float(np.nanmax(vals) - np.nanmin(vals))
        else:
            r_dict[t] = 1.0
    return n_dict, r_dict


# ------------------ EarlyStopping ------------------
class EarlyStopping:
    def __init__(self, patience=20, verbose=True):
        self.patience = patience
        self.best_loss = float("inf")
        self.counter = 0
        self.early_stop = False
        self.verbose = verbose

    def step(self, val_loss):
        if val_loss < self.best_loss:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.verbose:
                print(f"EarlyStopping counter: {self.counter}/{self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True


# ------------------ Train / Eval ------------------
def train_one_epoch(model, loader, optimizer, loss_fn, device, tasks, epoch=None, max_epochs=None):
    model.train()
    total_loss = 0.0
    per_task_accum = {t: 0.0 for t in tasks}
    n_batches = 0

    pbar = tqdm(loader, desc=f"Train Epoch {epoch}/{max_epochs}", leave=False)  
    for batch in pbar:
        batch = batch.to(device)
        optimizer.zero_grad()
        outputs = model(batch)  # dict {task: [B]}

        targets = {t: batch.y[:, i] for i, t in enumerate(tasks)}
        loss, per_task = loss_fn(outputs, targets, return_per_task=True)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        for t in per_task:
            per_task_accum[t] += per_task[t]
        n_batches += 1

        # 在进度条上显示当前 loss
        pbar.set_postfix(loss=loss.item())

    avg_loss = total_loss / n_batches
    avg_task_loss = {t: per_task_accum[t] / n_batches for t in tasks}
    return avg_loss, avg_task_loss


@torch.no_grad()
def evaluate(model, loader, loss_fn, device, tasks, epoch=None, max_epochs=None):
    model.eval()
    total_loss = 0.0
    per_task_accum = {t: 0.0 for t in tasks}
    n_batches = 0

    pbar = tqdm(loader, desc=f"Val   Epoch {epoch}/{max_epochs}", leave=False)
    for batch in pbar:
        batch = batch.to(device)
        outputs = model(batch)

        targets = {t: batch.y[:, i] for i, t in enumerate(tasks)}
        loss, per_task = loss_fn(outputs, targets, return_per_task=True)
        total_loss += loss.item()
        for t in per_task:
            per_task_accum[t] += per_task[t]
        n_batches += 1

        pbar.set_postfix(loss=loss.item())

    avg_loss = total_loss / n_batches
    avg_task_loss = {t: per_task_accum[t] / n_batches for t in tasks}
    return avg_loss, avg_task_loss


# ------------------ Main Loop ------------------
def run_training(model, train_loader, val_loader,
                 optimizer, tasks, train_df,
                 device="cuda", max_epochs=100, patience=20):
    # 1) wMAE Loss
    n_dict, r_dict = compute_task_stats(train_df, tasks)
    loss_fn = WMAELoss(tasks, n_dict, r_dict)

    # 2) Scheduler + EarlyStopping
    scheduler = CosineAnnealingLR(optimizer, T_max=max_epochs)
    early_stopper = EarlyStopping(patience=patience)

    history = []
    best_val = float("inf")
    best_state = None

    for epoch in range(1, max_epochs + 1):
        train_loss, train_task_loss = train_one_epoch(
            model, train_loader, optimizer, loss_fn, device, tasks, epoch=epoch, max_epochs=max_epochs
        )
        val_loss, val_task_loss = evaluate(
            model, val_loader, loss_fn, device, tasks, epoch=epoch, max_epochs=max_epochs
        )
        scheduler.step()

        # 打印结果
        task_str = " | ".join([f"{t}: {val_task_loss[t]:.4f}" for t in tasks])
        print(f"Epoch {epoch:03d}: "
              f"Train={train_loss:.4f}, Val={val_loss:.4f} || {task_str}")

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_task_loss": val_task_loss,
        })

        # 保存最好模型
        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        # early stopping
        early_stopper.step(val_loss)
        if early_stopper.early_stop:
            print("Early stopping triggered.")
            break

    # 恢复最优参数
    if best_state is not None:
        model.load_state_dict(best_state)

    return model, history

# ------------------ Traditional ML Training Pipeline ------------------
import joblib
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.feature_selection import SelectFromModel, VarianceThreshold
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

class TraditionalMLPipeline:
    """Complete pipeline for traditional ML models (XGBoost, LightGBM, CatBoost)"""
    
    def __init__(self, model_type: str = 'xgboost', config: Optional['MLModelConfig'] = None):
        from model import MLModelConfig  # Import here to avoid circular import
        
        self.model_type = model_type.lower()
        self.config = config or MLModelConfig()
        self.scalers = {}
        self.feature_selectors = {}
        self.models = {}
        self.cv_results = {}
        self.preprocessors = {}
        
        # Create output directory
        os.makedirs(f"models/{self.model_type}", exist_ok=True)
        os.makedirs("results", exist_ok=True)
    
    def prepare_features(self, features_df: pd.DataFrame, targets: List[str], 
                        fit: bool = False) -> pd.DataFrame:
        """Extract and process features"""
        # Get feature columns (exclude SMILES, targets, id)
        exclude_cols = set(['SMILES', 'id'] + targets)
        feature_cols = [col for col in features_df.columns if col not in exclude_cols]
        
        if len(feature_cols) == 0:
            raise ValueError("No feature columns found!")
        
        X = features_df[feature_cols].copy()
        
        # Handle missing values and infinities
        X = X.replace([np.inf, -np.inf], np.nan)
        
        if fit:
            # Fit preprocessing steps
            
            # Variance threshold
            if self.config.use_variance_threshold:
                self.preprocessors['variance_selector'] = VarianceThreshold(
                    threshold=self.config.variance_threshold)
                X = pd.DataFrame(
                    self.preprocessors['variance_selector'].fit_transform(X),
                    columns=X.columns[self.preprocessors['variance_selector'].get_support()],
                    index=X.index
                )
                print(f"Variance threshold removed {len(feature_cols) - len(X.columns)} features")
            
            # Correlation filter
            if self.config.use_correlation_filter:
                corr_matrix = X.corr().abs()
                upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
                high_corr_features = [column for column in upper_tri.columns 
                                    if any(upper_tri[column] > self.config.correlation_threshold)]
                X = X.drop(columns=high_corr_features)
                self.preprocessors['kept_features'] = X.columns.tolist()
                print(f"Correlation filter removed {len(high_corr_features)} features")
            
            # Fill missing values with median
            self.preprocessors['feature_medians'] = X.median()
            X = X.fillna(self.preprocessors['feature_medians'])
            
            # Fit scaler
            if self.config.use_robust_scaler:
                self.preprocessors['feature_scaler'] = RobustScaler()
            else:
                self.preprocessors['feature_scaler'] = StandardScaler()
            
            scaled_features = self.preprocessors['feature_scaler'].fit_transform(X)
            X = pd.DataFrame(scaled_features, columns=X.columns, index=X.index)
        
        else:
            # Transform using fitted preprocessors
            if 'variance_selector' in self.preprocessors:
                X = pd.DataFrame(
                    self.preprocessors['variance_selector'].transform(X),
                    columns=X.columns[self.preprocessors['variance_selector'].get_support()],
                    index=X.index
                )
            
            if 'kept_features' in self.preprocessors:
                available_cols = [col for col in self.preprocessors['kept_features'] 
                                if col in X.columns]
                X = X[available_cols]
            
            # Fill missing values
            if 'feature_medians' in self.preprocessors:
                X = X.fillna(self.preprocessors['feature_medians'])
            
            # Scale features
            if 'feature_scaler' in self.preprocessors:
                scaled_features = self.preprocessors['feature_scaler'].transform(X)
                X = pd.DataFrame(scaled_features, columns=X.columns, index=X.index)
        
        return X
    
    def train_target(self, X: pd.DataFrame, y: pd.Series, target: str) -> Dict:
        """Train model for a specific target"""
        from model import create_ml_model  # Import here to avoid circular import
        
        print(f"\nTraining {self.model_type.upper()} for {target}...")
        
        # Set up cross-validation
        if self.config.use_stratified_cv and len(y.unique()) > self.config.n_folds:
            try:
                y_bins = pd.qcut(y, q=self.config.n_folds, labels=False, duplicates='drop')
                kf = StratifiedKFold(n_splits=self.config.n_folds, shuffle=True, 
                                   random_state=self.config.random_state)
                splits = list(kf.split(X, y_bins))
            except:
                kf = KFold(n_splits=self.config.n_folds, shuffle=True, 
                          random_state=self.config.random_state)
                splits = list(kf.split(X, y))
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
            
            # Create and train model
            model = create_ml_model(self.model_type, target, self.config)
            
            # Train with validation set for early stopping
            if self.model_type == 'xgboost':
                model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
            elif self.model_type == 'lightgbm':
                model.fit(X_train, y_train, eval_set=[(X_val, y_val)], 
                         callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)])
            elif self.model_type == 'catboost':
                model.fit(X_train, y_train, eval_set=(X_val, y_val))
            
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
            if self.model_type == 'xgboost':
                feature_importance = np.mean([model.feature_importances_ for model in models], axis=0)
            elif self.model_type == 'lightgbm':
                feature_importance = np.mean([model.feature_importances_ for model in models], axis=0)
            elif self.model_type == 'catboost':
                feature_importance = np.mean([model.feature_importances_ for model in models], axis=0)
            
            # Select features above median importance
            threshold = np.median(feature_importance)
            selected_features = X.columns[feature_importance >= threshold]
            print(f"  Selected {len(selected_features)}/{len(X.columns)} features")
            
            # Retrain models with selected features if significantly fewer
            if len(selected_features) < len(X.columns) * 0.8:
                X_selected = X[selected_features]
                models = []
                
                for fold, (train_idx, val_idx) in enumerate(splits):
                    X_train, X_val = X_selected.iloc[train_idx], X_selected.iloc[val_idx]
                    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
                    
                    model = create_ml_model(self.model_type, target, self.config)
                    if self.model_type == 'xgboost':
                        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
                    elif self.model_type == 'lightgbm':
                        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], 
                                 callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)])
                    elif self.model_type == 'catboost':
                        model.fit(X_train, y_train, eval_set=(X_val, y_val))
                    models.append(model)
                
                self.feature_selectors[target] = selected_features
        
        return {
            'models': models,
            'scores': avg_scores,
            'fold_scores': fold_scores
        }
    
    def train_all_targets(self, features_df: pd.DataFrame, targets: List[str]):
        """Train models for all targets"""
        print("Preparing features...")
        X = self.prepare_features(features_df, targets, fit=True)
        
        print(f"Feature matrix shape: {X.shape}")
        
        for target in targets:
            if target not in features_df.columns:
                print(f"Warning: Target {target} not found in data")
                continue
            
            # Get valid samples for this target
            valid_mask = features_df[target].notna()
            if valid_mask.sum() < 10:
                print(f"Warning: Not enough valid samples for {target} ({valid_mask.sum()})")
                continue
            
            X_target = X[valid_mask].copy()
            y_target = features_df[target][valid_mask].copy()
            
            # Train model
            result = self.train_target(X_target, y_target, target)
            self.models[target] = result['models']
            self.cv_results[target] = result
        
        # Save models and preprocessors
        self.save_models()
    
    def predict(self, features_df: pd.DataFrame, targets: List[str]) -> pd.DataFrame:
        """Make predictions on test data"""
        print("Preparing test features...")
        X_test = self.prepare_features(features_df, targets, fit=False)
        
        predictions = {'id': features_df['id']}
        
        for target in targets:
            if target not in self.models:
                print(f"Warning: No model found for {target}")
                predictions[target] = np.nan
                continue
            
            X_target = X_test.copy()
            
            # Apply feature selection if used
            if target in self.feature_selectors:
                available_features = [f for f in self.feature_selectors[target] 
                                    if f in X_target.columns]
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
        # Save preprocessors
        for name, obj in self.preprocessors.items():
            joblib.dump(obj, f"models/{self.model_type}/{name}.pkl")
        
        # Save models for each target
        for target, models in self.models.items():
            target_dir = f"models/{self.model_type}/{target}"
            os.makedirs(target_dir, exist_ok=True)
            
            for i, model in enumerate(models):
                joblib.dump(model, f"{target_dir}/model_fold_{i}.pkl")
            
            if target in self.feature_selectors:
                joblib.dump(self.feature_selectors[target], f"{target_dir}/selected_features.pkl")
        
        # Save CV results
        results_df = pd.DataFrame({
            target: {
                'mae_mean': results['scores']['mae'],
                'mae_std': results['scores']['mae_std'],
                'rmse_mean': results['scores']['rmse'],
                'rmse_std': results['scores']['rmse_std'],
                'r2_mean': results['scores']['r2'],
                'r2_std': results['scores']['r2_std']
            }
            for target, results in self.cv_results.items()
        }).T
        results_df.to_csv(f"results/{self.model_type}_cv_results.csv")
        
        print("Models and results saved!")
    
    def load_models(self):
        """Load trained models and preprocessors"""
        # Load preprocessors
        preprocessor_files = ['feature_scaler.pkl', 'feature_medians.pkl', 
                             'variance_selector.pkl', 'kept_features.pkl']
        for name in preprocessor_files:
            path = f"models/{self.model_type}/{name}"
            if os.path.exists(path):
                self.preprocessors[name.replace('.pkl', '')] = joblib.load(path)
        
        # Load models for each target
        self.models = {}
        self.feature_selectors = {}
        
        model_base_dir = f"models/{self.model_type}"
        if not os.path.exists(model_base_dir):
            print(f"Model directory {model_base_dir} not found")
            return
        
        for target_dir in os.listdir(model_base_dir):
            target_path = os.path.join(model_base_dir, target_dir)
            if os.path.isdir(target_path) and not target_dir.endswith('.pkl'):
                target = target_dir
                models = []
                
                # Load all fold models
                fold_files = [f for f in os.listdir(target_path) if f.startswith('model_fold_')]
                fold_files.sort()
                
                for fold_file in fold_files:
                    model = joblib.load(os.path.join(target_path, fold_file))
                    models.append(model)
                
                self.models[target] = models
                
                # Load feature selection if exists
                selected_features_path = os.path.join(target_path, 'selected_features.pkl')
                if os.path.exists(selected_features_path):
                    self.feature_selectors[target] = joblib.load(selected_features_path)
        
        print(f"Loaded {self.model_type} models for targets: {list(self.models.keys())}")
    
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
            axes[i].set_title(f'{self.model_type.upper()} Cross-Validation {metric_name}')
            axes[i].set_ylabel(metric_name)
            axes[i].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig(f"results/{self.model_type}_cv_results.png", dpi=300, bbox_inches='tight')
        plt.show()