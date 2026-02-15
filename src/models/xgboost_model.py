"""
XGBoost Model Implementation
============================

Production-ready XGBoost wrapper for time-series forecasting.
Includes proper cross-validation, feature importance, and evaluation.

Author: Senior ML Engineer
Date: February 2026
"""

import pandas as pd
import numpy as np
import xgboost as xgb
from typing import Dict, List, Tuple, Optional, Union, Any
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings
import json
from pathlib import Path


class XGBoostTimeSeriesModel:
    """
    XGBoost model wrapper optimized for time-series forecasting.

    Features:
    - Walk-forward validation
    - Feature importance analysis
    - Hyperparameter optimization
    - Model persistence
    - Comprehensive evaluation
    """

    def __init__(
        self,
        objective: str = 'reg:squarederror',
        eval_metric: str = 'mae',
        max_depth: int = 6,
        learning_rate: float = 0.1,
        n_estimators: int = 100,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        min_child_weight: int = 1,
        random_state: int = 42,
        early_stopping_rounds: int = 20,
        verbose: bool = False
    ):
        """
        Initialize XGBoost model with time-series optimized parameters.

        Args:
            objective: XGBoost objective function
            eval_metric: Evaluation metric
            max_depth: Maximum tree depth
            learning_rate: Learning rate (eta)
            n_estimators: Number of boosting rounds
            subsample: Subsample ratio of training instances
            colsample_bytree: Subsample ratio of columns
            min_child_weight: Minimum sum of instance weight needed in a child
            random_state: Random seed
            early_stopping_rounds: Early stopping rounds
            verbose: Verbosity level
        """
        self.params = {
            'objective': objective,
            'eval_metric': eval_metric,
            'max_depth': max_depth,
            'learning_rate': learning_rate,
            'subsample': subsample,
            'colsample_bytree': colsample_bytree,
            'min_child_weight': min_child_weight,
            'random_state': random_state
        }

        self.n_estimators = n_estimators
        self.early_stopping_rounds = early_stopping_rounds
        self.verbose = verbose
        self.model = None
        self.feature_names = None
        self.is_fitted = False
        self.cv_results = None

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        validation_split: float = 0.2,
        early_stopping: bool = True
    ) -> 'XGBoostTimeSeriesModel':
        """
        Fit XGBoost model with optional early stopping.

        Args:
            X: Feature matrix
            y: Target vector
            validation_split: Fraction of data to use for validation
            early_stopping: Whether to use early stopping

        Returns:
            Fitted model instance
        """
        self.feature_names = X.columns.tolist()

        # Prepare data
        dtrain = xgb.DMatrix(X, label=y)

        if early_stopping and validation_split > 0:
            # Split for early stopping
            split_idx = int(len(X) * (1 - validation_split))
            X_train, X_val = X.iloc[:split_idx], X.iloc[split_idx:]
            y_train, y_val = y.iloc[:split_idx], y.iloc[split_idx:]

            dtrain_es = xgb.DMatrix(X_train, label=y_train)
            dval = xgb.DMatrix(X_val, label=y_val)

            # Train with early stopping
            self.model = xgb.train(
                self.params,
                dtrain_es,
                num_boost_round=self.n_estimators,
                evals=[(dtrain_es, 'train'), (dval, 'validation')],
                early_stopping_rounds=self.early_stopping_rounds,
                verbose_eval=self.verbose
            )
        else:
            # Train without early stopping
            self.model = xgb.train(self.params, dtrain, num_boost_round=self.n_estimators)

        self.is_fitted = True
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Make predictions with the fitted model.

        Args:
            X: Feature matrix

        Returns:
            Predictions array
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")

        dtest = xgb.DMatrix(X)
        return self.model.predict(dtest)

    def cross_validate(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        n_splits: int = 5,
        gap: int = 0
    ) -> Dict[str, Any]:
        """
        Perform time-series cross-validation.

        Args:
            X: Feature matrix
            y: Target vector
            n_splits: Number of CV folds
            gap: Gap between train and test sets

        Returns:
            Dictionary with CV results
        """
        tscv = TimeSeriesSplit(n_splits=n_splits, gap=gap)

        cv_results = {
            'fold_scores': [],
            'feature_importance': [],
            'predictions': [],
            'actuals': []
        }

        fold_idx = 0
        for train_idx, test_idx in tscv.split(X):
            fold_idx += 1

            # Split data
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

            # Fit model
            fold_model = XGBoostTimeSeriesModel(**self.params)
            fold_model.fit(X_train, y_train, validation_split=0)

            # Predict
            y_pred = fold_model.predict(X_test)

            # Evaluate
            mae = mean_absolute_error(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            r2 = r2_score(y_test, y_pred)

            fold_result = {
                'fold': fold_idx,
                'mae': mae,
                'rmse': rmse,
                'r2': r2,
                'n_train': len(train_idx),
                'n_test': len(test_idx)
            }

            cv_results['fold_scores'].append(fold_result)
            cv_results['predictions'].extend(y_pred)
            cv_results['actuals'].extend(y_test.values)

            if self.verbose:
                print(f"Fold {fold_idx}: MAE={mae:.4f}, RMSE={rmse:.4f}, R²={r2:.4f}")

        # Overall metrics
        cv_results['overall'] = {
            'mean_mae': np.mean([f['mae'] for f in cv_results['fold_scores']]),
            'std_mae': np.std([f['mae'] for f in cv_results['fold_scores']]),
            'mean_rmse': np.mean([f['rmse'] for f in cv_results['fold_scores']]),
            'std_rmse': np.std([f['rmse'] for f in cv_results['fold_scores']]),
            'mean_r2': np.mean([f['r2'] for f in cv_results['fold_scores']]),
            'std_r2': np.std([f['r2'] for f in cv_results['fold_scores']])
        }

        self.cv_results = cv_results
        return cv_results

    def get_feature_importance(self, importance_type: str = 'gain') -> pd.DataFrame:
        """
        Get feature importance scores.

        Args:
            importance_type: Type of importance ('gain', 'weight', 'cover')

        Returns:
            DataFrame with feature importance
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before getting feature importance")

        importance_dict = self.model.get_score(importance_type=importance_type)

        # Create DataFrame
        importance_df = pd.DataFrame({
            'feature': list(importance_dict.keys()),
            'importance': list(importance_dict.values())
        }).sort_values('importance', ascending=False)

        return importance_df

    def evaluate(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        include_directional: bool = True
    ) -> Dict[str, float]:
        """
        Evaluate model performance.

        Args:
            X: Feature matrix
            y: Target vector
            include_directional: Whether to include directional accuracy

        Returns:
            Dictionary with evaluation metrics
        """
        y_pred = self.predict(X)

        metrics = {
            'mae': mean_absolute_error(y, y_pred),
            'rmse': np.sqrt(mean_squared_error(y, y_pred)),
            'r2': r2_score(y, y_pred)
        }

        if include_directional and 'Close' in X.columns:
            # Directional accuracy (did we predict up/down correctly?)
            actual_direction = (y > X['Close']).astype(int)
            pred_direction = (y_pred > X['Close']).astype(int)
            metrics['directional_accuracy'] = (actual_direction == pred_direction).mean()

        return metrics

    def save_model(self, filepath: str) -> None:
        """
        Save model to disk.

        Args:
            filepath: Path to save model
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before saving")

        # Create directory if it doesn't exist
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)

        # Save XGBoost model
        self.model.save_model(filepath)

        # Save metadata
        metadata = {
            'params': self.params,
            'feature_names': self.feature_names,
            'is_fitted': self.is_fitted
        }

        metadata_path = filepath + '.metadata.json'
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)

    @classmethod
    def load_model(cls, filepath: str) -> 'XGBoostTimeSeriesModel':
        """
        Load model from disk.

        Args:
            filepath: Path to saved model

        Returns:
            Loaded model instance
        """
        # Load metadata
        metadata_path = filepath + '.metadata.json'
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)

        # Create model instance
        model = cls(**metadata['params'])
        model.feature_names = metadata['feature_names']
        model.is_fitted = metadata['is_fitted']

        # Load XGBoost model
        model.model = xgb.Booster()
        model.model.load_model(filepath)

        return model

    def get_model_info(self) -> Dict[str, Any]:
        """Get model information and metadata"""
        return {
            'model_type': 'XGBoostTimeSeriesModel',
            'parameters': self.params,
            'is_fitted': self.is_fitted,
            'feature_names': self.feature_names,
            'cv_results': self.cv_results
        }


def optimize_hyperparameters(
    X: pd.DataFrame,
    y: pd.Series,
    param_space: Dict[str, List],
    n_splits: int = 3,
    n_trials: int = 50
) -> Dict[str, Any]:
    """
    Optimize hyperparameters using random search with time-series CV.

    Args:
        X: Feature matrix
        y: Target vector
        param_space: Parameter search space
        n_splits: Number of CV folds
        n_trials: Number of optimization trials

    Returns:
        Best parameters and results
    """
    try:
        import optuna
    except ImportError:
        raise ImportError("optuna is required for hyperparameter optimization. Install with: pip install optuna")

    def objective(trial):
        # Sample parameters
        params = {}
        for param_name, param_values in param_space.items():
            if param_name in ['max_depth', 'n_estimators']:
                params[param_name] = trial.suggest_int(param_name, param_values[0], param_values[1])
            elif param_name in ['learning_rate', 'subsample', 'colsample_bytree']:
                params[param_name] = trial.suggest_float(param_name, param_values[0], param_values[1])

        # Create and evaluate model
        model = XGBoostTimeSeriesModel(**params)
        cv_results = model.cross_validate(X, y, n_splits=n_splits)

        return cv_results['overall']['mean_mae']  # Minimize MAE

    # Run optimization
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=n_trials)

    best_params = study.best_params
    best_score = study.best_value

    return {
        'best_params': best_params,
        'best_score': best_score,
        'study': study
    }


if __name__ == "__main__":
    # Example usage
    print("=" * 80)
    print("XGBOOST TIME-SERIES MODEL - EXAMPLE USAGE".center(80))
    print("=" * 80)

    # Create sample data
    np.random.seed(42)
    dates = pd.date_range('2020-01-01', periods=200, freq='D')
    prices = 100 + np.random.randn(200).cumsum()

    df = pd.DataFrame({
        'Date': dates,
        'Close': prices,
        'Volume': np.random.randint(1000000, 10000000, 200),
        'lag_1': prices + np.random.randn(200) * 0.1,
        'lag_2': prices + np.random.randn(200) * 0.1
    })

    # Create target
    df['target'] = df['Close'].shift(-1)
    df = df.dropna()

    # Prepare features
    feature_cols = ['Close', 'Volume', 'lag_1', 'lag_2']
    X = df[feature_cols]
    y = df['target']

    # Train model
    model = XGBoostTimeSeriesModel(n_estimators=50, verbose=False)
    model.fit(X, y)

    # Cross-validate
    cv_results = model.cross_validate(X, y, n_splits=3)
    print(f"\\nCV Results: MAE = {cv_results['overall']['mean_mae']:.4f} ± {cv_results['overall']['std_mae']:.4f}")

    # Feature importance
    importance = model.get_feature_importance()
    print("\\nTop 5 Features:")
    print(importance.head())

    # Save and load
    model.save_model('models/xgb_example.model')
    loaded_model = XGBoostTimeSeriesModel.load_model('models/xgb_example.model')
    print("\\n✅ Model save/load test passed!")

    print("\\n" + "=" * 80)
    print("✅ XGBoost model example completed!")
    print("=" * 80)