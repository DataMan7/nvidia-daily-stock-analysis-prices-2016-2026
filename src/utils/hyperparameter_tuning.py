"""
Hyperparameter Tuning Utilities
==============================

Automated hyperparameter optimization for time-series models.
Supports grid search, random search, and Bayesian optimization.

Author: Senior ML Engineer
Date: February 2026
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Any, Callable
from sklearn.model_selection import ParameterGrid, ParameterSampler
from sklearn.metrics import make_scorer
import warnings
import time
from datetime import datetime

# Optional dependencies
try:
    import optuna
    HAS_OPTUNA = True
except ImportError:
    HAS_OPTUNA = False

try:
    from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
    HAS_HYPEROPT = True
except ImportError:
    HAS_HYPEROPT = False


class HyperparameterTuner:
    """
    Hyperparameter tuning for time-series models.

    Supports multiple optimization strategies:
    - Grid Search
    - Random Search
    - Bayesian Optimization (Optuna)
    - Tree-structured Parzen Estimator (Hyperopt)
    """

    def __init__(
        self,
        model_class: Any,
        param_space: Dict[str, List],
        cv_strategy: str = 'walk_forward',
        cv_params: Optional[Dict] = None,
        scoring: Optional[Callable] = None,
        n_jobs: int = 1,
        verbose: int = 1
    ):
        """
        Initialize hyperparameter tuner.

        Args:
            model_class: Model class to optimize
            param_space: Parameter search space
            cv_strategy: Cross-validation strategy
            cv_params: CV parameters
            scoring: Scoring function (higher = better)
            n_jobs: Number of parallel jobs
            verbose: Verbosity level
        """
        self.model_class = model_class
        self.param_space = param_space
        self.cv_strategy = cv_strategy
        self.cv_params = cv_params or {}
        self.scoring = scoring
        self.n_jobs = n_jobs
        self.verbose = verbose

        # Import here to avoid circular imports
        from src.evaluation.cross_validation import TimeSeriesCrossValidator

        self.cv_class = TimeSeriesCrossValidator

    def grid_search(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        max_evals: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Perform grid search over parameter space.

        Args:
            X: Feature matrix
            y: Target vector
            max_evals: Maximum evaluations (for large grids)

        Returns:
            Best parameters and results
        """
        param_grid = list(ParameterGrid(self.param_space))

        if max_evals and len(param_grid) > max_evals:
            # Random sample from grid
            np.random.seed(42)
            indices = np.random.choice(len(param_grid), max_evals, replace=False)
            param_grid = [param_grid[i] for i in indices]

        return self._evaluate_param_combinations(param_grid, X, y, "Grid Search")

    def random_search(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        n_iter: int = 50
    ) -> Dict[str, Any]:
        """
        Perform random search over parameter space.

        Args:
            X: Feature matrix
            y: Target vector
            n_iter: Number of random combinations to try

        Returns:
            Best parameters and results
        """
        param_sampler = ParameterSampler(self.param_space, n_iter=n_iter, random_state=42)
        param_list = list(param_sampler)

        return self._evaluate_param_combinations(param_list, X, y, "Random Search")

    def bayesian_optimization(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        n_trials: int = 50,
        timeout: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Perform Bayesian optimization using Optuna.

        Args:
            X: Feature matrix
            y: Target vector
            n_trials: Number of optimization trials
            timeout: Timeout in seconds

        Returns:
            Best parameters and results
        """
        if not HAS_OPTUNA:
            raise ImportError("Optuna required for Bayesian optimization. Install with: pip install optuna")

        def objective(trial):
            # Sample parameters
            params = {}
            for param_name, param_values in self.param_space.items():
                if isinstance(param_values, list):
                    if isinstance(param_values[0], int):
                        params[param_name] = trial.suggest_int(param_name, min(param_values), max(param_values))
                    elif isinstance(param_values[0], float):
                        params[param_name] = trial.suggest_float(param_name, min(param_values), max(param_values))
                    else:
                        params[param_name] = trial.suggest_categorical(param_name, param_values)
                else:
                    # Assume it's a distribution specification
                    if 'suggest_' in str(param_values):
                        params[param_name] = param_values(trial)
                    else:
                        params[param_name] = trial.suggest_categorical(param_name, param_values)

            # Evaluate
            score = self._evaluate_single_params(params, X, y)
            return score

        # Run optimization
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=n_trials, timeout=timeout)

        return {
            'best_params': study.best_params,
            'best_score': study.best_value,
            'study': study,
            'method': 'Bayesian Optimization (Optuna)'
        }

    def hyperopt_optimization(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        max_evals: int = 50
    ) -> Dict[str, Any]:
        """
        Perform optimization using Hyperopt (TPE).

        Args:
            X: Feature matrix
            y: Target vector
            max_evals: Maximum evaluations

        Returns:
            Best parameters and results
        """
        if not HAS_HYPEROPT:
            raise ImportError("Hyperopt required for TPE optimization. Install with: pip install hyperopt")

        # Convert parameter space to hyperopt format
        space = {}
        for param_name, param_values in self.param_space.items():
            if isinstance(param_values, list):
                if isinstance(param_values[0], int):
                    space[param_name] = hp.choice(param_name, param_values)
                elif isinstance(param_values[0], float):
                    space[param_name] = hp.uniform(param_name, min(param_values), max(param_values))
                else:
                    space[param_name] = hp.choice(param_name, param_values)
            else:
                space[param_name] = param_values

        def objective(params):
            # Convert choice indices back to values
            converted_params = {}
            for param_name, param_values in self.param_space.items():
                if isinstance(param_values, list) and not isinstance(param_values[0], (int, float)):
                    converted_params[param_name] = param_values[params[param_name]]
                else:
                    converted_params[param_name] = params[param_name]

            score = self._evaluate_single_params(converted_params, X, y)
            return {'loss': -score, 'status': STATUS_OK}

        # Run optimization
        trials = Trials()
        best = fmin(objective, space, algo=tpe.suggest, max_evals=max_evals, trials=trials)

        # Convert best params back
        best_params = {}
        for param_name, param_values in self.param_space.items():
            if isinstance(param_values, list) and not isinstance(param_values[0], (int, float)):
                best_params[param_name] = param_values[best[param_name]]
            else:
                best_params[param_name] = best[param_name]

        return {
            'best_params': best_params,
            'best_score': -trials.best_trial['result']['loss'],
            'trials': trials,
            'method': 'TPE (Hyperopt)'
        }

    def _evaluate_param_combinations(
        self,
        param_list: List[Dict],
        X: pd.DataFrame,
        y: pd.Series,
        method_name: str
    ) -> Dict[str, Any]:
        """Evaluate multiple parameter combinations"""
        results = []

        for i, params in enumerate(param_list):
            if self.verbose > 0:
                print(f"{method_name}: Evaluating combination {i+1}/{len(param_list)}")

            start_time = time.time()
            score = self._evaluate_single_params(params, X, y)
            elapsed = time.time() - start_time

            results.append({
                'params': params,
                'score': score,
                'time': elapsed
            })

        # Find best
        best_result = max(results, key=lambda x: x['score'])

        return {
            'best_params': best_result['params'],
            'best_score': best_result['score'],
            'all_results': results,
            'method': method_name,
            'total_evaluations': len(results)
        }

    def _evaluate_single_params(self, params: Dict, X: pd.DataFrame, y: pd.Series) -> float:
        """Evaluate a single parameter combination"""
        # Create CV splitter
        cv = self.cv_class(strategy=self.cv_strategy, **self.cv_params)

        scores = []

        for train_idx, test_idx in cv.split(X):
            # Split data
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

            # Train model
            model = self.model_class(**params)
            model.fit(X_train, y_train)

            # Evaluate
            y_pred = model.predict(X_test)

            if self.scoring:
                score = self.scoring(y_test, y_pred)
            else:
                # Default: negative MAE (higher = better for maximization)
                from sklearn.metrics import mean_absolute_error
                score = -mean_absolute_error(y_test, y_pred)

            scores.append(score)

        return np.mean(scores)


def create_param_space_for_xgboost() -> Dict[str, List]:
    """
    Create reasonable parameter space for XGBoost tuning.

    Returns:
        Parameter space dictionary
    """
    return {
        'max_depth': [3, 6, 9],
        'learning_rate': [0.01, 0.1, 0.3],
        'n_estimators': [50, 100, 200],
        'subsample': [0.8, 0.9, 1.0],
        'colsample_bytree': [0.8, 0.9, 1.0],
        'min_child_weight': [1, 3, 5]
    }


def create_param_space_for_random_forest() -> Dict[str, List]:
    """
    Create reasonable parameter space for Random Forest tuning.

    Returns:
        Parameter space dictionary
    """
    return {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['auto', 'sqrt', 'log2']
    }


def optimize_model_hyperparameters(
    model_class: Any,
    X: pd.DataFrame,
    y: pd.Series,
    method: str = 'auto',
    **tune_kwargs
) -> Dict[str, Any]:
    """
    Convenience function for hyperparameter optimization.

    Args:
        model_class: Model class to optimize
        X: Feature matrix
        y: Target vector
        method: Optimization method ('auto', 'grid', 'random', 'bayesian', 'hyperopt')
                 'auto' will choose the best available method
        **tune_kwargs: Additional arguments for tuner

    Returns:
        Optimization results
    """
    # Set default parameter spaces based on model
    if 'xgboost' in str(model_class).lower():
        param_space = create_param_space_for_xgboost()
    elif 'randomforest' in str(model_class).lower():
        param_space = create_param_space_for_random_forest()
    else:
        raise ValueError(f"No default parameter space for {model_class}")

    # Override with custom params if provided
    if 'param_space' in tune_kwargs:
        param_space = tune_kwargs.pop('param_space')

    # Auto-select best method
    if method == 'auto':
        if HAS_OPTUNA:
            method = 'bayesian'
            print("Using Bayesian optimization (Optuna) - recommended for best results")
        elif HAS_HYPEROPT:
            method = 'hyperopt'
            print("Using Hyperopt TPE optimization")
        else:
            method = 'random'
            print("Using random search (install Optuna for better optimization)")

    # Create tuner
    tuner = HyperparameterTuner(model_class, param_space, **tune_kwargs)

    # Run optimization
    if method == 'grid':
        return tuner.grid_search(X, y)
    elif method == 'random':
        return tuner.random_search(X, y)
    elif method == 'bayesian':
        return tuner.bayesian_optimization(X, y)
    elif method == 'hyperopt':
        return tuner.hyperopt_optimization(X, y)
    else:
        raise ValueError(f"Unknown optimization method: {method}")


if __name__ == "__main__":
    # Example usage
    print("=" * 80)
    print("HYPERPARAMETER TUNING - EXAMPLE USAGE".center(80))
    print("=" * 80)

    try:
        from sklearn.ensemble import RandomForestRegressor
        import numpy as np

        # Create sample data
        np.random.seed(42)
        X = pd.DataFrame({
            'feature1': np.random.randn(200),
            'feature2': np.random.randn(200),
            'feature3': np.random.randn(200)
        })
        y = X.sum(axis=1) + np.random.randn(200) * 0.1

        print("🔧 Setting up hyperparameter tuning...")

        # Create tuner
        param_space = {
            'n_estimators': [10, 50, 100],
            'max_depth': [3, 5, None],
            'min_samples_split': [2, 5]
        }

        tuner = HyperparameterTuner(
            RandomForestRegressor,
            param_space,
            cv_strategy='walk_forward',
            cv_params={'n_splits': 3}
        )

        print("🎯 Running random search...")
        results = tuner.random_search(X, y, n_iter=5)

        print(f"\\n🏆 Best Parameters: {results['best_params']}")
        print(f"🏆 Best Score: {results['best_score']:.4f}")
        print(f"📊 Total Evaluations: {results['total_evaluations']}")

        # Show available methods
        print("\\n📚 Available Optimization Methods:")
        print("  • Grid Search: Exhaustive search over all combinations")
        print("  • Random Search: Random sampling from parameter space")
        if HAS_OPTUNA:
            print("  • Bayesian Optimization: Smart search using Optuna")
        if HAS_HYPEROPT:
            print("  • TPE (Hyperopt): Tree-structured Parzen Estimator")

    except ImportError as e:
        print(f"⚠️  Import error: {e}")
        print("Install required packages: pip install scikit-learn pandas numpy")

    print("\\n" + "=" * 80)
    print("✅ Hyperparameter tuning example completed!")
    print("=" * 80)