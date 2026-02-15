"""
MLflow Integration Utilities
============================

Experiment tracking and model management with MLflow.
Provides comprehensive logging for model training and evaluation.

Features:
- Experiment tracking
- Model logging
- Performance metrics logging
- Run comparison
- Model registry integration

Author: Senior ML Engineer
Date: February 2026
"""

import sys
from pathlib import Path
from typing import Dict, List, Any, Optional, Union
import pandas as pd
import numpy as np
from datetime import datetime
import json
import tempfile

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Optional MLflow import
try:
    import mlflow
    import mlflow.sklearn
    import mlflow.xgboost
    HAS_MLFLOW = True
except ImportError:
    HAS_MLFLOW = False
    mlflow = None

from src.evaluation.metrics import comprehensive_evaluation


class MLflowTracker:
    """
    MLflow experiment tracker for NVIDIA stock prediction project.

    Handles experiment logging, model versioning, and performance tracking.
    """

    def __init__(
        self,
        experiment_name: str = "nvidia_stock_prediction",
        tracking_uri: Optional[str] = None
    ):
        """
        Initialize MLflow tracker.

        Args:
            experiment_name: Name of the MLflow experiment
            tracking_uri: MLflow tracking server URI (optional)
        """
        if not HAS_MLFLOW:
            raise ImportError("MLflow not installed. Install with: pip install mlflow")

        self.experiment_name = experiment_name

        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)

        # Set experiment
        try:
            mlflow.set_experiment(experiment_name)
        except Exception as e:
            print(f"Warning: Could not set experiment: {e}")

    def start_run(self, run_name: Optional[str] = None) -> 'MLflowRun':
        """Start a new MLflow run"""
        return MLflowRun(run_name)

    def log_model_training(
        self,
        model: Any,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        model_params: Dict[str, Any],
        additional_metrics: Optional[Dict[str, float]] = None,
        tags: Optional[Dict[str, str]] = None
    ) -> str:
        """
        Log complete model training run.

        Args:
            model: Trained model
            X_train: Training features
            y_train: Training targets
            X_test: Test features
            y_test: Test targets
            model_params: Model hyperparameters
            additional_metrics: Additional metrics to log
            tags: Run tags

        Returns:
            Run ID
        """
        with mlflow.start_run() as run:
            # Log parameters
            mlflow.log_params(model_params)

            # Log training info
            mlflow.log_param("train_samples", len(X_train))
            mlflow.log_param("test_samples", len(X_test))
            mlflow.log_param("feature_count", X_train.shape[1])
            mlflow.log_param("model_type", type(model).__name__)

            # Evaluate model
            y_pred = model.predict(X_test)
            metrics = comprehensive_evaluation(y_test, y_pred, X_test, "evaluation")

            # Log metrics
            mlflow.log_metrics({
                "mae": metrics["mae"],
                "rmse": metrics["rmse"],
                "r2": metrics["r2"],
                "directional_accuracy": metrics["directional_accuracy"]
            })

            # Log additional metrics
            if additional_metrics:
                mlflow.log_metrics(additional_metrics)

            # Log model
            self._log_model(model, "model")

            # Log tags
            if tags:
                mlflow.set_tags(tags)

            # Log feature importance if available
            if hasattr(model, 'feature_importances_'):
                feature_importance = dict(zip(X_train.columns, model.feature_importances_))
                mlflow.log_dict(feature_importance, "feature_importance.json")

            # Log dataset info
            dataset_info = {
                "train_shape": X_train.shape,
                "test_shape": X_test.shape,
                "features": list(X_train.columns),
                "date_logged": datetime.now().isoformat()
            }
            mlflow.log_dict(dataset_info, "dataset_info.json")

            return run.info.run_id

    def log_backtest_results(
        self,
        backtest_result: Any,
        model_name: str,
        strategy_params: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Log backtesting results.

        Args:
            backtest_result: BacktestResult object
            model_name: Name of the model
            strategy_params: Strategy parameters

        Returns:
            Run ID
        """
        with mlflow.start_run(run_name=f"backtest_{model_name}") as run:
            # Log backtest metrics
            mlflow.log_metrics({
                "total_return": backtest_result.total_return,
                "annualized_return": backtest_result.annualized_return,
                "volatility": backtest_result.volatility,
                "sharpe_ratio": backtest_result.sharpe_ratio,
                "max_drawdown": backtest_result.max_drawdown,
                "win_rate": backtest_result.win_rate,
                "profit_factor": backtest_result.profit_factor,
                "total_trades": backtest_result.total_trades,
                "avg_trade_duration": backtest_result.avg_trade_duration
            })

            # Log strategy parameters
            if strategy_params:
                mlflow.log_params(strategy_params)

            # Log equity curve
            equity_df = backtest_result.equity_curve.reset_index()
            equity_df.columns = ['date', 'equity']
            with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
                equity_df.to_csv(f, index=False)
                mlflow.log_artifact(f.name, "equity_curve.csv")

            # Log tags
            mlflow.set_tags({
                "run_type": "backtest",
                "model_name": model_name
            })

            return run.info.run_id

    def log_cross_validation_results(
        self,
        cv_results: Dict[str, Any],
        model_name: str,
        cv_params: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Log cross-validation results.

        Args:
            cv_results: CV results dictionary
            model_name: Name of the model
            cv_params: CV parameters

        Returns:
            Run ID
        """
        with mlflow.start_run(run_name=f"cv_{model_name}") as run:
            # Log CV parameters
            if cv_params:
                mlflow.log_params(cv_params)

            # Log overall metrics
            if 'overall' in cv_results:
                mlflow.log_metrics({
                    f"cv_{k}": v for k, v in cv_results['overall'].items()
                })

            # Log fold results
            if 'fold_scores' in cv_results:
                for i, fold in enumerate(cv_results['fold_scores']):
                    mlflow.log_metrics({
                        f"fold_{i}_{k}": v for k, v in fold.items() if k != 'fold'
                    })

            # Log tags
            mlflow.set_tags({
                "run_type": "cross_validation",
                "model_name": model_name,
                "cv_folds": len(cv_results.get('fold_scores', []))
            })

            return run.info.run_id

    def _log_model(self, model: Any, artifact_path: str):
        """Log model to MLflow"""
        try:
            # Try XGBoost logging
            if hasattr(model, 'get_booster'):
                mlflow.xgboost.log_model(model, artifact_path)
            # Try sklearn logging
            elif hasattr(model, 'predict'):
                mlflow.sklearn.log_model(model, artifact_path)
            else:
                # Fallback: log as pickle
                import pickle
                with tempfile.NamedTemporaryFile(delete=False) as f:
                    pickle.dump(model, f)
                    mlflow.log_artifact(f.name, f"{artifact_path}.pkl")
        except Exception as e:
            print(f"Warning: Could not log model to MLflow: {e}")

    def get_experiment_runs(self) -> pd.DataFrame:
        """Get all runs for the current experiment"""
        try:
            runs = mlflow.search_runs()
            return runs
        except Exception as e:
            print(f"Warning: Could not retrieve runs: {e}")
            return pd.DataFrame()

    def compare_runs(self, run_ids: List[str]) -> pd.DataFrame:
        """Compare specific runs"""
        try:
            runs = mlflow.search_runs(filter_string=f"run_id in {run_ids}")
            return runs
        except Exception as e:
            print(f"Warning: Could not compare runs: {e}")
            return pd.DataFrame()


class MLflowRun:
    """Context manager for MLflow runs"""

    def __init__(self, run_name: Optional[str] = None):
        self.run_name = run_name
        self.run = None

    def __enter__(self):
        self.run = mlflow.start_run(run_name=self.run_name)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        mlflow.end_run()

    def log_params(self, params: Dict[str, Any]):
        """Log parameters"""
        mlflow.log_params(params)

    def log_metrics(self, metrics: Dict[str, float]):
        """Log metrics"""
        mlflow.log_metrics(metrics)

    def log_param(self, key: str, value: Any):
        """Log single parameter"""
        mlflow.log_param(key, value)

    def log_metric(self, key: str, value: float):
        """Log single metric"""
        mlflow.log_metric(key, value)

    def set_tags(self, tags: Dict[str, str]):
        """Set run tags"""
        mlflow.set_tags(tags)

    def log_artifact(self, local_path: str, artifact_path: Optional[str] = None):
        """Log artifact"""
        mlflow.log_artifact(local_path, artifact_path)

    def log_dict(self, dictionary: Dict, artifact_file: str):
        """Log dictionary as JSON"""
        mlflow.log_dict(dictionary, artifact_file)

    def get_run_id(self) -> str:
        """Get current run ID"""
        return self.run.info.run_id if self.run else None


def setup_mlflow_tracking(experiment_name: str = "nvidia_stock_prediction") -> MLflowTracker:
    """
    Set up MLflow tracking for the project.

    Args:
        experiment_name: Name of the experiment

    Returns:
        Configured MLflowTracker
    """
    if not HAS_MLFLOW:
        raise ImportError("MLflow not available. Install with: pip install mlflow")

    tracker = MLflowTracker(experiment_name)

    print("🔍 MLflow tracking enabled")
    print(f"📊 Experiment: {experiment_name}")
    print(f"📍 Tracking URI: {mlflow.get_tracking_uri()}")

    return tracker


def log_model_comparison(
    tracker: MLflowTracker,
    model_results: Dict[str, Dict[str, Any]],
    comparison_name: str = "model_comparison"
) -> str:
    """
    Log model comparison results.

    Args:
        tracker: MLflow tracker
        model_results: Dictionary of model results
        comparison_name: Name for the comparison

    Returns:
        Run ID
    """
    with tracker.start_run(comparison_name) as run:
        # Log comparison data
        comparison_data = {
            "models_compared": list(model_results.keys()),
            "comparison_timestamp": datetime.now().isoformat(),
            "results": model_results
        }

        # Log as JSON
        run.log_dict(comparison_data, "model_comparison.json")

        # Log summary metrics
        for model_name, results in model_results.items():
            if "metrics" in results:
                metrics = results["metrics"]
                prefixed_metrics = {f"{model_name}_{k}": v for k, v in metrics.items()}
                run.log_metrics(prefixed_metrics)

        run.set_tags({
            "run_type": "model_comparison",
            "models_count": len(model_results)
        })

        return run.get_run_id()


if __name__ == "__main__":
    # Example usage
    print("MLflow Integration Example")
    print("=" * 40)

    if not HAS_MLFLOW:
        print("❌ MLflow not installed")
        print("Install with: pip install mlflow")
        sys.exit(1)

    try:
        # Setup tracking
        tracker = setup_mlflow_tracking()

        print("\\n✅ MLflow tracking ready!")
        print("\\nExample usage:")
        print("# Log model training")
        print("with tracker.start_run('my_model') as run:")
        print("    run.log_params({'learning_rate': 0.1})")
        print("    run.log_metrics({'accuracy': 0.95})")
        print("    run.log_artifact('model.pkl')")

        print("\\n# View experiments")
        print("tracker.get_experiment_runs()")

    except Exception as e:
        print(f"❌ MLflow setup failed: {e}")
        print("Make sure MLflow is properly configured")