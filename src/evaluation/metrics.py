"""
Evaluation Metrics Module
=========================

Comprehensive evaluation metrics for time-series forecasting models.
Includes financial-specific metrics and directional accuracy.

Author: Senior ML Engineer
Date: February 2026
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    mean_absolute_percentage_error
)
import warnings


def regression_metrics(
    y_true: Union[pd.Series, np.ndarray],
    y_pred: Union[pd.Series, np.ndarray]
) -> Dict[str, float]:
    """
    Standard regression metrics.

    Args:
        y_true: True values
        y_pred: Predicted values

    Returns:
        Dictionary with MAE, RMSE, R², MAPE
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # Handle edge cases
    if len(y_true) == 0 or len(y_pred) == 0:
        raise ValueError("Empty arrays provided")

    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have same length")

    metrics = {
        'mae': mean_absolute_error(y_true, y_pred),
        'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
        'r2': r2_score(y_true, y_pred)
    }

    # MAPE (handle division by zero)
    try:
        mape = mean_absolute_percentage_error(y_true, y_pred)
        if np.isfinite(mape):
            metrics['mape'] = mape
        else:
            metrics['mape'] = np.nan
    except:
        metrics['mape'] = np.nan

    return metrics


def directional_accuracy(
    y_true: Union[pd.Series, np.ndarray],
    y_pred: Union[pd.Series, np.ndarray],
    current_price: Optional[Union[pd.Series, np.ndarray]] = None
) -> Dict[str, float]:
    """
    Directional accuracy metrics for financial forecasting.

    Args:
        y_true: True future prices
        y_pred: Predicted future prices
        current_price: Current prices (if None, assumes y_true is returns)

    Returns:
        Dictionary with directional accuracy metrics
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    if current_price is not None:
        current_price = np.array(current_price)
        # Calculate directions from current price
        actual_direction = np.sign(y_true - current_price)
        pred_direction = np.sign(y_pred - current_price)
    else:
        # Assume y_true and y_pred are already directional (e.g., returns)
        actual_direction = np.sign(y_true)
        pred_direction = np.sign(y_pred)

    # Calculate accuracy
    correct_predictions = np.sum(actual_direction == pred_direction)
    total_predictions = len(actual_direction)
    accuracy = correct_predictions / total_predictions

    # Matthews Correlation Coefficient for directional predictions
    # MCC = (TP*TN - FP*FN) / sqrt((TP+FP)(TP+FN)(TN+FP)(TN+FN))
    tp = np.sum((actual_direction == 1) & (pred_direction == 1))
    tn = np.sum((actual_direction == -1) & (pred_direction == -1))
    fp = np.sum((actual_direction == -1) & (pred_direction == 1))
    fn = np.sum((actual_direction == 1) & (pred_direction == -1))

    denominator = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc = (tp * tn - fp * fn) / denominator if denominator != 0 else 0

    return {
        'directional_accuracy': accuracy,
        'matthews_correlation': mcc,
        'correct_predictions': correct_predictions,
        'total_predictions': total_predictions
    }


def financial_metrics(
    y_true: Union[pd.Series, np.ndarray],
    y_pred: Union[pd.Series, np.ndarray],
    current_price: Optional[Union[pd.Series, np.ndarray]] = None,
    risk_free_rate: float = 0.02
) -> Dict[str, float]:
    """
    Financial-specific evaluation metrics.

    Args:
        y_true: True future prices
        y_pred: Predicted future prices
        current_price: Current prices for return calculation
        risk_free_rate: Risk-free rate for Sharpe ratio

    Returns:
        Dictionary with financial metrics
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    metrics = {}

    if current_price is not None:
        current_price = np.array(current_price)

        # Calculate returns
        actual_returns = (y_true - current_price) / current_price
        pred_returns = (y_pred - current_price) / current_price

        # Sharpe ratio (annualized)
        excess_returns = actual_returns - risk_free_rate/252  # Daily risk-free rate
        if np.std(excess_returns) > 0:
            sharpe_ratio = np.sqrt(252) * np.mean(excess_returns) / np.std(excess_returns)
            metrics['sharpe_ratio'] = sharpe_ratio

        # Maximum drawdown
        cumulative_returns = np.cumprod(1 + actual_returns)
        running_max = np.maximum.accumulate(cumulative_returns)
        drawdown = (cumulative_returns - running_max) / running_max
        max_drawdown = np.min(drawdown)
        metrics['max_drawdown'] = max_drawdown

        # Win rate (percentage of profitable predictions)
        profitable_predictions = np.sum(pred_returns > 0)
        total_predictions = len(pred_returns)
        win_rate = profitable_predictions / total_predictions
        metrics['win_rate'] = win_rate

        # Profit factor (gross profit / gross loss)
        winning_trades = pred_returns[pred_returns > 0]
        losing_trades = pred_returns[pred_returns < 0]

        if len(losing_trades) > 0 and np.sum(np.abs(losing_trades)) > 0:
            profit_factor = np.sum(winning_trades) / np.sum(np.abs(losing_trades))
            metrics['profit_factor'] = profit_factor

    return metrics


def comprehensive_evaluation(
    y_true: Union[pd.Series, np.ndarray],
    y_pred: Union[pd.Series, np.ndarray],
    current_price: Optional[Union[pd.Series, np.ndarray]] = None,
    model_name: str = "Model"
) -> Dict[str, Union[float, str]]:
    """
    Comprehensive model evaluation combining all metrics.

    Args:
        y_true: True values
        y_pred: Predicted values
        current_price: Current prices for financial metrics
        model_name: Name of the model

    Returns:
        Dictionary with all evaluation metrics
    """
    results = {'model': model_name}

    # Regression metrics
    reg_metrics = regression_metrics(y_true, y_pred)
    results.update(reg_metrics)

    # Directional accuracy
    dir_metrics = directional_accuracy(y_true, y_pred, current_price)
    results.update(dir_metrics)

    # Financial metrics
    fin_metrics = financial_metrics(y_true, y_pred, current_price)
    results.update(fin_metrics)

    return results


def compare_models(
    models_results: List[Dict[str, Union[float, str]]],
    sort_by: str = 'mae'
) -> pd.DataFrame:
    """
    Compare multiple models side-by-side.

    Args:
        models_results: List of model evaluation results
        sort_by: Metric to sort by (default: 'mae')

    Returns:
        DataFrame with model comparison
    """
    df = pd.DataFrame(models_results)

    # Sort by specified metric (ascending for error metrics, descending for accuracy)
    ascending = sort_by in ['mae', 'rmse', 'mape', 'max_drawdown']
    df = df.sort_values(sort_by, ascending=ascending)

    return df


def cross_validation_summary(
    cv_results: List[Dict[str, float]],
    model_name: str = "Model"
) -> Dict[str, Union[float, str]]:
    """
    Summarize cross-validation results.

    Args:
        cv_results: List of fold results
        model_name: Name of the model

    Returns:
        Summary statistics
    """
    df = pd.DataFrame(cv_results)

    summary = {
        'model': model_name,
        'cv_folds': len(cv_results)
    }

    # Calculate mean and std for each metric
    for col in df.columns:
        if col != 'fold' and pd.api.types.is_numeric_dtype(df[col]):
            summary[f'{col}_mean'] = df[col].mean()
            summary[f'{col}_std'] = df[col].std()

    return summary


def print_evaluation_report(
    results: Dict[str, Union[float, str]],
    title: str = "Model Evaluation Report"
) -> None:
    """
    Print formatted evaluation report.

    Args:
        results: Evaluation results dictionary
    """
    print(f"\n{'='*60}")
    print(f"{title}".center(60))
    print(f"{'='*60}")

    print(f"Model: {results.get('model', 'Unknown')}")

    # Regression metrics
    print(f"\n📊 REGRESSION METRICS:")
    print(f"  MAE:           ${results.get('mae', 'N/A'):.4f}")
    print(f"  RMSE:          ${results.get('rmse', 'N/A'):.4f}")
    print(f"  R²:            {results.get('r2', 'N/A'):.4f}")
    if 'mape' in results and not np.isnan(results['mape']):
        print(f"  MAPE:          {results['mape']:.2%}")

    # Directional metrics
    print(f"\n📈 DIRECTIONAL METRICS:")
    print(f"  Directional Acc: {results.get('directional_accuracy', 'N/A'):.1%}")
    if 'matthews_correlation' in results:
        print(f"  Matthews Corr:   {results['matthews_correlation']:.4f}")

    # Financial metrics
    financial_keys = ['sharpe_ratio', 'max_drawdown', 'win_rate', 'profit_factor']
    if any(k in results for k in financial_keys):
        print(f"\n💰 FINANCIAL METRICS:")
        if 'sharpe_ratio' in results:
            print(f"  Sharpe Ratio:   {results['sharpe_ratio']:.4f}")
        if 'max_drawdown' in results:
            print(f"  Max Drawdown:   {results['max_drawdown']:.2%}")
        if 'win_rate' in results:
            print(f"  Win Rate:       {results['win_rate']:.1%}")
        if 'profit_factor' in results:
            print(f"  Profit Factor:  {results['profit_factor']:.4f}")

    # CV metrics
    cv_keys = [k for k in results.keys() if k.endswith('_mean') or k.endswith('_std')]
    if cv_keys:
        print(f"\n🔄 CROSS-VALIDATION ({results.get('cv_folds', 'N/A')} folds):")
        for key in sorted(cv_keys):
            if key.endswith('_mean'):
                metric = key.replace('_mean', '')
                mean_val = results[key]
                std_val = results.get(key.replace('_mean', '_std'), 0)
                if metric in ['mae', 'rmse']:
                    print(f"  {metric.upper()}: ${mean_val:.4f} ± ${std_val:.4f}")
                else:
                    print(f"  {metric}: {mean_val:.4f} ± {std_val:.4f}")

    print(f"{'='*60}")


if __name__ == "__main__":
    # Example usage
    print("=" * 80)
    print("EVALUATION METRICS - EXAMPLE USAGE".center(80))
    print("=" * 80)

    # Create sample data
    np.random.seed(42)
    n_samples = 100

    # Simulate stock prices and predictions
    true_prices = 100 + np.random.randn(n_samples).cumsum()
    current_prices = true_prices + np.random.randn(n_samples) * 0.5
    pred_prices = true_prices + np.random.randn(n_samples) * 2

    # Basic regression metrics
    reg_results = regression_metrics(true_prices, pred_prices)
    print("\n📊 Regression Metrics:")
    for k, v in reg_results.items():
        print(f"  {k.upper()}: {v:.4f}")

    # Directional accuracy
    dir_results = directional_accuracy(true_prices, pred_prices, current_prices)
    print("\n📈 Directional Metrics:")
    for k, v in dir_results.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}")
        else:
            print(f"  {k}: {v}")

    # Financial metrics
    fin_results = financial_metrics(true_prices, pred_prices, current_prices)
    print("\n💰 Financial Metrics:")
    for k, v in fin_results.items():
        print(f"  {k}: {v:.4f}")

    # Comprehensive evaluation
    comp_results = comprehensive_evaluation(
        true_prices, pred_prices, current_prices, "Example Model"
    )
    print_evaluation_report(comp_results)

    print("\n" + "=" * 80)
    print("✅ Evaluation metrics example completed!")
    print("=" * 80)