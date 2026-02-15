"""
Cross-Validation Utilities for Time-Series
==========================================

Proper cross-validation strategies for time-series data.
Prevents data leakage while maintaining temporal order.

Author: Senior ML Engineer
Date: February 2026
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Iterator, Callable
from sklearn.model_selection import TimeSeriesSplit
from sklearn.base import BaseEstimator
import warnings


class TimeSeriesCrossValidator:
    """
    Advanced time-series cross-validation with multiple strategies.

    Supports:
    - Walk-forward validation
    - Rolling window validation
    - Anchored window validation
    - Purged validation (for financial data)
    """

    def __init__(
        self,
        n_splits: int = 5,
        test_size: Optional[int] = None,
        gap: int = 0,
        max_train_size: Optional[int] = None,
        strategy: str = 'walk_forward'
    ):
        """
        Initialize time-series cross-validator.

        Args:
            n_splits: Number of CV folds
            test_size: Size of test set for each fold
            gap: Gap between train and test sets (to prevent leakage)
            max_train_size: Maximum training set size (for rolling windows)
            strategy: CV strategy ('walk_forward', 'rolling', 'anchored')
        """
        self.n_splits = n_splits
        self.test_size = test_size
        self.gap = gap
        self.max_train_size = max_train_size
        self.strategy = strategy

        if strategy not in ['walk_forward', 'rolling', 'anchored']:
            raise ValueError(f"Unknown strategy: {strategy}")

    def split(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        """
        Generate train/test indices for cross-validation.

        Args:
            X: Feature matrix
            y: Target vector (optional)

        Yields:
            Tuple of (train_indices, test_indices)
        """
        n_samples = len(X)

        if self.strategy == 'walk_forward':
            yield from self._walk_forward_split(n_samples)
        elif self.strategy == 'rolling':
            yield from self._rolling_split(n_samples)
        elif self.strategy == 'anchored':
            yield from self._anchored_split(n_samples)

    def _walk_forward_split(self, n_samples: int) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        """Standard walk-forward validation"""
        tscv = TimeSeriesSplit(
            n_splits=self.n_splits,
            test_size=self.test_size,
            gap=self.gap,
            max_train_size=self.max_train_size
        )

        indices = np.arange(n_samples)
        for train_idx, test_idx in tscv.split(indices):
            yield train_idx, test_idx

    def _rolling_split(self, n_samples: int) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        """Rolling window validation with fixed training size"""
        if self.max_train_size is None:
            raise ValueError("max_train_size required for rolling strategy")

        test_size = self.test_size or n_samples // (self.n_splits + 1)

        for i in range(self.n_splits):
            # Calculate test set start
            test_start = n_samples - (self.n_splits - i) * test_size
            test_end = min(test_start + test_size, n_samples)

            # Training set: most recent max_train_size samples before test
            train_end = max(0, test_start - self.gap)
            train_start = max(0, train_end - self.max_train_size)

            train_idx = np.arange(train_start, train_end)
            test_idx = np.arange(test_start, test_end)

            if len(train_idx) > 0 and len(test_idx) > 0:
                yield train_idx, test_idx

    def _anchored_split(self, n_samples: int) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        """Anchored window validation (fixed start, expanding end)"""
        test_size = self.test_size or n_samples // (self.n_splits + 1)

        for i in range(self.n_splits):
            # Test set grows from the end
            test_start = n_samples - (i + 1) * test_size
            test_end = n_samples - i * test_size

            # Training set: from beginning up to test start
            train_end = max(0, test_start - self.gap)
            train_start = 0

            train_idx = np.arange(train_start, train_end)
            test_idx = np.arange(test_start, test_end)

            if len(train_idx) > 0 and len(test_idx) > 0:
                yield train_idx, test_idx


class PurgedTimeSeriesSplit:
    """
    Purged cross-validation for financial time-series.

    Removes observations within a purge window around test periods
    to prevent any information leakage from correlated features.
    """

    def __init__(
        self,
        n_splits: int = 5,
        test_size: int = 1,
        purge_window: int = 1,
        embargo: int = 0
    ):
        """
        Initialize purged cross-validation.

        Args:
            n_splits: Number of CV folds
            test_size: Size of each test set
            purge_window: Number of periods to purge around test sets
            embargo: Additional embargo period after test sets
        """
        self.n_splits = n_splits
        self.test_size = test_size
        self.purge_window = purge_window
        self.embargo = embargo

    def split(self, X: pd.DataFrame) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        """
        Generate purged train/test splits.

        Args:
            X: Feature matrix

        Yields:
            Tuple of (train_indices, test_indices)
        """
        n_samples = len(X)

        for i in range(self.n_splits):
            # Calculate test set boundaries
            test_end = n_samples - i * self.test_size
            test_start = test_end - self.test_size

            if test_start < 0:
                continue

            test_idx = np.arange(test_start, test_end)

            # Purge window around test set
            purge_start = max(0, test_start - self.purge_window)
            purge_end = min(n_samples, test_end + self.purge_window + self.embargo)

            # Training set: everything except purged regions
            all_idx = np.arange(n_samples)
            purged_idx = np.concatenate([
                np.arange(purge_start, test_start),  # Before test
                test_idx,  # Test set
                np.arange(test_end, purge_end)  # After test
            ])

            train_idx = np.setdiff1d(all_idx, purged_idx)

            if len(train_idx) > 0:
                yield train_idx, test_idx


def cross_validate_model(
    model: BaseEstimator,
    X: pd.DataFrame,
    y: pd.Series,
    cv_strategy: str = 'walk_forward',
    cv_params: Optional[Dict] = None,
    scoring_func: Optional[Callable] = None
) -> Dict[str, List[float]]:
    """
    Cross-validate a model with time-series aware splitting.

    Args:
        model: Scikit-learn compatible model
        X: Feature matrix
        y: Target vector
        cv_strategy: CV strategy ('walk_forward', 'rolling', 'anchored', 'purged')
        cv_params: Parameters for CV strategy
        scoring_func: Custom scoring function

    Returns:
        Dictionary with CV results
    """
    cv_params = cv_params or {}

    if cv_strategy == 'purged':
        cv = PurgedTimeSeriesSplit(**cv_params)
    else:
        cv_params['strategy'] = cv_strategy
        cv = TimeSeriesCrossValidator(**cv_params)

    scores = []
    fold_info = []

    for fold_idx, (train_idx, test_idx) in enumerate(cv.split(X)):
        # Split data
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        # Train model
        model_clone = type(model)(**model.get_params())
        model_clone.fit(X_train, y_train)

        # Score
        if scoring_func:
            score = scoring_func(model_clone, X_test, y_test)
        else:
            # Default: R² score
            y_pred = model_clone.predict(X_test)
            score = r2_score(y_test, y_pred)

        scores.append(score)

        fold_info.append({
            'fold': fold_idx + 1,
            'train_size': len(train_idx),
            'test_size': len(test_idx),
            'score': score
        })

    return {
        'scores': scores,
        'mean_score': np.mean(scores),
        'std_score': np.std(scores),
        'fold_info': fold_info
    }


def evaluate_cv_stability(cv_results: Dict[str, List[float]]) -> Dict[str, float]:
    """
    Evaluate cross-validation stability and reliability.

    Args:
        cv_results: Results from cross_validate_model

    Returns:
        Stability metrics
    """
    scores = cv_results['scores']

    return {
        'mean_score': np.mean(scores),
        'std_score': np.std(scores),
        'cv_stability': 1 - np.std(scores) / abs(np.mean(scores)) if np.mean(scores) != 0 else 0,
        'min_score': np.min(scores),
        'max_score': np.max(scores),
        'score_range': np.max(scores) - np.min(scores)
    }


def plot_cv_results(cv_results: Dict, title: str = "Cross-Validation Results"):
    """
    Plot cross-validation results.

    Args:
        cv_results: Results from cross_validate_model
    """
    try:
        import matplotlib.pyplot as plt

        fold_info = cv_results['fold_info']
        folds = [f['fold'] for f in fold_info]
        scores = [f['score'] for f in fold_info]

        plt.figure(figsize=(10, 6))
        plt.plot(folds, scores, 'bo-', linewidth=2, markersize=8)
        plt.axhline(y=cv_results['mean_score'], color='r', linestyle='--', label=f'Mean: {cv_results["mean_score"]:.4f}')
        plt.fill_between(folds,
                        cv_results['mean_score'] - cv_results['std_score'],
                        cv_results['mean_score'] + cv_results['std_score'],
                        alpha=0.2, color='r', label=f'±1 STD: {cv_results["std_score"]:.4f}')

        plt.title(title)
        plt.xlabel('Fold')
        plt.ylabel('Score')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

    except ImportError:
        warnings.warn("matplotlib not available for plotting")


if __name__ == "__main__":
    # Example usage
    print("=" * 80)
    print("TIME-SERIES CROSS-VALIDATION - EXAMPLE USAGE".center(80))
    print("=" * 80)

    # Create sample time-series data
    np.random.seed(42)
    dates = pd.date_range('2020-01-01', periods=200, freq='D')
    X = pd.DataFrame({
        'feature1': np.random.randn(200).cumsum(),
        'feature2': np.random.randn(200),
        'feature3': np.sin(np.arange(200) * 0.1)
    })
    y = X['feature1'] + np.random.randn(200) * 0.1

    print(f"Dataset: {len(X)} samples, {len(X.columns)} features")

    # Test different CV strategies
    strategies = ['walk_forward', 'rolling', 'anchored']

    for strategy in strategies:
        print(f"\n🔄 Testing {strategy.upper()} strategy:")

        if strategy == 'rolling':
            cv_params = {'max_train_size': 100}
        else:
            cv_params = {}

        cv = TimeSeriesCrossValidator(n_splits=5, strategy=strategy, **cv_params)

        fold_count = 0
        for train_idx, test_idx in cv.split(X):
            fold_count += 1
            print(f"  Fold {fold_count}: Train {len(train_idx)}, Test {len(test_idx)}")

        print(f"  Total folds generated: {fold_count}")

    # Test purged CV
    print(f"\n🔄 Testing PURGED strategy:")
    purged_cv = PurgedTimeSeriesSplit(n_splits=5, test_size=10, purge_window=5)

    fold_count = 0
    for train_idx, test_idx in purged_cv.split(X):
        fold_count += 1
        print(f"  Fold {fold_count}: Train {len(train_idx)}, Test {len(test_idx)}")

    print(f"  Total folds generated: {fold_count}")

    print("\n" + "=" * 80)
    print("✅ Cross-validation example completed!")
    print("=" * 80)