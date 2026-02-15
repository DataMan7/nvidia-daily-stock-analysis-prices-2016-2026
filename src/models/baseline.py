"""
Baseline Model Implementations
==============================

Simple baseline models for time-series forecasting.
These establish minimum performance benchmarks.

Author: Senior ML Engineer
Date: February 2026
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
from abc import ABC, abstractmethod
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings


class BaseModel(ABC):
    """Abstract base class for all models"""

    def __init__(self, name: str):
        self.name = name
        self.is_fitted = False

    @abstractmethod
    def fit(self, X: pd.DataFrame, y: pd.Series) -> 'BaseModel':
        """Fit the model"""
        pass

    @abstractmethod
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions"""
        pass

    def evaluate(self, y_true: pd.Series, y_pred: np.ndarray) -> Dict[str, float]:
        """Evaluate model performance"""
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        r2 = r2_score(y_true, y_pred)

        return {
            'mae': mae,
            'rmse': rmse,
            'r2': r2
        }


class NaiveModel(BaseModel):
    """
    Naive Forecast Model

    Prediction: Tomorrow's price = Today's price
    This is the simplest possible forecast and often surprisingly effective.
    """

    def __init__(self):
        super().__init__("Naive")
        self.last_value = None

    def fit(self, X: pd.DataFrame, y: pd.Series) -> 'NaiveModel':
        """Fit model by storing the last known value"""
        if len(y) == 0:
            raise ValueError("Cannot fit on empty data")

        self.last_value = y.iloc[-1]
        self.is_fitted = True
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict using the last fitted value"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")

        return np.full(len(X), self.last_value)


class MovingAverageModel(BaseModel):
    """
    Moving Average Forecast Model

    Prediction: Tomorrow's price = Average of last N days
    """

    def __init__(self, window: int = 7):
        super().__init__(f"MovingAverage_{window}")
        self.window = window
        self.last_values = None

    def fit(self, X: pd.DataFrame, y: pd.Series) -> 'MovingAverageModel':
        """Fit model by storing the last window of values"""
        if len(y) < self.window:
            raise ValueError(f"Need at least {self.window} samples for fitting")

        self.last_values = y.iloc[-self.window:].values
        self.is_fitted = True
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict using moving average of last fitted values"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")

        return np.full(len(X), np.mean(self.last_values))


class SeasonalNaiveModel(BaseModel):
    """
    Seasonal Naive Model

    Prediction: Tomorrow's price = Same day last week
    Exploits weekly patterns in stock prices.
    """

    def __init__(self, seasonal_period: int = 5):  # 5 trading days = 1 week
        super().__init__(f"SeasonalNaive_{seasonal_period}")
        self.seasonal_period = seasonal_period
        self.seasonal_values = {}

    def fit(self, X: pd.DataFrame, y: pd.Series) -> 'SeasonalNaiveModel':
        """Fit model by storing seasonal patterns"""
        if len(y) < self.seasonal_period:
            raise ValueError(f"Need at least {self.seasonal_period} samples for fitting")

        # Store values by their position in the seasonal cycle
        for i in range(len(y)):
            position = i % self.seasonal_period
            if position not in self.seasonal_values:
                self.seasonal_values[position] = []
            self.seasonal_values[position].append(y.iloc[i])

        self.is_fitted = True
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict using seasonal pattern"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")

        predictions = []
        for i in range(len(X)):
            position = (len(self.seasonal_values) + i) % self.seasonal_period
            if position in self.seasonal_values:
                # Use the most recent value for this seasonal position
                predictions.append(self.seasonal_values[position][-1])
            else:
                # Fallback to overall mean
                all_values = [v for values in self.seasonal_values.values() for v in values]
                predictions.append(np.mean(all_values))

        return np.array(predictions)


class EnsembleBaselineModel(BaseModel):
    """
    Ensemble of all baseline models

    Combines predictions from Naive, Moving Average, and Seasonal Naive models.
    """

    def __init__(self, weights: Optional[Dict[str, float]] = None):
        super().__init__("EnsembleBaseline")
        self.weights = weights or {'naive': 0.4, 'ma': 0.3, 'seasonal': 0.3}
        self.models = {}

    def fit(self, X: pd.DataFrame, y: pd.Series) -> 'EnsembleBaselineModel':
        """Fit all baseline models"""
        self.models['naive'] = NaiveModel().fit(X, y)
        self.models['ma'] = MovingAverageModel().fit(X, y)
        self.models['seasonal'] = SeasonalNaiveModel().fit(X, y)
        self.is_fitted = True
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Weighted ensemble prediction"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")

        predictions = []
        for model_name, model in self.models.items():
            pred = model.predict(X)
            weight = self.weights.get(model_name, 1.0 / len(self.models))
            predictions.append(pred * weight)

        return np.sum(predictions, axis=0)


def create_baseline_models() -> Dict[str, BaseModel]:
    """
    Factory function to create all baseline models

    Returns:
        Dictionary of model name -> model instance
    """
    return {
        'naive': NaiveModel(),
        'moving_average_7': MovingAverageModel(window=7),
        'moving_average_30': MovingAverageModel(window=30),
        'seasonal_naive': SeasonalNaiveModel(),
        'ensemble': EnsembleBaselineModel()
    }


def evaluate_baselines_on_data(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series
) -> pd.DataFrame:
    """
    Evaluate all baseline models on provided data

    Returns:
        DataFrame with model performance metrics
    """
    models = create_baseline_models()
    results = []

    for model_name, model in models.items():
        try:
            # Fit and predict
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            # Evaluate
            metrics = model.evaluate(y_test, y_pred)
            metrics['model'] = model_name

            # Add directional accuracy
            actual_direction = (y_test > X_test.get('Close', y_test.shift(1))).astype(int)
            pred_direction = (y_pred > X_test.get('Close', y_test.shift(1))).astype(int)
            metrics['directional_accuracy'] = (actual_direction == pred_direction).mean()

            results.append(metrics)

        except Exception as e:
            warnings.warn(f"Failed to evaluate {model_name}: {e}")
            continue

    return pd.DataFrame(results).set_index('model')


if __name__ == "__main__":
    # Example usage
    print("=" * 80)
    print("BASELINE MODELS - EXAMPLE USAGE".center(80))
    print("=" * 80)

    # Create sample data
    np.random.seed(42)
    dates = pd.date_range('2020-01-01', periods=100, freq='D')
    prices = 100 + np.random.randn(100).cumsum()

    df = pd.DataFrame({
        'Date': dates,
        'Close': prices,
        'Volume': np.random.randint(1000000, 10000000, 100)
    })

    # Create target (next day's price)
    df['target'] = df['Close'].shift(-1)
    df = df.dropna()

    # Split data
    split_idx = int(len(df) * 0.8)
    train_df = df.iloc[:split_idx]
    test_df = df.iloc[split_idx:]

    # Evaluate baselines
    results = evaluate_baselines_on_data(
        train_df[['Close', 'Volume']],
        train_df['target'],
        test_df[['Close', 'Volume']],
        test_df['target']
    )

    print("\nBaseline Model Performance:")
    print("=" * 50)
    print(results.round(4))

    print("\n" + "=" * 80)
    print("✅ Baseline models example completed!")
    print("=" * 80)