"""
Models Module
=============

Production-ready model implementations for time-series forecasting.

Available Models:
- Baseline models (Naive, Moving Average, Seasonal Naive, Ensemble)
- XGBoost model with time-series optimizations

Author: Senior ML Engineer
Date: February 2026
"""

from .baseline import (
    BaseModel,
    NaiveModel,
    MovingAverageModel,
    SeasonalNaiveModel,
    EnsembleBaselineModel,
    create_baseline_models,
    evaluate_baselines_on_data
)

from .xgboost_model import (
    XGBoostTimeSeriesModel,
    optimize_hyperparameters
)

__all__ = [
    # Base classes
    'BaseModel',

    # Baseline models
    'NaiveModel',
    'MovingAverageModel',
    'SeasonalNaiveModel',
    'EnsembleBaselineModel',
    'create_baseline_models',
    'evaluate_baselines_on_data',

    # XGBoost models
    'XGBoostTimeSeriesModel',
    'optimize_hyperparameters'
]

__version__ = "1.0.0"