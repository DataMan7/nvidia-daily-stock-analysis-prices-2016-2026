"""
Utilities Module
================

Helper functions for model management, tuning, and persistence.

Components:
- model_utils: Model saving/loading and registry
- hyperparameter_tuning: Automated parameter optimization

Author: Senior ML Engineer
Date: February 2026
"""

from .model_utils import (
    ModelRegistry,
    save_model,
    load_model,
    compare_models,
    cleanup_old_models
)

from .hyperparameter_tuning import (
    HyperparameterTuner,
    create_param_space_for_xgboost,
    create_param_space_for_random_forest,
    optimize_model_hyperparameters
)

__all__ = [
    # Model utilities
    'ModelRegistry',
    'save_model',
    'load_model',
    'compare_models',
    'cleanup_old_models',

    # Hyperparameter tuning
    'HyperparameterTuner',
    'create_param_space_for_xgboost',
    'create_param_space_for_random_forest',
    'optimize_model_hyperparameters'
]

__version__ = "1.0.0"