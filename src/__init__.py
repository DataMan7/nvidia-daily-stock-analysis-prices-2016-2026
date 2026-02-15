"""
NVIDIA Stock Analysis Package
=============================

A production-ready ML pipeline for time-series forecasting.

Modules:
- data: Data loading and validation
- features: Feature engineering (leakage-safe)
- models: Model implementations (baselines, XGBoost)
- evaluation: Performance metrics, backtesting, CV
- utils: Utilities for persistence, tuning, etc.

Author: Senior ML Engineer
Date: February 2026
"""

__version__ = "1.0.0"
__author__ = "Senior ML Engineer"

# Import main modules for easy access
from . import data
from . import features
from . import models
from . import evaluation
from . import utils

__all__ = [
    'data',
    'features',
    'models',
    'evaluation',
    'utils'
]