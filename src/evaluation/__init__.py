"""
Evaluation Module
=================

Comprehensive evaluation tools for time-series models.

Components:
- validators: Data leakage detection
- metrics: Performance evaluation metrics
- backtesting: Financial backtesting framework
- cross_validation: Time-series aware CV strategies

Author: Senior ML Engineer
Date: February 2026
"""

from .validators import (
    DataLeakageDetector,
    LeakageReport,
    deliberately_introduce_leakage
)

from .metrics import (
    regression_metrics,
    directional_accuracy,
    financial_metrics,
    comprehensive_evaluation,
    compare_models,
    cross_validation_summary,
    print_evaluation_report
)

from .backtesting import (
    Backtester,
    backtest_predictions,
    compare_strategies,
    print_backtest_report
)

from .cross_validation import (
    TimeSeriesCrossValidator,
    PurgedTimeSeriesSplit,
    cross_validate_model,
    evaluate_cv_stability,
    plot_cv_results
)

__all__ = [
    # Leakage detection
    'DataLeakageDetector',
    'LeakageReport',
    'deliberately_introduce_leakage',

    # Performance metrics
    'regression_metrics',
    'directional_accuracy',
    'financial_metrics',
    'comprehensive_evaluation',
    'compare_models',
    'cross_validation_summary',
    'print_evaluation_report',

    # Backtesting
    'Backtester',
    'backtest_predictions',
    'compare_strategies',
    'print_backtest_report',

    # Cross-validation
    'TimeSeriesCrossValidator',
    'PurgedTimeSeriesSplit',
    'cross_validate_model',
    'evaluate_cv_stability',
    'plot_cv_results'
]

__version__ = "1.0.0"