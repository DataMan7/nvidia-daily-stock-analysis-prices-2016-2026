"""
Test Suite
==========

Unit tests for the NVIDIA stock analysis package.

Run all tests:
    python -m pytest tests/

Run specific test:
    python -m pytest tests/test_data_loader.py

Run with coverage:
    python -m pytest --cov=src tests/

Author: Senior ML Engineer
Date: February 2026
"""

# Test configuration
import pytest

# Set up test environment
pytest_plugins = []

# Common test fixtures can be defined here
import pandas as pd
import numpy as np

@pytest.fixture
def sample_stock_data():
    """Sample stock data for testing"""
    dates = pd.date_range('2020-01-01', periods=100, freq='D')
    np.random.seed(42)

    data = {
        'Date': dates,
        'Open': 100 + np.random.randn(100).cumsum(),
        'High': 105 + np.random.randn(100).cumsum(),
        'Low': 95 + np.random.randn(100).cumsum(),
        'Close': 102 + np.random.randn(100).cumsum(),
        'Volume': np.random.randint(1000000, 5000000, 100)
    }

    return pd.DataFrame(data)

@pytest.fixture
def sample_features():
    """Sample feature data for testing"""
    np.random.seed(42)
    n_samples = 50

    data = {
        'lag_1_close': np.random.randn(n_samples),
        'lag_1_volume': np.random.randint(1000000, 5000000, n_samples),
        'ma_7': np.random.randn(n_samples),
        'rsi': np.random.uniform(0, 100, n_samples),
        'bb_upper': np.random.randn(n_samples),
        'bb_lower': np.random.randn(n_samples)
    }

    return pd.DataFrame(data)

__version__ = "1.0.0"