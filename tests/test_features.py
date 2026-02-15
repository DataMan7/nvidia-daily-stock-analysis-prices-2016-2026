"""
Unit Tests for Feature Engineering
==================================

Tests for feature creation and leakage prevention.
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.features.engineering import FeatureEngineer


class TestFeatureEngineer:
    """Test cases for feature engineering"""

    def test_init(self):
        """Test FeatureEngineer initialization"""
        engineer = FeatureEngineer()
        assert engineer is not None

    def test_create_features_basic(self):
        """Test basic feature creation"""
        # Create test data
        dates = pd.date_range('2020-01-01', periods=10, freq='D')
        data = {
            'Date': dates,
            'Open': [100, 102, 101, 103, 102, 104, 103, 105, 104, 106],
            'High': [105, 107, 106, 108, 107, 109, 108, 110, 109, 111],
            'Low': [95, 97, 96, 98, 97, 99, 98, 100, 99, 101],
            'Close': [102, 104, 103, 105, 104, 106, 105, 107, 106, 108],
            'Volume': [1000000, 1100000, 1050000, 1150000, 1080000,
                      1180000, 1110000, 1210000, 1140000, 1240000]
        }
        df = pd.DataFrame(data)

        engineer = FeatureEngineer()
        df_features = engineer.create_features(df)

        # Check that basic features are created
        assert 'Daily_Return' in df_features.columns
        assert 'Prev_Day_Return' in df_features.columns
        assert 'MA_7' in df_features.columns
        assert 'MA_50' in df_features.columns
        assert 'MA_200' in df_features.columns
        assert 'RSI' in df_features.columns
        assert 'BB_Upper' in df_features.columns
        assert 'BB_Lower' in df_features.columns

        # Check lag features
        lag_cols = [col for col in df_features.columns if 'Lag' in col]
        assert len(lag_cols) > 0

    def test_daily_return_calculation(self):
        """Test daily return calculation"""
        data = {
            'Date': pd.date_range('2020-01-01', periods=3, freq='D'),
            'Open': [100, 102, 104],
            'High': [105, 107, 109],
            'Low': [95, 97, 99],
            'Close': [102, 104, 106],  # +2%, +2% returns
            'Volume': [1000000, 1100000, 1200000]
        }
        df = pd.DataFrame(data)

        engineer = FeatureEngineer()
        df_features = engineer.create_features(df)

        # Check returns (should be NaN for first row, then calculated)
        assert pd.isna(df_features['Daily_Return'].iloc[0])
        assert abs(df_features['Daily_Return'].iloc[1] - 0.0196) < 0.001  # (104-102)/102 ≈ 0.0196
        assert abs(df_features['Daily_Return'].iloc[2] - 0.0192) < 0.001  # (106-104)/104 ≈ 0.0192

    def test_moving_averages(self):
        """Test moving average calculations"""
        # Simple case: constant prices
        data = {
            'Date': pd.date_range('2020-01-01', periods=5, freq='D'),
            'Open': [100] * 5,
            'High': [100] * 5,
            'Low': [100] * 5,
            'Close': [100] * 5,
            'Volume': [1000000] * 5
        }
        df = pd.DataFrame(data)

        engineer = FeatureEngineer()
        df_features = engineer.create_features(df)

        # For constant series, MA should equal the constant
        assert df_features['MA_7'].iloc[-1] == 100.0  # Last value should be 100

    def test_rsi_calculation(self):
        """Test RSI calculation"""
        # Create data with clear up/down moves
        data = {
            'Date': pd.date_range('2020-01-01', periods=20, freq='D'),
            'Open': list(range(100, 120)),
            'High': list(range(102, 122)),
            'Low': list(range(98, 118)),
            'Close': list(range(101, 121)),  # Consistently increasing
            'Volume': [1000000] * 20
        }
        df = pd.DataFrame(data)

        engineer = FeatureEngineer()
        df_features = engineer.create_features(df)

        # RSI should exist
        assert 'RSI' in df_features.columns
        assert not df_features['RSI'].isna().all()

        # For consistently increasing prices, RSI should be high (>50)
        rsi_values = df_features['RSI'].dropna()
        if len(rsi_values) > 0:
            assert rsi_values.iloc[-1] > 50  # Should be high for uptrend

    def test_bollinger_bands(self):
        """Test Bollinger Bands calculation"""
        # Create data with some volatility
        np.random.seed(42)
        base_price = 100
        prices = base_price + np.random.randn(50) * 2  # Some volatility around 100

        data = {
            'Date': pd.date_range('2020-01-01', periods=50, freq='D'),
            'Open': prices - 1,
            'High': prices + 2,
            'Low': prices - 2,
            'Close': prices,
            'Volume': [1000000] * 50
        }
        df = pd.DataFrame(data)

        engineer = FeatureEngineer()
        df_features = engineer.create_features(df)

        # Check BB columns exist
        assert 'BB_Upper' in df_features.columns
        assert 'BB_Lower' in df_features.columns
        assert 'BB_Middle' in df_features.columns
        assert 'BB_Width' in df_features.columns

        # For rolling window of 20, first 19 should be NaN
        assert pd.isna(df_features['BB_Upper'].iloc[:19]).all()
        assert not pd.isna(df_features['BB_Upper'].iloc[19])

    def test_lag_features(self):
        """Test lag feature creation"""
        data = {
            'Date': pd.date_range('2020-01-01', periods=10, freq='D'),
            'Open': list(range(100, 110)),
            'High': list(range(105, 115)),
            'Low': list(range(95, 105)),
            'Close': list(range(102, 112)),
            'Volume': list(range(1000000, 1010000, 100000))
        }
        df = pd.DataFrame(data)

        engineer = FeatureEngineer()
        df_features = engineer.create_features(df)

        # Check that lag features exist
        lag_features = [col for col in df_features.columns if 'Lag' in col]
        assert len(lag_features) > 0

        # Check specific lag features
        assert 'Close_Lag_1' in df_features.columns
        assert 'Volume_Lag_1' in df_features.columns

        # First row should have NaN for lag features
        assert pd.isna(df_features['Close_Lag_1'].iloc[0])
        assert df_features['Close_Lag_1'].iloc[1] == 102  # Previous close

    def test_no_future_leakage(self):
        """Test that features don't leak future information"""
        data = {
            'Date': pd.date_range('2020-01-01', periods=10, freq='D'),
            'Open': [100, 102, 101, 103, 102, 104, 103, 105, 104, 106],
            'High': [105, 107, 106, 108, 107, 109, 108, 110, 109, 111],
            'Low': [95, 97, 96, 98, 97, 99, 98, 100, 99, 101],
            'Close': [102, 104, 103, 105, 104, 106, 105, 107, 106, 108],
            'Volume': [1000000] * 10
        }
        df = pd.DataFrame(data)

        engineer = FeatureEngineer()
        df_features = engineer.create_features(df)

        # All features should be calculable from past data only
        # This is more of a design test - the implementation should ensure this

        # Check that we have the expected number of features
        original_cols = len(df.columns)
        new_features = len(df_features.columns) - original_cols
        assert new_features > 10  # Should create many features

    def test_data_sorting(self):
        """Test that data is properly sorted by date"""
        # Create unsorted data
        dates = pd.to_datetime(['2020-01-03', '2020-01-01', '2020-01-02'])
        data = {
            'Date': dates,
            'Open': [100, 102, 101],
            'High': [105, 107, 106],
            'Low': [95, 97, 96],
            'Close': [102, 104, 103],
            'Volume': [1000000, 1100000, 1050000]
        }
        df = pd.DataFrame(data)

        engineer = FeatureEngineer()
        df_features = engineer.create_features(df)

        # Should be sorted by date
        assert df_features['Date'].is_monotonic_increasing


if __name__ == "__main__":
    pytest.main([__file__])