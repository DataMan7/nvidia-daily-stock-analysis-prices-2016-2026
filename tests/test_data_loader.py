"""
Unit Tests for Data Loader
==========================

Tests for data loading, validation, and preprocessing.
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile

# Add project root to path
import sys
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data.loader import NVDADataLoader, DataQualityReport


class TestNVDADataLoader:
    """Test cases for NVDA data loader"""

    def test_init_with_valid_file(self):
        """Test initialization with valid file"""
        # Create temporary CSV file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write("Date,Open,High,Low,Close,Volume\n")
            f.write("2020-01-01,100,105,95,102,1000000\n")
            temp_path = f.name

        try:
            loader = NVDADataLoader(temp_path)
            assert loader.filepath == Path(temp_path)
            assert loader.date_column == 'Date'
        finally:
            Path(temp_path).unlink()

    def test_init_with_missing_file(self):
        """Test initialization with missing file raises error"""
        with pytest.raises(FileNotFoundError):
            NVDADataLoader("nonexistent_file.csv")

    def test_load_and_validate_clean_data(self):
        """Test loading and validation of clean data"""
        # Create clean test data
        dates = pd.date_range('2020-01-01', periods=10, freq='D')
        data = {
            'Date': dates,
            'Open': np.random.uniform(100, 110, 10),
            'High': np.random.uniform(105, 115, 10),
            'Low': np.random.uniform(95, 105, 10),
            'Close': np.random.uniform(100, 110, 10),
            'Volume': np.random.randint(1000000, 2000000, 10)
        }
        df = pd.DataFrame(data)

        # Ensure OHLC logic
        for i in range(len(df)):
            df.loc[i, 'High'] = max(df.loc[i, ['Open', 'High', 'Low', 'Close']])
            df.loc[i, 'Low'] = min(df.loc[i, ['Open', 'High', 'Low', 'Close']])

        # Save to temp file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            df.to_csv(f, index=False)
            temp_path = f.name

        try:
            loader = NVDADataLoader(temp_path)
            loaded_df, report = loader.load_and_validate(verbose=False)

            assert isinstance(loaded_df, pd.DataFrame)
            assert isinstance(report, DataQualityReport)
            assert report.is_valid
            assert len(loaded_df) == 10
            assert list(loaded_df.columns) == ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
            assert pd.api.types.is_datetime64_any_dtype(loaded_df['Date'])

        finally:
            Path(temp_path).unlink()

    def test_load_with_features(self):
        """Test loading with feature engineering"""
        # Create test data
        dates = pd.date_range('2020-01-01', periods=10, freq='D')
        data = {
            'Date': dates,
            'Open': [100, 102, 101, 103, 102, 104, 103, 105, 104, 106],
            'High': [105, 107, 106, 108, 107, 109, 108, 110, 109, 111],
            'Low': [95, 97, 96, 98, 97, 99, 98, 100, 99, 101],
            'Close': [102, 104, 103, 105, 104, 106, 105, 107, 106, 108],
            'Volume': [1000000] * 10
        }
        df = pd.DataFrame(data)

        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            df.to_csv(f, index=False)
            temp_path = f.name

        try:
            loader = NVDADataLoader(temp_path)
            df_with_features = loader.load_with_features()

            # Check that returns are calculated
            assert 'daily_return' in df_with_features.columns
            assert 'log_return' in df_with_features.columns

            # Check return calculations
            expected_returns = df['Close'].pct_change()
            pd.testing.assert_series_equal(
                df_with_features['daily_return'].iloc[1:],
                expected_returns.iloc[1:],
                check_names=False
            )

        finally:
            Path(temp_path).unlink()

    def test_validation_detects_missing_columns(self):
        """Test that validation detects missing required columns"""
        # Create data with missing columns
        dates = pd.date_range('2020-01-01', periods=5, freq='D')
        df = pd.DataFrame({
            'Date': dates,
            'Open': np.random.uniform(100, 110, 5),
            # Missing High, Low, Close, Volume
        })

        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            df.to_csv(f, index=False)
            temp_path = f.name

        try:
            loader = NVDADataLoader(temp_path)
            loaded_df, report = loader.load_and_validate(verbose=False)

            assert not report.is_valid
            assert 'Missing required columns' in str(report.issues)

        finally:
            Path(temp_path).unlink()

    def test_validation_detects_duplicate_dates(self):
        """Test that validation detects duplicate dates"""
        dates = ['2020-01-01', '2020-01-01', '2020-01-03']  # Duplicate first date
        data = {
            'Date': dates,
            'Open': [100, 101, 102],
            'High': [105, 106, 107],
            'Low': [95, 96, 97],
            'Close': [102, 103, 104],
            'Volume': [1000000, 1000001, 1000002]
        }
        df = pd.DataFrame(data)

        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            df.to_csv(f, index=False)
            temp_path = f.name

        try:
            loader = NVDADataLoader(temp_path)
            loaded_df, report = loader.load_and_validate(verbose=False)

            assert not report.is_valid
            assert any('duplicate' in issue.lower() for issue in report.issues)

        finally:
            Path(temp_path).unlink()


if __name__ == "__main__":
    pytest.main([__file__])