"""
Data Loading and Validation Module
===================================

This module handles data loading with built-in validation checks.
Ensures data quality BEFORE any analysis begins.

Author: Senior ML Engineer
Date: February 2026
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Tuple, Dict
from dataclasses import dataclass
import warnings


@dataclass
class DataQualityReport:
    """Container for data quality metrics"""
    n_rows: int
    n_columns: int
    missing_values: Dict[str, int]
    duplicate_rows: int
    date_range: Tuple[str, str]
    data_types: Dict[str, str]
    is_valid: bool
    issues: list


class NVDADataLoader:
    """
    Professional data loader for NVIDIA stock data.
    
    Features:
    - Automatic data validation
    - Type conversion
    - Missing value detection
    - Temporal ordering verification
    - Data quality reporting
    
    Example:
        >>> loader = NVDADataLoader('data/raw/NVDA_yfinance_clean.csv')
        >>> df, report = loader.load_and_validate()
        >>> if report.is_valid:
        >>>     print("Data is clean!")
    """
    
    REQUIRED_COLUMNS = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
    NUMERIC_COLUMNS = ['Open', 'High', 'Low', 'Close', 'Volume']
    
    def __init__(self, filepath: str, date_column: str = 'Date'):
        """
        Initialize data loader.
        
        Args:
            filepath: Path to CSV file
            date_column: Name of date column
        """
        self.filepath = Path(filepath)
        self.date_column = date_column
        
        if not self.filepath.exists():
            raise FileNotFoundError(f"Data file not found: {self.filepath}")
    
    def load_and_validate(
        self,
        validate: bool = True,
        verbose: bool = True
    ) -> Tuple[pd.DataFrame, Optional[DataQualityReport]]:
        """
        Load data with optional validation.
        
        Args:
            validate: Run validation checks
            verbose: Print validation messages
        
        Returns:
            Tuple of (DataFrame, DataQualityReport)
        """
        # Load data
        df = pd.read_csv(self.filepath)
        
        if verbose:
            print(f"✅ Loaded {len(df):,} rows from {self.filepath.name}")
        
        # Convert date column
        df[self.date_column] = pd.to_datetime(df[self.date_column])
        
        # Sort by date
        df = df.sort_values(self.date_column).reset_index(drop=True)
        
        # Convert numeric columns
        for col in self.NUMERIC_COLUMNS:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Validate if requested
        report = None
        if validate:
            report = self._validate_data(df, verbose=verbose)
        
        return df, report
    
    def _validate_data(
        self,
        df: pd.DataFrame,
        verbose: bool = True
    ) -> DataQualityReport:
        """
        Comprehensive data validation.
        
        Checks:
        1. Required columns present
        2. No missing values in critical columns
        3. Proper data types
        4. Temporal ordering
        5. Reasonable value ranges
        6. Duplicate detection
        """
        issues = []
        
        # Check 1: Required columns
        missing_cols = set(self.REQUIRED_COLUMNS) - set(df.columns)
        if missing_cols:
            issues.append(f"Missing required columns: {missing_cols}")
        
        # Check 2: Missing values
        missing_values = df.isnull().sum().to_dict()
        critical_missing = {k: v for k, v in missing_values.items() if v > 0 and k in self.REQUIRED_COLUMNS}
        if critical_missing:
            issues.append(f"Missing values in critical columns: {critical_missing}")
        
        # Check 3: Data types
        data_types = df.dtypes.astype(str).to_dict()
        if self.date_column in df.columns:
            if not pd.api.types.is_datetime64_any_dtype(df[self.date_column]):
                issues.append(f"Date column '{self.date_column}' is not datetime type")
        
        # Check 4: Temporal ordering
        if self.date_column in df.columns:
            if not df[self.date_column].is_monotonic_increasing:
                issues.append("Data is not sorted by date")
        
        # Check 5: Value ranges (for stock prices)
        for col in ['Open', 'High', 'Low', 'Close']:
            if col in df.columns:
                if (df[col] < 0).any():
                    issues.append(f"Negative values found in {col}")
                if df[col].min() == df[col].max():
                    issues.append(f"Column {col} has constant value")
        
        # Check 6: OHLC logic
        if all(col in df.columns for col in ['Open', 'High', 'Low', 'Close']):
            invalid_high = (df['High'] < df[['Open', 'Low', 'Close']].max(axis=1)).any()
            invalid_low = (df['Low'] > df[['Open', 'High', 'Close']].min(axis=1)).any()
            
            if invalid_high:
                issues.append("High prices are lower than Open/Low/Close (data integrity issue)")
            if invalid_low:
                issues.append("Low prices are higher than Open/High/Close (data integrity issue)")
        
        # Check 7: Duplicates
        duplicate_rows = df.duplicated().sum()
        if duplicate_rows > 0:
            issues.append(f"Found {duplicate_rows} duplicate rows")
        
        # Check 8: Date duplicates
        if self.date_column in df.columns:
            duplicate_dates = df[self.date_column].duplicated().sum()
            if duplicate_dates > 0:
                issues.append(f"Found {duplicate_dates} duplicate dates")
        
        # Generate report
        date_range = (
            str(df[self.date_column].min().date()) if self.date_column in df.columns else "N/A",
            str(df[self.date_column].max().date()) if self.date_column in df.columns else "N/A"
        )
        
        report = DataQualityReport(
            n_rows=len(df),
            n_columns=len(df.columns),
            missing_values=missing_values,
            duplicate_rows=int(duplicate_rows),
            date_range=date_range,
            data_types=data_types,
            is_valid=len(issues) == 0,
            issues=issues
        )
        
        if verbose:
            self._print_quality_report(report)
        
        return report
    
    def _print_quality_report(self, report: DataQualityReport):
        """Print formatted quality report"""
        print("\n" + "=" * 80)
        print("📊 DATA QUALITY REPORT".center(80))
        print("=" * 80)
        
        print(f"\n📏 Dimensions: {report.n_rows:,} rows × {report.n_columns} columns")
        print(f"📅 Date Range: {report.date_range[0]} to {report.date_range[1]}")
        print(f"🔢 Duplicate Rows: {report.duplicate_rows}")
        
        # Missing values
        total_missing = sum(report.missing_values.values())
        print(f"❓ Total Missing Values: {total_missing}")
        
        if total_missing > 0:
            print("\nMissing values by column:")
            for col, count in report.missing_values.items():
                if count > 0:
                    pct = (count / report.n_rows) * 100
                    print(f"  • {col}: {count} ({pct:.2f}%)")
        
        # Validation status
        print("\n" + "-" * 80)
        if report.is_valid:
            print("✅ VALIDATION PASSED - Data quality is excellent!")
        else:
            print("⚠️  VALIDATION FAILED - Issues found:")
            for i, issue in enumerate(report.issues, 1):
                print(f"  {i}. {issue}")
        
        print("=" * 80 + "\n")
    
    def load_with_features(
        self,
        add_returns: bool = True,
        add_log_returns: bool = True
    ) -> pd.DataFrame:
        """
        Load data with basic feature engineering.
        
        Args:
            add_returns: Add daily returns
            add_log_returns: Add log returns
        
        Returns:
            DataFrame with additional features
        """
        df, _ = self.load_and_validate(verbose=False)
        
        if add_returns:
            df['daily_return'] = df['Close'].pct_change()
        
        if add_log_returns:
            df['log_return'] = np.log(df['Close'] / df['Close'].shift(1))
        
        return df


def quick_load(filepath: str) -> pd.DataFrame:
    """
    Quick load function for interactive use.
    
    Args:
        filepath: Path to CSV file
    
    Returns:
        Loaded and validated DataFrame
    """
    loader = NVDADataLoader(filepath)
    df, _ = loader.load_and_validate(verbose=False)
    return df


if __name__ == "__main__":
    # Example usage
    print("=" * 80)
    print("NVDA DATA LOADER - EXAMPLE USAGE".center(80))
    print("=" * 80)
    
    # This would need actual data file
    print("\nTo use this module:")
    print(">>> from src.data.loader import NVDADataLoader")
    print(">>> loader = NVDADataLoader('data/raw/NVDA_yfinance_clean.csv')")
    print(">>> df, report = loader.load_and_validate()")
    print("\n" + "=" * 80)
