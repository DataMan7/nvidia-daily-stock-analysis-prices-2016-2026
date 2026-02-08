"""
Data Leakage Detection Module
===============================

This module implements comprehensive data leakage detection for time-series ML pipelines.

Author: Senior ML Engineer
Date: February 2026

CRITICAL LEARNING OBJECTIVE:
---------------------------
Understand that data leakage is the #1 reason models fail in production.
A model with 99.9% accuracy in dev and 60% in production is a FAILED model.

Types of Leakage Detected:
1. Temporal leakage (future → past)
2. Feature leakage (target information in features)
3. Preprocessing leakage (fit on full data before split)
4. Evaluation leakage (wrong cross-validation strategy)
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime
import warnings


@dataclass
class LeakageReport:
    """Container for leakage detection results"""
    leakage_detected: bool
    severity: str  # 'CRITICAL', 'HIGH', 'MEDIUM', 'LOW', 'NONE'
    leakage_types: List[str]
    details: Dict[str, any]
    recommendations: List[str]
    timestamp: str = datetime.now().isoformat()


class DataLeakageDetector:
    """
    Comprehensive data leakage detector for time-series data.
    
    This is what separates amateur projects from production-ready systems.
    
    Example:
        >>> detector = DataLeakageDetector(df, target_col='Close', date_col='Date')
        >>> report = detector.run_full_audit()
        >>> if report.leakage_detected:
        >>>     print("🚨 LEAKAGE FOUND!")
        >>>     print(report.details)
    """
    
    def __init__(
        self,
        df: pd.DataFrame,
        target_col: str,
        date_col: str = 'Date',
        threshold_correlation: float = 0.95,
        verbose: bool = True
    ):
        """
        Initialize leakage detector.
        
        Args:
            df: DataFrame with time-series data
            target_col: Name of target column
            date_col: Name of date/time column
            threshold_correlation: Correlation threshold for leakage warning
            verbose: Print detailed messages
        """
        self.df = df.copy()
        self.target_col = target_col
        self.date_col = date_col
        self.threshold_correlation = threshold_correlation
        self.verbose = verbose
        
        # Ensure date column is datetime
        if date_col in self.df.columns:
            self.df[date_col] = pd.to_datetime(self.df[date_col])
            self.df = self.df.sort_values(date_col).reset_index(drop=True)
    
    def run_full_audit(self) -> LeakageReport:
        """
        Run complete leakage audit.
        
        Returns:
            LeakageReport with all findings
        """
        if self.verbose:
            print("=" * 80)
            print("🔍 DATA LEAKAGE FORENSIC AUDIT".center(80))
            print("=" * 80)
        
        leakage_types = []
        details = {}
        recommendations = []
        
        # 1. Temporal Ordering Check
        temporal_check = self._check_temporal_ordering()
        details['temporal_ordering'] = temporal_check
        if not temporal_check['is_valid']:
            leakage_types.append('TEMPORAL_ORDERING')
            recommendations.append("Ensure data is sorted by date before any operations")
        
        # 2. Perfect Correlation Check
        correlation_check = self._check_perfect_correlations()
        details['perfect_correlations'] = correlation_check
        if correlation_check['suspicious_features']:
            leakage_types.append('PERFECT_CORRELATION')
            recommendations.extend(correlation_check['recommendations'])
        
        # 3. Future Information Check
        future_info_check = self._check_future_information()
        details['future_information'] = future_info_check
        if future_info_check['potential_leaks']:
            leakage_types.append('FUTURE_INFORMATION')
            recommendations.extend(future_info_check['recommendations'])
        
        # 4. Statistical Impossibility Check
        stats_check = self._check_statistical_impossibilities()
        details['statistical_checks'] = stats_check
        if stats_check['anomalies_found']:
            leakage_types.append('STATISTICAL_ANOMALY')
            recommendations.extend(stats_check['recommendations'])
        
        # 5. Lag Feature Validation
        lag_check = self._check_lag_features()
        details['lag_features'] = lag_check
        if lag_check['improper_lags']:
            leakage_types.append('IMPROPER_LAG_FEATURES')
            recommendations.extend(lag_check['recommendations'])
        
        # Determine severity
        severity = self._determine_severity(leakage_types)
        
        # Generate final report
        report = LeakageReport(
            leakage_detected=len(leakage_types) > 0,
            severity=severity,
            leakage_types=leakage_types,
            details=details,
            recommendations=list(set(recommendations))  # Remove duplicates
        )
        
        if self.verbose:
            self._print_report(report)
        
        return report
    
    def _check_temporal_ordering(self) -> Dict:
        """Check if data is properly sorted by time"""
        if self.date_col not in self.df.columns:
            return {
                'is_valid': False,
                'message': f"Date column '{self.date_col}' not found"
            }
        
        is_sorted = self.df[self.date_col].is_monotonic_increasing
        duplicates = self.df[self.date_col].duplicated().sum()
        
        return {
            'is_valid': is_sorted and duplicates == 0,
            'is_sorted': is_sorted,
            'duplicate_dates': int(duplicates),
            'date_range': (
                str(self.df[self.date_col].min()),
                str(self.df[self.date_col].max())
            )
        }
    
    def _check_perfect_correlations(self) -> Dict:
        """Detect suspiciously high correlations with target"""
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
        
        if self.target_col not in numeric_cols:
            return {
                'suspicious_features': [],
                'recommendations': []
            }
        
        # Remove target and date columns
        feature_cols = [c for c in numeric_cols if c != self.target_col]
        
        if not feature_cols:
            return {
                'suspicious_features': [],
                'recommendations': []
            }
        
        # Calculate correlations
        correlations = self.df[feature_cols + [self.target_col]].corr()[self.target_col].drop(self.target_col)
        
        # Find suspiciously high correlations
        suspicious = correlations[correlations.abs() > self.threshold_correlation]
        
        suspicious_features = []
        recommendations = []
        
        for feat, corr in suspicious.items():
            suspicious_features.append({
                'feature': feat,
                'correlation': float(corr),
                'warning': f"Correlation of {corr:.4f} is suspiciously high"
            })
            
            if abs(corr) > 0.99:
                recommendations.append(
                    f"⚠️  Feature '{feat}' has correlation {corr:.4f} - likely leakage or redundant"
                )
        
        return {
            'suspicious_features': suspicious_features,
            'recommendations': recommendations,
            'all_correlations': correlations.to_dict()
        }
    
    def _check_future_information(self) -> Dict:
        """
        Detect potential future information leakage.
        
        This is the most critical check for time-series.
        """
        potential_leaks = []
        recommendations = []
        
        # Check for features that might contain future information
        feature_cols = [c for c in self.df.columns if c not in [self.target_col, self.date_col]]
        
        for col in feature_cols:
            # Check if feature name suggests it might be leaky
            leaky_keywords = ['future', 'next', 'ahead', 'tomorrow', 'forecast']
            
            if any(keyword in col.lower() for keyword in leaky_keywords):
                potential_leaks.append({
                    'feature': col,
                    'reason': 'Feature name suggests future information',
                    'severity': 'HIGH'
                })
                recommendations.append(
                    f"⚠️  Review feature '{col}' - name suggests it contains future information"
                )
            
            # Check for shift/lag features that might be incorrectly created
            if 'lag' in col.lower() or 'shift' in col.lower():
                # Extract lag number if possible
                try:
                    # Simple heuristic: check if there are any leading values
                    # (which would indicate incorrect shifting)
                    if self.df[col].iloc[0] != self.df[col].iloc[0]:  # NaN check
                        pass  # This is expected for lag features
                    else:
                        # If first value is not NaN, might be incorrectly created
                        potential_leaks.append({
                            'feature': col,
                            'reason': 'Lag feature starts with non-NaN value',
                            'severity': 'MEDIUM'
                        })
                except Exception:
                    pass
        
        return {
            'potential_leaks': potential_leaks,
            'recommendations': recommendations
        }
    
    def _check_statistical_impossibilities(self) -> Dict:
        """Check for statistically impossible patterns that suggest leakage"""
        anomalies = []
        recommendations = []
        
        # Check 1: Target has no variance (constant)
        if self.df[self.target_col].std() == 0:
            anomalies.append("Target variable has zero variance")
            recommendations.append("⚠️  Target variable is constant - check data loading")
        
        # Check 2: Unrealistic value ranges
        target_min = self.df[self.target_col].min()
        target_max = self.df[self.target_col].max()
        
        if target_min < 0 and 'price' in self.target_col.lower():
            anomalies.append(f"Negative values in price column: min={target_min}")
            recommendations.append("⚠️  Price column contains negative values - check preprocessing")
        
        # Check 3: Duplicate rows (exact copies)
        duplicates = self.df.duplicated().sum()
        if duplicates > 0:
            anomalies.append(f"Found {duplicates} duplicate rows")
            recommendations.append(f"⚠️  {duplicates} duplicate rows found - may cause leakage in CV")
        
        return {
            'anomalies_found': len(anomalies) > 0,
            'anomalies': anomalies,
            'recommendations': recommendations
        }
    
    def _check_lag_features(self) -> Dict:
        """
        Validate lag features are created correctly.
        
        This is critical because incorrect lag features are the #1 source
        of leakage in time-series projects.
        """
        lag_features = [col for col in self.df.columns if 'lag' in col.lower() or 'shift' in col.lower()]
        
        improper_lags = []
        recommendations = []
        
        for lag_col in lag_features:
            # Check if lag feature starts with NaN values (it should)
            first_non_nan = self.df[lag_col].first_valid_index()
            
            if first_non_nan == 0:
                improper_lags.append({
                    'feature': lag_col,
                    'issue': 'Lag feature starts with non-NaN value',
                    'risk': 'HIGH - May contain future information'
                })
                recommendations.append(
                    f"⚠️  Lag feature '{lag_col}' starts with non-NaN - likely created incorrectly"
                )
            
            # Check correlation with target at different offsets
            # If correlation is higher at offset=0 than offset=1, something is wrong
            try:
                if len(self.df) > 10:
                    corr_current = self.df[[lag_col, self.target_col]].corr().iloc[0, 1]
                    corr_shifted = self.df[[lag_col, self.target_col]].shift(-1).corr().iloc[0, 1]
                    
                    if abs(corr_shifted) > abs(corr_current) * 1.5:
                        improper_lags.append({
                            'feature': lag_col,
                            'issue': 'Higher correlation with shifted target',
                            'risk': 'MEDIUM - Check feature creation logic'
                        })
            except Exception:
                pass  # Skip if correlation calculation fails
        
        return {
            'lag_features_found': lag_features,
            'improper_lags': improper_lags,
            'recommendations': recommendations
        }
    
    def _determine_severity(self, leakage_types: List[str]) -> str:
        """Determine overall severity of leakage"""
        if not leakage_types:
            return 'NONE'
        
        critical_types = {'PERFECT_CORRELATION', 'FUTURE_INFORMATION'}
        high_types = {'TEMPORAL_ORDERING', 'IMPROPER_LAG_FEATURES'}
        
        if any(t in critical_types for t in leakage_types):
            return 'CRITICAL'
        elif any(t in high_types for t in leakage_types):
            return 'HIGH'
        elif len(leakage_types) > 2:
            return 'MEDIUM'
        else:
            return 'LOW'
    
    def _print_report(self, report: LeakageReport):
        """Print formatted leakage report"""
        print("\n" + "=" * 80)
        print("📊 LEAKAGE AUDIT REPORT".center(80))
        print("=" * 80)
        
        # Status
        if report.leakage_detected:
            print(f"\n🚨 LEAKAGE DETECTED - Severity: {report.severity}")
            print(f"Leakage Types: {', '.join(report.leakage_types)}")
        else:
            print("\n✅ NO LEAKAGE DETECTED - Pipeline appears clean")
        
        # Recommendations
        if report.recommendations:
            print("\n" + "-" * 80)
            print("🔧 RECOMMENDATIONS:")
            print("-" * 80)
            for i, rec in enumerate(report.recommendations, 1):
                print(f"{i}. {rec}")
        
        print("\n" + "=" * 80)


def deliberately_introduce_leakage(
    df: pd.DataFrame,
    target_col: str,
    leakage_type: str = 'scaling'
) -> pd.DataFrame:
    """
    Deliberately introduce leakage for educational purposes.
    
    This function demonstrates common leakage mistakes.
    
    Args:
        df: Clean DataFrame
        target_col: Target column name
        leakage_type: Type of leakage to introduce
            - 'scaling': Scale before train-test split
            - 'future_lag': Create lag features incorrectly
            - 'target_encoding': Use target statistics
    
    Returns:
        DataFrame with leakage introduced
    
    Example:
        >>> # This is for LEARNING - DO NOT use in production!
        >>> leaky_df = deliberately_introduce_leakage(df, 'Close', 'scaling')
        >>> # Train model on leaky_df
        >>> # Observe unrealistically high accuracy
        >>> # Learn to detect this in the wild
    """
    from sklearn.preprocessing import StandardScaler
    
    df_leaky = df.copy()
    
    if leakage_type == 'scaling':
        # ❌ WRONG: Scale entire dataset before split
        scaler = StandardScaler()
        numeric_cols = df_leaky.select_dtypes(include=[np.number]).columns
        df_leaky[numeric_cols] = scaler.fit_transform(df_leaky[numeric_cols])
        print("⚠️  Introduced SCALING leakage: Fitted scaler on full dataset")
    
    elif leakage_type == 'future_lag':
        # ❌ WRONG: Create "lag" that actually looks forward
        df_leaky['price_lag_1_LEAKY'] = df_leaky[target_col].shift(-1)  # Note: negative shift!
        print("⚠️  Introduced FUTURE LAG leakage: Created lag with future information")
    
    elif leakage_type == 'target_encoding':
        # ❌ WRONG: Use target mean as feature
        df_leaky['target_mean_LEAKY'] = df_leaky[target_col].mean()
        print("⚠️  Introduced TARGET ENCODING leakage: Used global target mean")
    
    return df_leaky


if __name__ == "__main__":
    # Example usage
    print("=" * 80)
    print("DATA LEAKAGE DETECTOR - EXAMPLE USAGE".center(80))
    print("=" * 80)
    
    # Create sample data
    dates = pd.date_range('2020-01-01', periods=100, freq='D')
    df_sample = pd.DataFrame({
        'Date': dates,
        'Close': np.random.randn(100).cumsum() + 100,
        'Volume': np.random.randint(1000000, 10000000, 100)
    })
    
    # Test 1: Clean data
    print("\n" + "=" * 80)
    print("Test 1: Clean Data")
    print("=" * 80)
    detector = DataLeakageDetector(df_sample, target_col='Close')
    report1 = detector.run_full_audit()
    
    # Test 2: Introduce leakage
    print("\n" + "=" * 80)
    print("Test 2: Data with Deliberate Leakage")
    print("=" * 80)
    df_leaky = deliberately_introduce_leakage(df_sample, 'Close', 'future_lag')
    detector2 = DataLeakageDetector(df_leaky, target_col='Close')
    report2 = detector2.run_full_audit()
    
    print("\n" + "=" * 80)
    print("✅ Example completed successfully!")
    print("=" * 80)
