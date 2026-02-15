#!/usr/bin/env python3
"""
Model Evaluation Script
=======================

Comprehensive evaluation of trained models including:
- Performance metrics
- Backtesting
- Cross-validation
- Feature importance
- Model comparison

Usage:
    python scripts/evaluate.py --model models/xgboost_model.model --data data/raw/NVDA_yfinance_clean.csv
    python scripts/evaluate.py --backtest --model models/my_model.pkl --data data/raw/NVDA_yfinance_clean.csv
    python scripts/evaluate.py --compare models/model1.pkl models/model2.pkl

Author: Senior ML Engineer
Date: February 2026
"""

import sys
import argparse
import json
from pathlib import Path
from typing import Dict, List, Any, Optional
import warnings

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data.loader import NVDADataLoader
from src.features.engineering import FeatureEngineer
from src.evaluation.metrics import comprehensive_evaluation, print_evaluation_report
from src.evaluation.backtesting import backtest_predictions, print_backtest_report
from src.evaluation.cross_validation import TimeSeriesCrossValidator
from src.utils.model_utils import load_model


def load_evaluation_data(data_path: str) -> tuple:
    """Load and prepare data for evaluation"""
    print("📂 Loading evaluation data...")

    loader = NVDADataLoader(data_path)
    df, report = loader.load_and_validate(verbose=False)

    # Create features
    feature_engineer = FeatureEngineer()
    df_features = feature_engineer.create_features(df)

    # Create target
    df_features['target'] = df_features['Close'].shift(-1)
    df_features = df_features.dropna()

    # Define features
    exclude_cols = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'target']
    feature_cols = [col for col in df_features.columns if col not in exclude_cols]

    print(f"✅ Evaluation data prepared: {len(df_features)} samples, {len(feature_cols)} features")

    return df_features, feature_cols


def evaluate_single_model(
    model_path: str,
    df: pd.DataFrame,
    feature_cols: List[str],
    output_dir: str = "reports"
) -> Dict[str, Any]:
    """Evaluate a single model comprehensively"""
    print(f"🔍 Evaluating model: {model_path}")

    # Load model
    model = load_model(model_path)
    model_name = Path(model_path).stem

    # Prepare data (use all data for evaluation)
    X = df[feature_cols]
    y = df['target']

    # Basic evaluation
    print("📊 Computing performance metrics...")
    y_pred = model.predict(X)
    metrics = comprehensive_evaluation(y, y_pred, df.get('Close'), model_name)

    # Cross-validation evaluation
    print("🔄 Performing cross-validation...")
    cv = TimeSeriesCrossValidator(n_splits=5, strategy='walk_forward')
    cv_scores = []

    for train_idx, test_idx in cv.split(X):
        X_train_cv, X_test_cv = X.iloc[train_idx], X.iloc[test_idx]
        y_train_cv, y_test_cv = y.iloc[train_idx], y.iloc[test_idx]

        # For CV, we need to fit the model (this assumes sklearn-like interface)
        if hasattr(model, 'fit'):
            model_cv = type(model)(**model.get_params()) if hasattr(model, 'get_params') else model
            model_cv.fit(X_train_cv, y_train_cv)
            y_pred_cv = model_cv.predict(X_test_cv)
        else:
            # For models without fit (like our baselines), use full model
            y_pred_cv = model.predict(X_test_cv)

        from sklearn.metrics import r2_score
        cv_scores.append(r2_score(y_test_cv, y_pred_cv))

    cv_summary = {
        'mean_r2': float(np.mean(cv_scores)),
        'std_r2': float(np.std(cv_scores)),
        'cv_folds': len(cv_scores)
    }

    # Feature importance (if available)
    feature_importance = None
    if hasattr(model, 'feature_importances_'):
        feature_importance = dict(zip(feature_cols, model.feature_importances_))
    elif hasattr(model, 'get_feature_importance'):
        try:
            importance_dict = model.get_feature_importance()
            feature_importance = dict(zip(feature_cols, importance_dict.values()))
        except:
            pass

    # Compile results
    results = {
        'model_name': model_name,
        'model_path': model_path,
        'metrics': metrics,
        'cv_results': cv_summary,
        'feature_importance': feature_importance,
        'evaluation_samples': len(X)
    }

    # Save results
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    results_file = output_dir / f"{model_name}_evaluation.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)

    print(f"💾 Results saved to: {results_file}")

    return results


def backtest_model(
    model_path: str,
    df: pd.DataFrame,
    feature_cols: List[str],
    output_dir: str = "reports"
) -> Dict[str, Any]:
    """Backtest model predictions"""
    print(f"📈 Backtesting model: {model_path}")

    # Load model
    model = load_model(model_path)
    model_name = Path(model_path).stem

    # Generate predictions
    X = df[feature_cols]
    predictions = model.predict(X)

    # Create signals (simplified: positive prediction = buy, negative = sell)
    # In practice, you'd want more sophisticated signal generation
    signals = pd.Series(predictions - df['Close'].values, index=df.index)

    # Backtest
    result = backtest_predictions(
        predictions=signals,
        actual_prices=df['Close'],
        dates=df['Date'],
        initial_capital=10000,
        signal_threshold=0.01  # 1% signal threshold
    )

    # Save backtest results
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    backtest_file = output_dir / f"{model_name}_backtest.json"
    backtest_data = {
        'model_name': model_name,
        'total_return': result.total_return,
        'annualized_return': result.annualized_return,
        'volatility': result.volatility,
        'sharpe_ratio': result.sharpe_ratio,
        'max_drawdown': result.max_drawdown,
        'win_rate': result.win_rate,
        'profit_factor': result.profit_factor,
        'total_trades': result.total_trades,
        'avg_trade_duration': result.avg_trade_duration
    }

    with open(backtest_file, 'w') as f:
        json.dump(backtest_data, f, indent=2, default=str)

    print(f"💾 Backtest results saved to: {backtest_file}")

    return backtest_data


def compare_models(
    model_paths: List[str],
    df: pd.DataFrame,
    feature_cols: List[str],
    output_dir: str = "reports"
) -> pd.DataFrame:
    """Compare multiple models"""
    print(f"⚖️ Comparing {len(model_paths)} models...")

    try:
        import pandas as pd
    except ImportError:
        raise ImportError("pandas required for model comparison")

    comparison_data = []

    for model_path in model_paths:
        try:
            # Load and evaluate
            results = evaluate_single_model(model_path, df, feature_cols, output_dir)

            # Extract key metrics
            data = {
                'model': results['model_name'],
                'mae': results['metrics'].get('mae', None),
                'rmse': results['metrics'].get('rmse', None),
                'r2': results['metrics'].get('r2', None),
                'directional_accuracy': results['metrics'].get('directional_accuracy', None),
                'cv_mean_r2': results['cv_results'].get('mean_r2', None),
                'cv_std_r2': results['cv_results'].get('std_r2', None)
            }

            comparison_data.append(data)

        except Exception as e:
            print(f"⚠️ Failed to evaluate {model_path}: {e}")
            continue

    comparison_df = pd.DataFrame(comparison_data)

    # Save comparison
    output_dir = Path(output_dir)
    comparison_file = output_dir / "model_comparison.csv"
    comparison_df.to_csv(comparison_file, index=False)

    print(f"💾 Comparison saved to: {comparison_file}")

    return comparison_df


def main():
    parser = argparse.ArgumentParser(
        description="🔍 NVIDIA Stock Model Evaluation Suite",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Evaluate single model
  python scripts/evaluate.py --model models/xgboost_model.model --data data/raw/NVDA_yfinance_clean.csv

  # Backtest model
  python scripts/evaluate.py --backtest --model models/my_model.pkl --data data/raw/NVDA_yfinance_clean.csv

  # Compare multiple models
  python scripts/evaluate.py --compare models/model1.pkl models/model2.pkl models/model3.pkl

  # Full evaluation with custom output
  python scripts/evaluate.py --model models/xgboost.model --backtest --output reports/evaluation/
        """
    )

    parser.add_argument(
        '--model',
        type=str,
        help='Path to single model file to evaluate'
    )

    parser.add_argument(
        '--compare',
        nargs='+',
        help='List of model paths to compare'
    )

    parser.add_argument(
        '--data',
        type=str,
        required=True,
        help='Path to CSV data file'
    )

    parser.add_argument(
        '--backtest',
        action='store_true',
        help='Perform backtesting evaluation'
    )

    parser.add_argument(
        '--output',
        type=str,
        default='reports',
        help='Output directory for results'
    )

    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Verbose output'
    )

    args = parser.parse_args()

    # Validate arguments
    if not args.model and not args.compare:
        parser.error("Must specify either --model or --compare")

    if args.model and args.compare:
        parser.error("Cannot specify both --model and --compare")

    # Validate file paths
    if args.model and not Path(args.model).exists():
        parser.error(f"Model file not found: {args.model}")

    if args.compare:
        for model_path in args.compare:
            if not Path(model_path).exists():
                parser.error(f"Model file not found: {model_path}")

    if not Path(args.data).exists():
        # Try common corrections
        data_path = Path(args.data)
        if data_path.suffix == '.cs' and not data_path.exists():
            # Common typo: .cs instead of .csv
            csv_path = data_path.with_suffix('.csv')
            if csv_path.exists():
                print(f"Correcting file extension: {args.data} -> {csv_path}")
                args.data = str(csv_path)
            else:
                parser.error(f"Data file not found: {args.data}")
        else:
            parser.error(f"Data file not found: {args.data}")

    print("=" * 80)
    print("🔍 NVIDIA STOCK MODEL - EVALUATION SUITE".center(80))
    print("=" * 80)

    try:
        # Load data
        df, feature_cols = load_evaluation_data(args.data)

        if args.model:
            # Evaluate single model
            results = evaluate_single_model(args.model, df, feature_cols, args.output)

            # Print results
            print_evaluation_report(results['metrics'], f"Model Evaluation: {results['model_name']}")

            print(f"\\n🔄 Cross-Validation (R²): {results['cv_results']['mean_r2']:.4f} ± {results['cv_results']['std_r2']:.4f}")

            if args.backtest:
                backtest_results = backtest_model(args.model, df, feature_cols, args.output)
                print_backtest_report(
                    type('BacktestResult', (), backtest_results),
                    f"Backtest Results: {results['model_name']}"
                )

        elif args.compare:
            # Compare models
            comparison_df = compare_models(args.compare, df, feature_cols, args.output)

            print("\\n🏆 MODEL COMPARISON:")
            print("=" * 60)
            print(comparison_df.to_string(index=False, float_format='%.4f'))

            # Find best model
            if 'mae' in comparison_df.columns:
                best_idx = comparison_df['mae'].idxmin()
                best_model = comparison_df.loc[best_idx, 'model']
                best_mae = comparison_df.loc[best_idx, 'mae']
                print(f"\\n🏅 Best Model by MAE: {best_model} (${best_mae:.4f})")

        print("\\n" + "=" * 80)
        print("✅ EVALUATION COMPLETED SUCCESSFULLY!")
        print("=" * 80)

    except Exception as e:
        print(f"\\n❌ Error during evaluation: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()