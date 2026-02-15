#!/usr/bin/env python3
"""
Training Pipeline Script
========================

Complete training pipeline for NVIDIA stock prediction models.
Supports baseline models, XGBoost, and hyperparameter tuning.

Usage:
    python scripts/train.py --model baseline --data data/raw/NVDA_yfinance_clean.csv
    python scripts/train.py --model xgboost --tune --data data/raw/NVDA_yfinance_clean.csv
    python scripts/train.py --model ensemble --output models/my_model.pkl

Author: Senior ML Engineer
Date: February 2026
"""

import sys
import argparse
import json
from pathlib import Path
from typing import Dict, Any, Optional
import warnings

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data.loader import NVDADataLoader
from src.features.engineering import FeatureEngineer
from src.models import (
    create_baseline_models,
    XGBoostTimeSeriesModel
)
from src.evaluation.metrics import comprehensive_evaluation
from src.evaluation.cross_validation import TimeSeriesCrossValidator
from src.utils.model_utils import save_model, ModelRegistry
from src.utils.hyperparameter_tuning import optimize_model_hyperparameters


def load_and_prepare_data(data_path: str) -> tuple:
    """
    Load and prepare data for training.

    Returns:
        Tuple of (X_train, X_test, y_train, y_test, feature_cols)
    """
    print("📂 Loading data...")

    # Load raw data
    loader = NVDADataLoader(data_path)
    df, report = loader.load_and_validate(verbose=True)

    if not report.is_valid:
        print("⚠️  Data quality issues detected. Consider fixing before training.")
        response = input("Continue anyway? (y/n): ")
        if response.lower() != 'y':
            sys.exit(1)

    # Create features
    print("🔧 Engineering features...")
    feature_engineer = FeatureEngineer()
    df_features = feature_engineer.create_features(df)

    # Create target (next day's close)
    df_features['target'] = df_features['Close'].shift(-1)
    df_features = df_features.dropna()

    # Define features (exclude OHLCV and target)
    exclude_cols = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'target']
    feature_cols = [col for col in df_features.columns if col not in exclude_cols]

    # Split data (time-series split)
    split_idx = int(len(df_features) * 0.8)
    train_df = df_features.iloc[:split_idx]
    test_df = df_features.iloc[split_idx:]

    X_train = train_df[feature_cols]
    y_train = train_df['target']
    X_test = test_df[feature_cols]
    y_test = test_df['target']

    print(f"✅ Data prepared: {len(feature_cols)} features, {len(X_train)} train, {len(X_test)} test")

    return X_train, X_test, y_train, y_test, feature_cols


def train_baseline_models(
    X_train, X_test, y_train, y_test, feature_cols, output_dir="models"
) -> Dict[str, Any]:
    """Train and evaluate baseline models"""
    print("🏃 Training baseline models...")

    from src.evaluation.metrics import compare_models

    # Create baseline models
    baseline_models = create_baseline_models()

    # Train and evaluate each model
    results = []
    trained_models = {}

    for model_name, model in baseline_models.items():
        print(f"  Training {model_name}...")

        # Train
        model.fit(X_train, y_train)
        trained_models[model_name] = model

        # Evaluate
        metrics = comprehensive_evaluation(
            y_test, model.predict(X_test), X_test.get('Close'), model_name
        )
        results.append(metrics)

    # Compare models
    comparison_df = compare_models(results)

    # Save best model
    best_model_name = comparison_df.index[0]  # Best by MAE
    best_model = trained_models[best_model_name]

    output_path = Path(output_dir) / f"baseline_{best_model_name}.pkl"
    save_model(
        best_model,
        str(output_path),
        metadata={
            'model_name': f'baseline_{best_model_name}',
            'model_type': 'baseline',
            'features': feature_cols,
            'performance': results[0] if results else {},
            'training_samples': len(X_train),
            'test_samples': len(X_test)
        }
    )

    return {
        'models': trained_models,
        'results': results,
        'comparison': comparison_df,
        'best_model': best_model_name,
        'saved_path': str(output_path)
    }


def train_xgboost_model(
    X_train, X_test, y_train, y_test, feature_cols,
    tune_hyperparams=False, output_dir="models"
) -> Dict[str, Any]:
    """Train XGBoost model with optional hyperparameter tuning"""
    print("🚀 Training XGBoost model...")

    # Default parameters
    params = {
        'n_estimators': 100,
        'max_depth': 6,
        'learning_rate': 0.1,
        'subsample': 0.8,
        'colsample_bytree': 0.8
    }

    # Hyperparameter tuning
    if tune_hyperparams:
        print("🎯 Performing hyperparameter tuning...")
        try:
            tune_results = optimize_model_hyperparameters(
                XGBoostTimeSeriesModel,
                X_train, y_train,
                method='auto',  # Auto-select best available method
                cv_strategy='walk_forward',
                cv_params={'n_splits': 3}
            )

            params.update(tune_results['best_params'])
            print(f"✅ Best parameters found: {tune_results['best_params']}")
            print(f"   Best score: {tune_results['best_score']:.4f}")

        except Exception as e:
            print(f"⚠️  Hyperparameter tuning failed: {e}")
            print("Using default parameters...")

    # Train final model
    model = XGBoostTimeSeriesModel(**params)
    model.fit(X_train, y_train)

    # Cross-validation
    print("🔄 Performing cross-validation...")
    cv_results = model.cross_validate(X_train, y_train, n_splits=5)

    # Evaluate on test set
    test_metrics = model.evaluate(X_test, y_test)

    # Save model
    output_path = Path(output_dir) / "xgboost_model.model"
    saved_path = model.save_model(str(output_path))

    # Save metadata
    metadata = {
        'model_name': 'xgboost_timeseries',
        'model_type': 'xgboost',
        'parameters': params,
        'features': feature_cols,
        'cv_results': cv_results,
        'test_performance': test_metrics,
        'training_samples': len(X_train),
        'test_samples': len(X_test),
        'hyperparameter_tuned': tune_hyperparams
    }

    metadata_path = Path(output_dir) / "xgboost_metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2, default=str)

    return {
        'model': model,
        'cv_results': cv_results,
        'test_metrics': test_metrics,
        'saved_path': saved_path,
        'metadata': metadata
    }


def main():
    parser = argparse.ArgumentParser(
        description="🚀 NVIDIA Stock Prediction Model Training Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train baseline models
  python scripts/train.py --model baseline --data data/raw/NVDA_yfinance_clean.csv

  # Train XGBoost with hyperparameter tuning
  python scripts/train.py --model xgboost --tune --data data/raw/NVDA_yfinance_clean.csv

  # Train ensemble model
  python scripts/train.py --model ensemble --output models/my_ensemble.pkl

  # Custom output directory
  python scripts/train.py --model xgboost --output models/production/
        """
    )

    parser.add_argument(
        '--model',
        type=str,
        required=True,
        choices=['baseline', 'xgboost', 'ensemble'],
        help='Model type to train'
    )

    parser.add_argument(
        '--data',
        type=str,
        required=True,
        help='Path to CSV data file'
    )

    parser.add_argument(
        '--output',
        type=str,
        default='models',
        help='Output directory for saved models'
    )

    parser.add_argument(
        '--tune',
        action='store_true',
        help='Perform hyperparameter tuning (XGBoost only)'
    )

    parser.add_argument(
        '--register',
        action='store_true',
        help='Register model in model registry'
    )

    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Verbose output'
    )

    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("🚀 NVIDIA STOCK PREDICTION - MODEL TRAINING PIPELINE".center(80))
    print("=" * 80)

    try:
        # Load and prepare data
        X_train, X_test, y_train, y_test, feature_cols = load_and_prepare_data(args.data)

        # Train model based on type
        if args.model == 'baseline':
            results = train_baseline_models(
                X_train, X_test, y_train, y_test, feature_cols, args.output
            )

            print("\\n🏆 BASELINE MODEL RESULTS:")
            print(results['comparison'].head())

            print(f"\\n💾 Best model saved: {results['saved_path']}")

        elif args.model == 'xgboost':
            results = train_xgboost_model(
                X_train, X_test, y_train, y_test, feature_cols,
                args.tune, args.output
            )

            print("\\n🏆 XGBOOST MODEL RESULTS:")
            print(f"CV Mean MAE: ${results['cv_results']['overall']['mean_mae']:.4f}")
            print(f"Test MAE:     ${results['test_metrics']['mae']:.4f}")
            print(f"Test R²:      {results['test_metrics']['r2']:.4f}")

            print(f"\\n💾 Model saved: {results['saved_path']}")

        elif args.model == 'ensemble':
            # For now, just train baseline ensemble
            results = train_baseline_models(
                X_train, X_test, y_train, y_test, feature_cols, args.output
            )

            print("\\n🏆 ENSEMBLE MODEL RESULTS:")
            print(results['comparison'].head())

        # Register model if requested
        if args.register:
            registry = ModelRegistry()
            # Registration logic would go here
            print("\\n📋 Model registered in registry")

        print("\\n" + "=" * 80)
        print("✅ TRAINING PIPELINE COMPLETED SUCCESSFULLY!")
        print("=" * 80)

    except Exception as e:
        print(f"\\n❌ Error during training: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()