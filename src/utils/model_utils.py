"""
Model Utilities
===============

Utilities for model persistence, loading, and management.
Supports saving/loading models with metadata and versioning.

Author: Senior ML Engineer
Date: February 2026
"""

import os
import json
import pickle
import hashlib
from pathlib import Path
from typing import Dict, Any, Optional, Union, List
from datetime import datetime
import warnings

try:
    import joblib
    HAS_JOBLIB = True
except ImportError:
    HAS_JOBLIB = False

try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False


class ModelRegistry:
    """
    Model registry for versioning and metadata management.

    Features:
    - Model versioning
    - Metadata storage
    - Performance tracking
    - Model comparison
    """

    def __init__(self, registry_path: str = "models/registry"):
        """
        Initialize model registry.

        Args:
            registry_path: Path to store registry data
        """
        self.registry_path = Path(registry_path)
        self.registry_path.mkdir(parents=True, exist_ok=True)
        self.registry_file = self.registry_path / "registry.json"

        # Load existing registry
        self.registry = self._load_registry()

    def _load_registry(self) -> Dict[str, Any]:
        """Load registry from disk"""
        if self.registry_file.exists():
            try:
                with open(self.registry_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                warnings.warn(f"Could not load registry: {e}")
                return {}
        return {}

    def _save_registry(self):
        """Save registry to disk"""
        with open(self.registry_file, 'w') as f:
            json.dump(self.registry, f, indent=2, default=str)

    def register_model(
        self,
        model_name: str,
        model_path: str,
        metadata: Dict[str, Any],
        performance_metrics: Optional[Dict[str, float]] = None
    ) -> str:
        """
        Register a new model version.

        Args:
            model_name: Name of the model
            model_path: Path to saved model file
            metadata: Model metadata
            performance_metrics: Performance metrics

        Returns:
            Version ID of registered model
        """
        # Generate version ID
        timestamp = datetime.now().isoformat()
        content_hash = hashlib.md5(f"{model_name}{timestamp}".encode()).hexdigest()[:8]
        version_id = f"{model_name}_v_{content_hash}"

        # Create model entry
        model_entry = {
            'model_name': model_name,
            'version_id': version_id,
            'model_path': str(model_path),
            'created_at': timestamp,
            'metadata': metadata,
            'performance': performance_metrics or {},
            'status': 'active'
        }

        # Add to registry
        if model_name not in self.registry:
            self.registry[model_name] = {}

        self.registry[model_name][version_id] = model_entry
        self._save_registry()

        return version_id

    def get_model_versions(self, model_name: str) -> List[Dict[str, Any]]:
        """Get all versions of a model"""
        if model_name not in self.registry:
            return []

        return list(self.registry[model_name].values())

    def get_latest_version(self, model_name: str) -> Optional[Dict[str, Any]]:
        """Get the latest version of a model"""
        versions = self.get_model_versions(model_name)
        if not versions:
            return None

        # Sort by creation time
        return max(versions, key=lambda x: x['created_at'])

    def deactivate_version(self, model_name: str, version_id: str):
        """Deactivate a model version"""
        if model_name in self.registry and version_id in self.registry[model_name]:
            self.registry[model_name][version_id]['status'] = 'deprecated'
            self._save_registry()


def save_model(
    model: Any,
    filepath: str,
    metadata: Optional[Dict[str, Any]] = None,
    registry: Optional[ModelRegistry] = None
) -> str:
    """
    Save model with metadata.

    Args:
        model: Model object to save
        filepath: Path to save model
        metadata: Additional metadata
        registry: Model registry for versioning

    Returns:
        Path where model was saved
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    # Determine save method based on model type
    if HAS_XGBOOST and isinstance(model, xgb.Booster):
        # XGBoost model
        model.save_model(str(filepath))
        model_type = 'xgboost'
    elif hasattr(model, 'save_model') and callable(getattr(model, 'save_model')):
        # Custom save method
        model.save_model(str(filepath))
        model_type = 'custom'
    elif HAS_JOBLIB:
        # Use joblib for sklearn models
        joblib.dump(model, filepath)
        model_type = 'joblib'
    else:
        # Fallback to pickle
        with open(filepath, 'wb') as f:
            pickle.dump(model, f)
        model_type = 'pickle'

    # Save metadata
    metadata = metadata or {}
    metadata.update({
        'model_type': model_type,
        'saved_at': datetime.now().isoformat(),
        'filepath': str(filepath)
    })

    metadata_file = filepath.parent / f"{filepath.stem}_metadata.json"
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2, default=str)

    # Register model if registry provided
    if registry is not None:
        model_name = metadata.get('model_name', filepath.stem)
        registry.register_model(
            model_name=model_name,
            model_path=str(filepath),
            metadata=metadata,
            performance_metrics=metadata.get('performance')
        )

    return str(filepath)


def load_model(
    filepath: str,
    metadata_only: bool = False
) -> Union[Any, Dict[str, Any]]:
    """
    Load model from disk.

    Args:
        filepath: Path to saved model
        metadata_only: Return only metadata

    Returns:
        Loaded model or metadata dict
    """
    filepath = Path(filepath)

    # Load metadata
    metadata_file = filepath.parent / f"{filepath.stem}_metadata.json"
    metadata = {}

    if metadata_file.exists():
        try:
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
        except Exception as e:
            warnings.warn(f"Could not load metadata: {e}")

    if metadata_only:
        return metadata

    # Load model based on type
    model_type = metadata.get('model_type', 'pickle')

    if model_type == 'xgboost' and HAS_XGBOOST:
        model = xgb.Booster()
        model.load_model(str(filepath))
    elif model_type == 'joblib' and HAS_JOBLIB:
        model = joblib.load(filepath)
    elif model_type == 'custom':
        # Assume model has load_model method
        model_class = metadata.get('model_class')
        if model_class:
            # This would need more sophisticated handling
            raise NotImplementedError("Custom model loading not implemented")
        else:
            raise ValueError("Cannot load custom model without model_class in metadata")
    else:
        # Fallback to pickle
        with open(filepath, 'rb') as f:
            model = pickle.load(f)

    return model


def compare_models(
    model_paths: List[str],
    test_data: Optional[Dict[str, Any]] = None
) -> pd.DataFrame:
    """
    Compare multiple saved models.

    Args:
        model_paths: List of paths to saved models
        test_data: Test data for evaluation (optional)

    Returns:
        DataFrame with model comparison
    """
    try:
        import pandas as pd
    except ImportError:
        raise ImportError("pandas required for model comparison")

    comparison_data = []

    for path in model_paths:
        try:
            # Load metadata
            metadata = load_model(path, metadata_only=True)

            # Load model for evaluation if test data provided
            performance = metadata.get('performance', {})

            if test_data and 'X_test' in test_data and 'y_test' in test_data:
                model = load_model(path)
                # This would need model-specific evaluation
                # For now, just use stored performance
                pass

            data = {
                'model_path': path,
                'model_name': metadata.get('model_name', Path(path).stem),
                'model_type': metadata.get('model_type', 'unknown'),
                'created_at': metadata.get('saved_at', 'unknown'),
                **performance
            }

            comparison_data.append(data)

        except Exception as e:
            warnings.warn(f"Could not load model {path}: {e}")
            continue

    return pd.DataFrame(comparison_data)


def cleanup_old_models(
    registry: ModelRegistry,
    model_name: str,
    keep_versions: int = 5
):
    """
    Clean up old model versions, keeping only the most recent ones.

    Args:
        registry: Model registry
        model_name: Name of model to clean up
        keep_versions: Number of versions to keep
    """
    versions = registry.get_model_versions(model_name)

    if len(versions) <= keep_versions:
        return

    # Sort by creation time, keep most recent
    versions.sort(key=lambda x: x['created_at'], reverse=True)
    versions_to_remove = versions[keep_versions:]

    for version in versions_to_remove:
        # Mark as deprecated
        registry.deactivate_version(model_name, version['version_id'])

        # Optionally delete files
        try:
            Path(version['model_path']).unlink(missing_ok=True)
            metadata_path = Path(version['model_path']).parent / f"{Path(version['model_path']).stem}_metadata.json"
            metadata_path.unlink(missing_ok=True)
        except Exception as e:
            warnings.warn(f"Could not delete old model files: {e}")


if __name__ == "__main__":
    # Example usage
    print("=" * 80)
    print("MODEL UTILITIES - EXAMPLE USAGE".center(80))
    print("=" * 80)

    # Create sample model (using sklearn if available)
    try:
        from sklearn.linear_model import LinearRegression
        import numpy as np

        # Create sample data
        np.random.seed(42)
        X = np.random.randn(100, 5)
        y = X.sum(axis=1) + np.random.randn(100) * 0.1

        # Train model
        model = LinearRegression()
        model.fit(X, y)

        print("📊 Training sample model...")

        # Save model
        save_path = "models/example_model.pkl"
        metadata = {
            'model_name': 'example_linear_regression',
            'description': 'Sample linear regression model',
            'features': ['feat1', 'feat2', 'feat3', 'feat4', 'feat5'],
            'performance': {
                'train_r2': 0.95,
                'cv_mean_r2': 0.89
            }
        }

        saved_path = save_model(model, save_path, metadata)
        print(f"✅ Model saved to: {saved_path}")

        # Load model
        loaded_model = load_model(saved_path)
        print("✅ Model loaded successfully")

        # Test registry
        registry = ModelRegistry()
        version_id = registry.register_model(
            model_name='example_linear_regression',
            model_path=saved_path,
            metadata=metadata,
            performance_metrics=metadata['performance']
        )
        print(f"✅ Model registered with version: {version_id}")

        # List versions
        versions = registry.get_model_versions('example_linear_regression')
        print(f"📋 Total versions registered: {len(versions)}")

    except ImportError:
        print("⚠️  sklearn not available for example")

    print("\n" + "=" * 80)
    print("✅ Model utilities example completed!")
    print("=" * 80)