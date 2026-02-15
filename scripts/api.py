#!/usr/bin/env python3
"""
FastAPI Prediction Service
==========================

REST API for NVIDIA stock price predictions using trained models.

Usage:
    # Development
    python scripts/api.py

    # Production
    uvicorn scripts.api:app --host 0.0.0.0 --port 8000

Endpoints:
    GET  / - Health check
    GET  /models - List available models
    POST /predict - Make predictions
    GET  /model/{model_name}/info - Model information

Author: Senior ML Engineer
Date: February 2026
"""

import sys
from pathlib import Path
from typing import Dict, List, Any, Optional
import pandas as pd
from datetime import datetime
import json

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn

from src.utils.model_utils import load_model, ModelRegistry
from src.data.loader import NVDADataLoader
from src.features.engineering import FeatureEngineer
from src.evaluation.metrics import comprehensive_evaluation

# Initialize FastAPI app
app = FastAPI(
    title="NVIDIA Stock Prediction API",
    description="Real-time stock price prediction service using machine learning",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables
model_registry = ModelRegistry()
available_models = {}
data_loader = None
feature_engineer = None


class PredictionRequest(BaseModel):
    """Request model for predictions"""
    model_name: str = Field(..., description="Name of the model to use")
    features: Optional[Dict[str, Any]] = Field(None, description="Custom features (optional)")
    use_latest_data: bool = Field(True, description="Use latest market data")

    class Config:
        schema_extra = {
            "example": {
                "model_name": "xgboost_model",
                "use_latest_data": True
            }
        }


class PredictionResponse(BaseModel):
    """Response model for predictions"""
    model_name: str
    prediction: float
    confidence: Optional[float] = None
    features_used: List[str]
    timestamp: str
    metadata: Dict[str, Any]


def load_available_models():
    """Load all available trained models"""
    global available_models

    models_dir = Path("models")
    if not models_dir.exists():
        return

    # Look for model files
    model_extensions = ['.pkl', '.model', '.joblib']
    model_files = []

    for ext in model_extensions:
        model_files.extend(models_dir.glob(f"*{ext}"))

    for model_file in model_files:
        try:
            model_name = model_file.stem
            model = load_model(str(model_file))
            available_models[model_name] = {
                'model': model,
                'path': model_file,
                'loaded_at': datetime.now().isoformat()
            }
            print(f"✅ Loaded model: {model_name}")
        except Exception as e:
            print(f"⚠️ Failed to load model {model_file}: {e}")


def get_latest_market_data() -> pd.DataFrame:
    """Get latest market data for feature engineering"""
    global data_loader

    if data_loader is None:
        data_path = "data/raw/NVDA_yfinance_clean.csv"
        if not Path(data_path).exists():
            raise HTTPException(status_code=404, detail="Market data not found")
        data_loader = NVDADataLoader(data_path)

    df, _ = data_loader.load_and_validate(verbose=False)
    return df


def prepare_features(df: pd.DataFrame) -> tuple:
    """Prepare features for prediction"""
    global feature_engineer

    if feature_engineer is None:
        feature_engineer = FeatureEngineer()

    # Create features
    df_features = feature_engineer.create_features(df)

    # Get latest row and prepare features
    latest_row = df_features.iloc[-1:]

    # Define feature columns (exclude OHLCV and target)
    exclude_cols = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'target']
    feature_cols = [col for col in df_features.columns if col not in exclude_cols]

    # Extract features
    features = latest_row[feature_cols].iloc[0].to_dict()

    return features, feature_cols


@app.on_event("startup")
async def startup_event():
    """Load models on startup"""
    load_available_models()


@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "message": "NVIDIA Stock Prediction API",
        "status": "healthy",
        "version": "1.0.0",
        "models_loaded": len(available_models),
        "docs": "/docs"
    }


@app.get("/models")
async def list_models():
    """List available models"""
    models_info = []
    for name, info in available_models.items():
        models_info.append({
            "name": name,
            "path": str(info['path']),
            "loaded_at": info['loaded_at']
        })

    return {
        "models": models_info,
        "count": len(models_info)
    }


@app.get("/model/{model_name}/info")
async def get_model_info(model_name: str):
    """Get information about a specific model"""
    if model_name not in available_models:
        raise HTTPException(status_code=404, detail=f"Model '{model_name}' not found")

    model_info = available_models[model_name]

    # Try to get model metadata
    metadata = {}
    try:
        metadata = load_model(str(model_info['path']), metadata_only=True)
    except:
        pass

    return {
        "model_name": model_name,
        "path": str(model_info['path']),
        "loaded_at": model_info['loaded_at'],
        "metadata": metadata
    }


@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """Make stock price prediction"""
    # Check if model exists
    if request.model_name not in available_models:
        raise HTTPException(status_code=404, detail=f"Model '{request.model_name}' not found")

    try:
        # Get model
        model_info = available_models[request.model_name]
        model = model_info['model']

        # Prepare features
        if request.use_latest_data:
            # Use latest market data
            df = get_latest_market_data()
            features, feature_cols = prepare_features(df)
        else:
            # Use provided features
            if not request.features:
                raise HTTPException(status_code=400, detail="Features must be provided when use_latest_data=False")

            features = request.features
            # For custom features, we assume they match the model's expected features
            # In production, you'd want better validation here
            feature_cols = list(features.keys())

        # Convert to DataFrame for prediction
        features_df = pd.DataFrame([features])

        # Make prediction
        prediction = model.predict(features_df)[0]

        # Calculate confidence (if model supports it)
        confidence = None
        if hasattr(model, 'predict_proba'):
            # For classification-style confidence
            confidence = 0.5  # Placeholder
        elif hasattr(model, 'get_booster'):
            # For XGBoost, we could implement prediction intervals
            confidence = 0.8  # Placeholder

        # Prepare response
        response = PredictionResponse(
            model_name=request.model_name,
            prediction=float(prediction),
            confidence=confidence,
            features_used=feature_cols,
            timestamp=datetime.now().isoformat(),
            metadata={
                "data_source": "latest_market_data" if request.use_latest_data else "custom_features",
                "feature_count": len(feature_cols),
                "model_path": str(model_info['path'])
            }
        )

        return response

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.post("/evaluate")
async def evaluate_model(model_name: str, background_tasks: BackgroundTasks):
    """Trigger model evaluation (async)"""
    if model_name not in available_models:
        raise HTTPException(status_code=404, detail=f"Model '{model_name}' not found")

    # In a real implementation, you'd run this in the background
    # For now, we'll run it synchronously
    try:
        # Load test data
        df = get_latest_market_data()

        # Prepare features
        features, feature_cols = prepare_features(df)
        X = df[feature_cols].iloc[-100:]  # Last 100 days for evaluation
        y = df['Close'].shift(-1).iloc[-100:]  # Next day's price
        X = X[:-1]  # Remove last row (no target)
        y = y.dropna()

        # Get model
        model = available_models[model_name]['model']

        # Evaluate
        metrics = comprehensive_evaluation(y, model.predict(X.iloc[:len(y)]), X.iloc[:len(y)], model_name)

        return {
            "model_name": model_name,
            "evaluation": metrics,
            "test_period": "last_100_days",
            "timestamp": datetime.now().isoformat()
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Evaluation failed: {str(e)}")


@app.get("/health")
async def health_check():
    """Detailed health check"""
    health_status = {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "models_loaded": len(available_models),
        "data_available": Path("data/raw/NVDA_yfinance_clean.csv").exists(),
        "version": "1.0.0"
    }

    # Check if we can make a prediction
    try:
        if available_models:
            test_model = list(available_models.keys())[0]
            # Quick prediction test
            df = get_latest_market_data()
            features, _ = prepare_features(df)
            features_df = pd.DataFrame([features])
            model = available_models[test_model]['model']
            test_pred = model.predict(features_df)[0]
            health_status["prediction_test"] = "passed"
        else:
            health_status["prediction_test"] = "no_models_loaded"
    except Exception as e:
        health_status["status"] = "degraded"
        health_status["prediction_test"] = f"failed: {str(e)}"

    return health_status


if __name__ == "__main__":
    # Run development server
    print("🚀 Starting NVIDIA Stock Prediction API...")
    print("📖 API Documentation: http://localhost:8000/docs")
    print("🔗 Health Check: http://localhost:8000/health")

    uvicorn.run(
        "scripts.api:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )