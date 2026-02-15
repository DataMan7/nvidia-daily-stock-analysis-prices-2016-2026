#!/usr/bin/env python3
"""
Streamlit Model Monitoring Dashboard
===================================

Interactive dashboard for NVIDIA stock prediction model monitoring.
Features real-time predictions, performance visualization, and backtesting.

Usage:
    streamlit run scripts/dashboard.py

Features:
    - Real-time stock price predictions
    - Model performance visualization
    - Backtesting results
    - Feature importance analysis
    - Model comparison

Author: Senior ML Engineer
Date: February 2026
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import json

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

from src.data.loader import NVDADataLoader
from src.features.engineering import FeatureEngineer
from src.utils.model_utils import load_model, compare_models
from src.evaluation.metrics import comprehensive_evaluation
from src.evaluation.backtesting import backtest_predictions

# Page configuration
st.set_page_config(
    page_title="NVIDIA Stock Prediction Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 0.25rem solid #1f77b4;
    }
    .prediction-positive {
        color: #28a745;
        font-weight: bold;
    }
    .prediction-negative {
        color: #dc3545;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_market_data():
    """Load market data with caching"""
    data_path = "data/raw/NVDA_yfinance_clean.csv"
    if not Path(data_path).exists():
        st.error("Market data not found. Please run data download first.")
        return None

    loader = NVDADataLoader(data_path)
    df, report = loader.load_and_validate(verbose=False)

    if not report.is_valid:
        st.warning("Data quality issues detected. Some features may not work correctly.")

    return df

@st.cache_data
def load_available_models():
    """Load available trained models"""
    models_dir = Path("models")
    available_models = {}

    if not models_dir.exists():
        return available_models

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
                'metadata': load_model(str(model_file), metadata_only=True)
            }
        except Exception as e:
            st.warning(f"Could not load model {model_file.name}: {e}")

    return available_models

def prepare_features_for_prediction(df, feature_engineer=None):
    """Prepare features for prediction"""
    if feature_engineer is None:
        feature_engineer = FeatureEngineer()

    # Create features
    df_features = feature_engineer.create_features(df)

    # Get latest data point
    latest_row = df_features.iloc[-1:]

    # Define feature columns
    exclude_cols = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'target']
    feature_cols = [col for col in df_features.columns if col not in exclude_cols]

    return latest_row[feature_cols], feature_cols

def make_prediction(model, features_df):
    """Make prediction with confidence"""
    prediction = model.predict(features_df)[0]

    # Calculate simple confidence based on feature stability
    # In production, you'd use prediction intervals
    confidence = 0.7  # Placeholder

    return float(prediction), confidence

def create_price_chart(df, predictions=None, title="NVIDIA Stock Price"):
    """Create interactive price chart"""
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    # Price line
    fig.add_trace(
        go.Scatter(
            x=df['Date'],
            y=df['Close'],
            name="Close Price",
            line=dict(color='#1f77b4', width=2)
        ),
        secondary_y=False
    )

    # Volume bars
    fig.add_trace(
        go.Bar(
            x=df['Date'],
            y=df['Volume'],
            name="Volume",
            opacity=0.3,
            marker_color='#ff7f0e'
        ),
        secondary_y=True
    )

    # Add predictions if provided
    if predictions:
        pred_dates = list(predictions.keys())
        pred_values = list(predictions.values())

        fig.add_trace(
            go.Scatter(
                x=pred_dates,
                y=pred_values,
                name="Predictions",
                mode="markers+lines",
                line=dict(color='#d62728', width=3, dash='dot'),
                marker=dict(size=8, symbol='diamond')
            ),
            secondary_y=False
        )

    fig.update_layout(
        title=title,
        xaxis_title="Date",
        yaxis_title="Price ($)",
        yaxis2_title="Volume",
        height=500
    )

    return fig

def create_performance_chart(metrics_history):
    """Create performance metrics chart"""
    if not metrics_history:
        return None

    df_metrics = pd.DataFrame(metrics_history)

    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=("MAE Over Time", "R² Over Time", "Directional Accuracy", "Sharpe Ratio")
    )

    # MAE
    fig.add_trace(
        go.Scatter(x=df_metrics['date'], y=df_metrics['mae'], name="MAE", line=dict(color='#1f77b4')),
        row=1, col=1
    )

    # R²
    fig.add_trace(
        go.Scatter(x=df_metrics['date'], y=df_metrics['r2'], name="R²", line=dict(color='#ff7f0e')),
        row=1, col=2
    )

    # Directional Accuracy
    fig.add_trace(
        go.Scatter(x=df_metrics['date'], y=df_metrics['directional_accuracy'], name="Dir Acc", line=dict(color='#2ca02c')),
        row=2, col=1
    )

    # Sharpe Ratio (if available)
    if 'sharpe_ratio' in df_metrics.columns:
        fig.add_trace(
            go.Scatter(x=df_metrics['date'], y=df_metrics['sharpe_ratio'], name="Sharpe", line=dict(color='#d62728')),
            row=2, col=2
        )

    fig.update_layout(height=600, showlegend=False)
    return fig

def main():
    """Main dashboard function"""

    # Header
    st.markdown('<h1 class="main-header">📈 NVIDIA Stock Prediction Dashboard</h1>', unsafe_allow_html=True)

    # Sidebar
    st.sidebar.title("🎛️ Controls")

    # Load data and models
    with st.spinner("Loading data and models..."):
        df = load_market_data()
        available_models = load_available_models()

    if df is None:
        st.error("Could not load market data. Please check data/raw/NVDA_yfinance_clean.csv")
        return

    # Model selection
    if available_models:
        model_names = list(available_models.keys())
        selected_model = st.sidebar.selectbox("Select Model", model_names)

        model_info = available_models[selected_model]
        st.sidebar.markdown("---")
        st.sidebar.markdown("**Model Info:**")
        st.sidebar.text(f"Path: {model_info['path'].name}")
        if 'metadata' in model_info and model_info['metadata']:
            metadata = model_info['metadata']
            if 'performance' in metadata:
                perf = metadata['performance']
                st.sidebar.metric("Train R²", ".3f")
    else:
        st.sidebar.error("No trained models found in /models directory")
        st.sidebar.info("Run training scripts to create models")
        return

    # Main content
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Predictions", "📈 Charts", "🔍 Analysis", "⚙️ Settings"])

    with tab1:
        st.header("Real-time Stock Price Predictions")

        col1, col2, col3 = st.columns(3)

        # Current price info
        latest_data = df.iloc[-1]
        current_price = latest_data['Close']

        with col1:
            st.metric("Current Price", f"${current_price:.2f}")

        # Make prediction
        if st.button("🔮 Generate Prediction", type="primary"):
            with st.spinner("Making prediction..."):
                try:
                    # Prepare features
                    features_df, feature_cols = prepare_features_for_prediction(df)

                    # Make prediction
                    model = model_info['model']
                    prediction, confidence = make_prediction(model, features_df)

                    # Display results
                    price_change = prediction - current_price
                    change_pct = (price_change / current_price) * 100

                    with col2:
                        st.metric(
                            "Predicted Price",
                            f"${prediction:.2f}",
                            f"{price_change:+.2f} ({change_pct:+.2f}%)",
                            delta_color="normal"
                        )

                    with col3:
                        st.metric("Confidence", f"{confidence:.1%}")

                    # Prediction interpretation
                    if price_change > 0:
                        st.success(f"📈 Bullish signal: Expected price increase of ${price_change:.2f}")
                    else:
                        st.error(f"📉 Bearish signal: Expected price decrease of ${abs(price_change):.2f}")

                except Exception as e:
                    st.error(f"Prediction failed: {e}")

    with tab2:
        st.header("Stock Price Visualization")

        # Date range selector
        date_range = st.selectbox(
            "Time Period",
            ["1 Month", "3 Months", "6 Months", "1 Year", "All Time"],
            index=2
        )

        # Filter data
        end_date = df['Date'].max()
        if date_range == "1 Month":
            start_date = end_date - pd.DateOffset(months=1)
        elif date_range == "3 Months":
            start_date = end_date - pd.DateOffset(months=3)
        elif date_range == "6 Months":
            start_date = end_date - pd.DateOffset(months=6)
        elif date_range == "1 Year":
            start_date = end_date - pd.DateOffset(years=1)
        else:
            start_date = df['Date'].min()

        filtered_df = df[df['Date'] >= start_date]

        # Create chart
        fig = create_price_chart(filtered_df, title=f"NVIDIA Stock Price - {date_range}")
        st.plotly_chart(fig, use_container_width=True)

        # Technical indicators
        st.subheader("Technical Indicators")
        tech_cols = st.multiselect(
            "Select Indicators",
            ["MA_7", "MA_50", "RSI", "BB_Upper", "BB_Lower"],
            default=["MA_7", "RSI"]
        )

        if tech_cols:
            # Calculate indicators
            feature_engineer = FeatureEngineer()
            df_with_indicators = feature_engineer.create_features(filtered_df)

            # Plot indicators
            fig_indicators = go.Figure()

            # Price
            fig_indicators.add_trace(
                go.Scatter(
                    x=df_with_indicators['Date'],
                    y=df_with_indicators['Close'],
                    name="Close Price",
                    line=dict(color='#1f77b4')
                )
            )

            # Add selected indicators
            colors = ['#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
            for i, col in enumerate(tech_cols):
                if col in df_with_indicators.columns:
                    fig_indicators.add_trace(
                        go.Scatter(
                            x=df_with_indicators['Date'],
                            y=df_with_indicators[col],
                            name=col,
                            line=dict(color=colors[i % len(colors)])
                        )
                    )

            fig_indicators.update_layout(
                title="Technical Indicators",
                xaxis_title="Date",
                yaxis_title="Value",
                height=400
            )

            st.plotly_chart(fig_indicators, use_container_width=True)

    with tab3:
        st.header("Model Analysis")

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Model Performance")

            # Quick evaluation on recent data
            if st.button("Evaluate Model Performance"):
                with st.spinner("Evaluating model..."):
                    try:
                        # Use last 100 days for evaluation
                        eval_df = df.iloc[-100:]
                        features_df, feature_cols = prepare_features_for_prediction(eval_df)

                        # Prepare evaluation data
                        X_eval = eval_df[feature_cols].iloc[:-1]  # Remove last row
                        y_eval = eval_df['Close'].shift(-1).iloc[:-1].dropna()  # Next day's price
                        X_eval = X_eval.iloc[:len(y_eval)]

                        # Evaluate
                        model = model_info['model']
                        y_pred = model.predict(X_eval)
                        metrics = comprehensive_evaluation(y_eval, y_pred, X_eval, selected_model)

                        # Display metrics
                        st.metric("MAE", f"${metrics['mae']:.4f}")
                        st.metric("R²", f"{metrics['r2']:.4f}")
                        st.metric("Directional Accuracy", f"{metrics['directional_accuracy']:.1%}")

                    except Exception as e:
                        st.error(f"Evaluation failed: {e}")

        with col2:
            st.subheader("Backtesting")

            if st.button("Run Backtest"):
                with st.spinner("Running backtest..."):
                    try:
                        # Simple backtest on recent data
                        backtest_df = df.iloc[-50:]  # Last 50 days

                        # Generate signals (simplified)
                        features_df, _ = prepare_features_for_prediction(backtest_df)
                        signals = model_info['model'].predict(features_df)

                        # Run backtest
                        result = backtest_predictions(
                            predictions=signals,
                            actual_prices=backtest_df['Close'],
                            dates=backtest_df['Date'],
                            initial_capital=10000
                        )

                        # Display results
                        st.metric("Total Return", f"{result.total_return:.1%}")
                        st.metric("Annualized Return", f"{result.annualized_return:.1%}")
                        st.metric("Sharpe Ratio", f"{result.sharpe_ratio:.2f}")
                        st.metric("Win Rate", f"{result.win_rate:.1%}")

                    except Exception as e:
                        st.error(f"Backtest failed: {e}")

        # Model comparison
        st.subheader("Model Comparison")
        if len(available_models) > 1:
            models_to_compare = st.multiselect(
                "Select models to compare",
                list(available_models.keys()),
                default=list(available_models.keys())[:2]
            )

            if len(models_to_compare) > 1 and st.button("Compare Models"):
                try:
                    # Simple comparison on recent data
                    eval_df = df.iloc[-50:]
                    features_df, feature_cols = prepare_features_for_prediction(eval_df)
                    X_eval = eval_df[feature_cols].iloc[:-1]
                    y_eval = eval_df['Close'].shift(-1).iloc[:-1].dropna()
                    X_eval = X_eval.iloc[:len(y_eval)]

                    comparison_data = []
                    for model_name in models_to_compare:
                        model = available_models[model_name]['model']
                        y_pred = model.predict(X_eval)
                        metrics = comprehensive_evaluation(y_eval, y_pred, X_eval, model_name)
                        comparison_data.append({
                            'Model': model_name,
                            'MAE': metrics['mae'],
                            'R²': metrics['r2'],
                            'Directional_Acc': metrics['directional_accuracy']
                        })

                    comparison_df = pd.DataFrame(comparison_data)
                    st.dataframe(comparison_df.style.highlight_min(axis=0, subset=['MAE']))

                except Exception as e:
                    st.error(f"Comparison failed: {e}")
        else:
            st.info("Need at least 2 models for comparison")

    with tab4:
        st.header("Settings & Information")

        st.subheader("System Information")
        st.write(f"**Data Points:** {len(df):,}")
        st.write(f"**Date Range:** {df['Date'].min()} to {df['Date'].max()}")
        st.write(f"**Models Loaded:** {len(available_models)}")
        st.write(f"**Selected Model:** {selected_model}")

        st.subheader("Model Details")
        if 'metadata' in model_info and model_info['metadata']:
            metadata = model_info['metadata']
            st.json(metadata)
        else:
            st.info("No metadata available for this model")

        st.subheader("Feature Information")
        try:
            features_df, feature_cols = prepare_features_for_prediction(df)
            st.write(f"**Number of Features:** {len(feature_cols)}")
            st.write("**Feature Names:**")
            st.code("\n".join(feature_cols))
        except Exception as e:
            st.error(f"Could not load feature information: {e}")

if __name__ == "__main__":
    main()