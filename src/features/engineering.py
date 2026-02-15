import pandas as pd
import numpy as np

class FeatureEngineer:
    """
    Handles feature engineering for stock price data.
    Designed to prevent data leakage by using only past data.
    """
    
    def __init__(self):
        pass
        
    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generates technical indicators and lag features.
        
        Args:
            df: DataFrame with 'Open', 'High', 'Low', 'Close', 'Volume' columns.
            
        Returns:
            DataFrame with added features.
        """
        # Work on a copy to avoid SettingWithCopy warnings
        df = df.copy()
        
        # Ensure data is sorted by date
        if 'Date' in df.columns:
            df = df.sort_values('Date')
        else:
            df = df.sort_index()
            
        # 1. Returns
        # Daily Return: (Close_t - Close_{t-1}) / Close_{t-1}
        df['Daily_Return'] = df['Close'].pct_change()
        
        # Previous Day's Return (Lag 1 of Daily Return)
        # This is safe to use for predicting T (if T is today) or T+1
        df['Prev_Day_Return'] = df['Daily_Return'].shift(1)
        
        # 2. Moving Averages (Trend)
        # Rolling mean of Close price. Value at T includes T, T-1, ...
        df['MA_7'] = df['Close'].rolling(window=7).mean()
        df['MA_50'] = df['Close'].rolling(window=50).mean()
        df['MA_200'] = df['Close'].rolling(window=200).mean()
        
        # 3. RSI (Relative Strength Index - Momentum)
        df = self._calculate_rsi(df)
        
        # 4. Bollinger Bands (Volatility)
        df = self._calculate_bollinger_bands(df)
        
        # 5. Lag Features (Critical for Time Series)
        # We shift features so that at row T, we have values from T-1, T-2, etc.
        # This ensures we don't use T's value to predict T (if that's the goal) 
        for lag in [1, 2, 3, 5, 7]:
            df[f'Close_Lag_{lag}'] = df['Close'].shift(lag)
            df[f'Volume_Lag_{lag}'] = df['Volume'].shift(lag)
            
        return df

    def _calculate_rsi(self, df: pd.DataFrame, window: int = 14) -> pd.DataFrame:
        """Calculates Relative Strength Index (RSI)"""
        delta = df['Close'].diff()
        
        # Separate gains and losses
        gain = delta.where(delta > 0, 0.0)
        loss = -delta.where(delta < 0, 0.0)
        
        # Calculate Exponential Moving Average (EMA)
        # Wilder's smoothing method is commonly used for RSI
        avg_gain = gain.ewm(alpha=1/window, min_periods=window).mean()
        avg_loss = loss.ewm(alpha=1/window, min_periods=window).mean()
        
        rs = avg_gain / avg_loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        # Fill initial NaNs with 50 (neutral) or drop later
        df['RSI'] = df['RSI'].fillna(50)
        
        return df

    def _calculate_bollinger_bands(self, df: pd.DataFrame, window: int = 20, num_std: int = 2) -> pd.DataFrame:
        """Calculates Bollinger Bands"""
        rolling_mean = df['Close'].rolling(window=window).mean()
        rolling_std = df['Close'].rolling(window=window).std()
        
        df['BB_Middle'] = rolling_mean
        df['BB_Upper'] = rolling_mean + (rolling_std * num_std)
        df['BB_Lower'] = rolling_mean - (rolling_std * num_std)
        
        # Band Width (Volatility indicator)
        df['BB_Width'] = (df['BB_Upper'] - df['BB_Lower']) / rolling_mean
        
        return df