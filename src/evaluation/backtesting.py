"""
Backtesting Module
==================

Financial backtesting framework for evaluating trading strategies.
Simulates realistic trading with transaction costs, slippage, and risk management.

Author: Senior ML Engineer
Date: February 2026
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Callable
from dataclasses import dataclass
from datetime import datetime
import warnings


@dataclass
class Trade:
    """Represents a single trade"""
    entry_date: pd.Timestamp
    exit_date: pd.Timestamp
    entry_price: float
    exit_price: float
    position_size: float  # Positive for long, negative for short
    pnl: float
    pnl_pct: float
    holding_period: int  # Days

    @property
    def is_long(self) -> bool:
        return self.position_size > 0

    @property
    def is_short(self) -> bool:
        return self.position_size < 0

    @property
    def is_profitable(self) -> bool:
        return self.pnl > 0


@dataclass
class BacktestResult:
    """Results of a backtest"""
    trades: List[Trade]
    total_return: float
    annualized_return: float
    volatility: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    profit_factor: float
    total_trades: int
    avg_trade_duration: float
    calmar_ratio: float
    sortino_ratio: float

    # Equity curve
    equity_curve: pd.Series
    drawdown_curve: pd.Series


class Backtester:
    """
    Financial backtesting engine for evaluating trading strategies.

    Features:
    - Transaction costs
    - Slippage simulation
    - Position sizing
    - Risk management
    - Performance analytics
    """

    def __init__(
        self,
        initial_capital: float = 100000,
        commission_per_trade: float = 0.001,  # 0.1% per trade
        slippage_bps: float = 5,  # 5 basis points slippage
        max_position_size: float = 0.1,  # Max 10% of capital per trade
        risk_free_rate: float = 0.02
    ):
        """
        Initialize backtester.

        Args:
            initial_capital: Starting capital
            commission_per_trade: Commission as fraction (0.001 = 0.1%)
            slippage_bps: Slippage in basis points
            max_position_size: Maximum position size as fraction of capital
            risk_free_rate: Risk-free rate for Sharpe ratio
        """
        self.initial_capital = initial_capital
        self.commission_per_trade = commission_per_trade
        self.slippage_bps = slippage_bps
        self.max_position_size = max_position_size
        self.risk_free_rate = risk_free_rate

    def backtest_strategy(
        self,
        predictions: pd.Series,
        actual_prices: pd.Series,
        dates: pd.Series,
        signal_threshold: float = 0.0,
        min_holding_period: int = 1,
        max_holding_period: int = 30
    ) -> BacktestResult:
        """
        Backtest a prediction-based trading strategy.

        Args:
            predictions: Model predictions (expected returns)
            actual_prices: Actual price series
            dates: Date series
            signal_threshold: Minimum signal strength to trade
            min_holding_period: Minimum days to hold position
            max_holding_period: Maximum days to hold position

        Returns:
            BacktestResult with performance metrics
        """
        if len(predictions) != len(actual_prices) or len(predictions) != len(dates):
            raise ValueError("All input series must have same length")

        # Initialize
        capital = self.initial_capital
        position = 0  # Current position size
        entry_price = 0
        entry_date = None
        trades = []
        equity_curve = [capital]

        for i in range(len(predictions)):
            current_price = actual_prices.iloc[i]
            current_date = dates.iloc[i]
            signal = predictions.iloc[i]

            # Check for exit signal
            if position != 0:
                # Exit if signal reverses or max holding period reached
                days_held = (current_date - entry_date).days if entry_date else 0

                should_exit = (
                    (position > 0 and signal < -signal_threshold) or  # Long position, negative signal
                    (position < 0 and signal > signal_threshold) or   # Short position, positive signal
                    days_held >= max_holding_period or
                    days_held >= min_holding_period  # Minimum hold period
                )

                if should_exit:
                    # Execute exit
                    exit_price = self._apply_slippage(current_price, position > 0)
                    pnl = position * (exit_price - entry_price)
                    pnl -= abs(position * exit_price) * self.commission_per_trade  # Commission

                    # Record trade
                    trade = Trade(
                        entry_date=entry_date,
                        exit_date=current_date,
                        entry_price=entry_price,
                        exit_price=exit_price,
                        position_size=position,
                        pnl=pnl,
                        pnl_pct=pnl / abs(position * entry_price),
                        holding_period=days_held
                    )
                    trades.append(trade)

                    # Update capital
                    capital += pnl
                    position = 0
                    entry_date = None

            # Check for entry signal
            elif abs(signal) > signal_threshold:
                # Calculate position size
                position_value = min(
                    capital * self.max_position_size,
                    capital * 0.5  # Don't risk more than 50% on single trade
                )

                # Determine direction
                if signal > 0:  # Bullish
                    position = position_value / current_price
                    entry_price = self._apply_slippage(current_price, False)  # Buy slippage
                else:  # Bearish
                    position = -position_value / current_price
                    entry_price = self._apply_slippage(current_price, True)   # Sell slippage

                # Apply commission
                capital -= abs(position * entry_price) * self.commission_per_trade
                entry_date = current_date

            # Record equity
            equity_curve.append(capital)

        # Close any open position at end
        if position != 0 and len(actual_prices) > 0:
            final_price = self._apply_slippage(actual_prices.iloc[-1], position < 0)
            pnl = position * (final_price - entry_price)
            pnl -= abs(position * final_price) * self.commission_per_trade

            days_held = (dates.iloc[-1] - entry_date).days if entry_date else 0
            trade = Trade(
                entry_date=entry_date,
                exit_date=dates.iloc[-1],
                entry_price=entry_price,
                exit_price=final_price,
                position_size=position,
                pnl=pnl,
                pnl_pct=pnl / abs(position * entry_price),
                holding_period=days_held
            )
            trades.append(trade)
            capital += pnl
            equity_curve[-1] = capital

        # Calculate performance metrics
        result = self._calculate_metrics(trades, equity_curve, dates)

        return result

    def _apply_slippage(self, price: float, is_sell: bool) -> float:
        """Apply slippage to price"""
        slippage = price * (self.slippage_bps / 10000)  # Convert bps to decimal
        return price + slippage if is_sell else price - slippage

    def _calculate_metrics(
        self,
        trades: List[Trade],
        equity_curve: List[float],
        dates: pd.Series
    ) -> BacktestResult:
        """Calculate comprehensive performance metrics"""

        if not trades:
            # No trades - return zero metrics
            empty_equity = pd.Series([self.initial_capital] * len(dates), index=dates)
            empty_drawdown = pd.Series([0.0] * len(dates), index=dates)

            return BacktestResult(
                trades=[],
                total_return=0.0,
                annualized_return=0.0,
                volatility=0.0,
                sharpe_ratio=0.0,
                max_drawdown=0.0,
                win_rate=0.0,
                profit_factor=0.0,
                total_trades=0,
                avg_trade_duration=0.0,
                calmar_ratio=0.0,
                sortino_ratio=0.0,
                equity_curve=empty_equity,
                drawdown_curve=empty_drawdown
            )

        # Basic metrics
        total_return = (equity_curve[-1] - self.initial_capital) / self.initial_capital
        total_days = (dates.iloc[-1] - dates.iloc[0]).days
        annualized_return = (1 + total_return) ** (365 / total_days) - 1 if total_days > 0 else 0

        # Equity curve
        equity_series = pd.Series(equity_curve, index=dates)

        # Returns
        returns = equity_series.pct_change().dropna()
        volatility = returns.std() * np.sqrt(252)  # Annualized

        # Sharpe ratio
        excess_returns = returns - self.risk_free_rate/252
        sharpe_ratio = np.sqrt(252) * excess_returns.mean() / excess_returns.std() if excess_returns.std() > 0 else 0

        # Drawdown
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = drawdown.min()

        # Calmar ratio
        calmar_ratio = annualized_return / abs(max_drawdown) if max_drawdown != 0 else 0

        # Sortino ratio (downside deviation)
        downside_returns = returns[returns < 0]
        sortino_ratio = np.sqrt(252) * excess_returns.mean() / downside_returns.std() if len(downside_returns) > 0 else 0

        # Trade metrics
        winning_trades = [t for t in trades if t.is_profitable]
        losing_trades = [t for t in trades if not t.is_profitable]

        win_rate = len(winning_trades) / len(trades) if trades else 0

        gross_profit = sum(t.pnl for t in winning_trades)
        gross_loss = abs(sum(t.pnl for t in losing_trades))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')

        avg_trade_duration = np.mean([t.holding_period for t in trades]) if trades else 0

        return BacktestResult(
            trades=trades,
            total_return=total_return,
            annualized_return=annualized_return,
            volatility=volatility,
            sharpe_ratio=sharpe_ratio,
            max_drawdown=max_drawdown,
            win_rate=win_rate,
            profit_factor=profit_factor,
            total_trades=len(trades),
            avg_trade_duration=avg_trade_duration,
            calmar_ratio=calmar_ratio,
            sortino_ratio=sortino_ratio,
            equity_curve=equity_series,
            drawdown_curve=drawdown
        )


def backtest_predictions(
    predictions: pd.Series,
    actual_prices: pd.Series,
    dates: pd.Series,
    **backtest_kwargs
) -> BacktestResult:
    """
    Convenience function for backtesting prediction-based strategies.

    Args:
        predictions: Model predictions (expected returns)
        actual_prices: Actual price series
        dates: Date series
        **backtest_kwargs: Arguments for Backtester

    Returns:
        BacktestResult
    """
    backtester = Backtester(**backtest_kwargs)
    return backtester.backtest_strategy(predictions, actual_prices, dates)


def compare_strategies(
    results: Dict[str, BacktestResult]
) -> pd.DataFrame:
    """
    Compare multiple backtest results.

    Args:
        results: Dictionary of strategy names -> BacktestResult

    Returns:
        DataFrame with strategy comparison
    """
    comparison_data = []

    for strategy_name, result in results.items():
        data = {
            'strategy': strategy_name,
            'total_return': result.total_return,
            'annualized_return': result.annualized_return,
            'volatility': result.volatility,
            'sharpe_ratio': result.sharpe_ratio,
            'max_drawdown': result.max_drawdown,
            'win_rate': result.win_rate,
            'profit_factor': result.profit_factor,
            'total_trades': result.total_trades,
            'calmar_ratio': result.calmar_ratio,
            'sortino_ratio': result.sortino_ratio
        }
        comparison_data.append(data)

    df = pd.DataFrame(comparison_data)
    return df.set_index('strategy')


def print_backtest_report(result: BacktestResult, title: str = "Backtest Results") -> None:
    """Print formatted backtest report"""
    print(f"\n{'='*70}")
    print(f"{title}".center(70))
    print(f"{'='*70}")

    print(f"\\n📊 PERFORMANCE METRICS:")
    print(f"  Total Return:       {result.total_return:.2%}")
    print(f"  Annualized Return:  {result.annualized_return:.2%}")
    print(f"  Volatility:         {result.volatility:.2%}")
    print(f"  Sharpe Ratio:       {result.sharpe_ratio:.4f}")
    print(f"  Max Drawdown:       {result.max_drawdown:.2%}")
    print(f"  Calmar Ratio:       {result.calmar_ratio:.4f}")
    print(f"  Sortino Ratio:      {result.sortino_ratio:.4f}")

    print(f"\\n📈 TRADING METRICS:")
    print(f"  Total Trades:       {result.total_trades}")
    print(f"  Win Rate:           {result.win_rate:.1%}")
    print(f"  Profit Factor:      {result.profit_factor:.4f}")
    print(f"  Avg Trade Duration: {result.avg_trade_duration:.1f} days")

    if result.trades:
        winning_trades = [t for t in result.trades if t.is_profitable]
        losing_trades = [t for t in result.trades if not t.is_profitable]

        print(f"\\n💰 TRADE ANALYSIS:")
        print(f"  Winning Trades:     {len(winning_trades)}")
        print(f"  Losing Trades:      {len(losing_trades)}")
        print(f"  Avg Win:           ${np.mean([t.pnl for t in winning_trades]):.2f}")
        print(f"  Avg Loss:          ${np.mean([t.pnl for t in losing_trades]):.2f}")

    print(f"{'='*70}")


if __name__ == "__main__":
    # Example usage
    print("=" * 80)
    print("BACKTESTING MODULE - EXAMPLE USAGE".center(80))
    print("=" * 80)

    # Create sample data
    np.random.seed(42)
    dates = pd.date_range('2020-01-01', periods=200, freq='D')
    prices = 100 + np.random.randn(200).cumsum() + np.sin(np.arange(200) * 0.1) * 5

    # Simulate predictions (with some skill)
    true_returns = pd.Series(prices).pct_change().fillna(0)
    predictions = true_returns + np.random.randn(200) * 0.5  # Add noise

    # Create DataFrame
    df = pd.DataFrame({
        'date': dates,
        'price': prices,
        'prediction': predictions
    })

    # Run backtest
    backtester = Backtester(initial_capital=10000)
    result = backtester.backtest_strategy(
        predictions=df['prediction'],
        actual_prices=df['price'],
        dates=df['date'],
        signal_threshold=0.001  # 0.1% signal threshold
    )

    # Print results
    print_backtest_report(result, "Sample Strategy Backtest")

    # Compare with buy-and-hold
    buy_hold_return = (df['price'].iloc[-1] - df['price'].iloc[0]) / df['price'].iloc[0]
    print(f"\\n📊 COMPARISON:")
    print(f"  Strategy Return: {result.total_return:.2%}")
    print(f"  Buy & Hold:      {buy_hold_return:.2%}")
    print(f"  Outperformance:  {result.total_return - buy_hold_return:.2%}")

    print("\\n" + "=" * 80)
    print("✅ Backtesting example completed!")
    print("=" * 80)