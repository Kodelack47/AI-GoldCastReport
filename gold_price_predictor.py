"""
==============================================================================
  Application:  GoldCast – Gold Price Prediction Using Machine Learning
  Author:       Ousmana Tunkara
  Date:         June 2026
  Description:  Predicts the next-day closing price of gold (XAU/USD) using
                a Random Forest Regressor trained on historical OHLCV data
                and derived technical indicators. Outputs evaluation metrics,
                a prediction for the next trading day, and two charts:
                  1. Actual vs. Predicted prices over the test set
                  2. Feature importance bar chart
==============================================================================
"""

# ── Standard library ──────────────────────────────────────────────────────
import os
import warnings
warnings.filterwarnings("ignore")

# ── Third-party libraries ─────────────────────────────────────────────────
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")          # non-interactive backend (works without display)
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (mean_absolute_error,
                             mean_squared_error,
                             r2_score)

# ── Output directory ──────────────────────────────────────────────────────
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "goldcast_output")
os.makedirs(OUT_DIR, exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
#  SECTION 1 – DATA GENERATION
#  Simulate realistic gold (XAU/USD) daily OHLCV data from Jan 2019 to
#  Jun 2026 using a geometric Brownian motion model seeded for reproducibility.
#  In a production environment this block is replaced by a CSV load or an
#  API call (e.g., Yahoo Finance, Quandl).
# ─────────────────────────────────────────────────────────────────────────────

def generate_gold_data(start="2019-01-01", end="2026-06-01", seed=42):
    """
    Generate synthetic but realistic daily gold price data.
    Uses geometric Brownian motion so the price series has the same
    statistical properties as real commodity data (drift + volatility).
    """
    np.random.seed(seed)

    dates = pd.bdate_range(start=start, end=end)   # business days only
    n = len(dates)

    # Gold GBM parameters (calibrated to historical gold 2019-2024)
    S0   = 1280.0   # starting price ~Jan 2019 gold price (USD/oz)
    mu   = 0.00015  # daily drift
    sigma = 0.0085  # daily volatility

    # Simulate closing prices
    daily_returns = np.random.normal(mu, sigma, n)
    close = S0 * np.cumprod(1 + daily_returns)

    # Simulate realistic OHLV from close
    daily_range = close * np.abs(np.random.normal(0.007, 0.003, n))
    high  = close + daily_range * np.random.uniform(0.3, 0.7, n)
    low   = close - daily_range * np.random.uniform(0.3, 0.7, n)
    open_ = low   + (high - low) * np.random.uniform(0.2, 0.8, n)
    volume = np.random.randint(150_000, 600_000, n).astype(float)

    df = pd.DataFrame({
        "Date":   dates,
        "Open":   np.round(open_, 2),
        "High":   np.round(high,  2),
        "Low":    np.round(low,   2),
        "Close":  np.round(close, 2),
        "Volume": volume,
    }).set_index("Date")

    return df


# ─────────────────────────────────────────────────────────────────────────────
#  SECTION 2 – FEATURE ENGINEERING
#  Derive technical indicators used as model features.
#  All indicators are standard in quantitative finance.
# ─────────────────────────────────────────────────────────────────────────────

def add_features(df):
    """
    Add technical indicator features to the dataframe.

    Features created:
      - MA_5, MA_10, MA_20  : Simple moving averages
      - EMA_12, EMA_26      : Exponential moving averages
      - MACD                : Moving Average Convergence Divergence
      - RSI_14              : Relative Strength Index (14-day)
      - BB_upper, BB_lower  : Bollinger Bands (20-day, 2 std dev)
      - ATR_14              : Average True Range (14-day)
      - Price_Change        : Daily close-to-close change
      - Return_1d           : 1-day log return
      - High_Low_Range      : Daily range (High - Low)
      - Lag_1, Lag_2, Lag_3 : Lagged closing prices (1, 2, 3 days)
    """
    d = df.copy()

    # Moving averages
    d["MA_5"]  = d["Close"].rolling(5).mean()
    d["MA_10"] = d["Close"].rolling(10).mean()
    d["MA_20"] = d["Close"].rolling(20).mean()

    # Exponential moving averages
    d["EMA_12"] = d["Close"].ewm(span=12, adjust=False).mean()
    d["EMA_26"] = d["Close"].ewm(span=26, adjust=False).mean()

    # MACD line
    d["MACD"] = d["EMA_12"] - d["EMA_26"]

    # RSI (14-day)
    delta  = d["Close"].diff()
    gain   = delta.clip(lower=0).rolling(14).mean()
    loss   = (-delta.clip(upper=0)).rolling(14).mean()
    rs     = gain / (loss + 1e-9)
    d["RSI_14"] = 100 - (100 / (1 + rs))

    # Bollinger Bands
    rolling_std = d["Close"].rolling(20).std()
    d["BB_upper"] = d["MA_20"] + 2 * rolling_std
    d["BB_lower"] = d["MA_20"] - 2 * rolling_std

    # Average True Range
    hl  = d["High"] - d["Low"]
    hc  = (d["High"] - d["Close"].shift(1)).abs()
    lc  = (d["Low"]  - d["Close"].shift(1)).abs()
    tr  = pd.concat([hl, hc, lc], axis=1).max(axis=1)
    d["ATR_14"] = tr.rolling(14).mean()

    # Price dynamics
    d["Price_Change"]   = d["Close"].diff()
    d["Return_1d"]      = np.log(d["Close"] / d["Close"].shift(1))
    d["High_Low_Range"] = d["High"] - d["Low"]

    # Lagged close prices
    d["Lag_1"] = d["Close"].shift(1)
    d["Lag_2"] = d["Close"].shift(2)
    d["Lag_3"] = d["Close"].shift(3)

    # Target: next-day closing price
    d["Target"] = d["Close"].shift(-1)

    # Drop rows with NaN (from rolling windows and shifts)
    d.dropna(inplace=True)

    return d


# ─────────────────────────────────────────────────────────────────────────────
#  SECTION 3 – MODEL TRAINING AND EVALUATION
# ─────────────────────────────────────────────────────────────────────────────

def train_and_evaluate(df):
    """
    Split data, train a Random Forest Regressor, and evaluate on the test set.
    Returns the trained model, scaler, predictions, and feature names.
    """

    feature_cols = [
        "Open", "High", "Low", "Close", "Volume",
        "MA_5", "MA_10", "MA_20",
        "EMA_12", "EMA_26", "MACD",
        "RSI_14",
        "BB_upper", "BB_lower",
        "ATR_14",
        "Price_Change", "Return_1d", "High_Low_Range",
        "Lag_1", "Lag_2", "Lag_3",
    ]

    X = df[feature_cols].values
    y = df["Target"].values

    # Chronological 80/20 split (no shuffle — time-series data)
    split = int(len(X) * 0.80)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    dates_test = df.index[split:]

    # Scale features
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)

    # ── Random Forest Regressor ──────────────────────────────────────────
    rf = RandomForestRegressor(
        n_estimators=200,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )
    rf.fit(X_train_s, y_train)
    y_pred_rf = rf.predict(X_test_s)

    # ── Gradient Boosting Regressor (secondary model) ────────────────────
    gb = GradientBoostingRegressor(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=4,
        random_state=42
    )
    gb.fit(X_train_s, y_train)
    y_pred_gb = gb.predict(X_test_s)

    # ── Ensemble: simple average of both models ──────────────────────────
    y_pred = (y_pred_rf + y_pred_gb) / 2

    # ── Metrics ──────────────────────────────────────────────────────────
    mae  = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2   = r2_score(y_test, y_pred)
    mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100

    metrics = {
        "MAE":  round(mae,  4),
        "RMSE": round(rmse, 4),
        "R²":   round(r2,   4),
        "MAPE": round(mape, 4),
    }

    return rf, scaler, feature_cols, X_test_s, y_test, y_pred, dates_test, metrics


# ─────────────────────────────────────────────────────────────────────────────
#  SECTION 4 – NEXT-DAY PREDICTION
# ─────────────────────────────────────────────────────────────────────────────

def predict_next_day(df, model, scaler, feature_cols):
    """
    Use the last available row of features to predict tomorrow's gold price.
    """
    last_row = df[feature_cols].iloc[-1].values.reshape(1, -1)
    last_scaled = scaler.transform(last_row)
    predicted_price = model.predict(last_scaled)[0]
    return round(predicted_price, 2)


# ─────────────────────────────────────────────────────────────────────────────
#  SECTION 5 – VISUALIZATIONS
# ─────────────────────────────────────────────────────────────────────────────

def plot_actual_vs_predicted(dates, y_actual, y_pred, metrics, save_path):
    """
    Line chart comparing actual vs. predicted gold prices on the test set.
    """
    fig, ax = plt.subplots(figsize=(13, 5))

    ax.plot(dates, y_actual, label="Actual Price",    color="#1a6faf", linewidth=1.5)
    ax.plot(dates, y_pred,   label="Predicted Price", color="#e8703a",
            linewidth=1.2, linestyle="--", alpha=0.85)

    ax.set_title("GoldCast — Actual vs. Predicted Gold Price (Test Set)",
                 fontsize=14, fontweight="bold", pad=12)
    ax.set_xlabel("Date", fontsize=11)
    ax.set_ylabel("Price (USD / oz)", fontsize=11)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    plt.xticks(rotation=30, ha="right")

    # Metrics annotation box
    metrics_text = (
        f"MAE  = ${metrics['MAE']:,.2f}\n"
        f"RMSE = ${metrics['RMSE']:,.2f}\n"
        f"MAPE = {metrics['MAPE']:.2f}%\n"
        f"R²   = {metrics['R²']:.4f}"
    )
    ax.text(0.02, 0.97, metrics_text, transform=ax.transAxes,
            fontsize=9, verticalalignment="top",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="#f0f4f8",
                      edgecolor="#cccccc", alpha=0.9))

    ax.legend(fontsize=10)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  [Chart 1 saved] {save_path}")


def plot_feature_importance(model, feature_cols, save_path, top_n=15):
    """
    Horizontal bar chart of the top N most important features.
    """
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1][:top_n]
    top_features = [feature_cols[i] for i in indices]
    top_values   = importances[indices]

    palette = sns.color_palette("Blues_d", top_n)

    fig, ax = plt.subplots(figsize=(9, 6))
    bars = ax.barh(range(top_n), top_values[::-1],
                   color=list(reversed(palette)), edgecolor="white")
    ax.set_yticks(range(top_n))
    ax.set_yticklabels(top_features[::-1], fontsize=10)
    ax.set_xlabel("Feature Importance (Mean Decrease Impurity)", fontsize=10)
    ax.set_title("GoldCast — Top Feature Importances (Random Forest)",
                 fontsize=13, fontweight="bold", pad=10)
    ax.grid(axis="x", linestyle="--", alpha=0.4)

    # Add value labels
    for bar, val in zip(bars, top_values[::-1]):
        ax.text(val + 0.001, bar.get_y() + bar.get_height() / 2,
                f"{val:.4f}", va="center", fontsize=8, color="#333333")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  [Chart 2 saved] {save_path}")


# ─────────────────────────────────────────────────────────────────────────────
#  SECTION 6 – MAIN PIPELINE
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("=" * 65)
    print("  GoldCast — Gold Price Prediction Application")
    print("  Author: Ousmana Tunkara  |  June 2026")
    print("=" * 65)

    # Step 1: Generate / load data
    print("\n[1] Generating gold price dataset...")
    raw_df = generate_gold_data()
    print(f"    Records generated : {len(raw_df):,}")
    print(f"    Date range        : {raw_df.index[0].date()} → {raw_df.index[-1].date()}")

    # Step 2: Feature engineering
    print("\n[2] Engineering features...")
    df = add_features(raw_df)
    print(f"    Records after cleaning : {len(df):,}")
    print(f"    Features created       : 21")

    # Step 3: Train and evaluate
    print("\n[3] Training models (Random Forest + Gradient Boosting)...")
    rf_model, scaler, feature_cols, X_test, y_test, y_pred, dates_test, metrics = \
        train_and_evaluate(df)

    print("\n── Evaluation Metrics (Ensemble — Test Set) ──")
    for k, v in metrics.items():
        unit = "%" if k == "MAPE" else ("" if k == "R²" else " USD")
        print(f"    {k:<6} = {v}{unit}")

    # Step 4: Next-day prediction
    print("\n[4] Predicting next trading day's closing price...")
    next_price = predict_next_day(df, rf_model, scaler, feature_cols)
    last_close  = df["Close"].iloc[-1]
    last_date   = df.index[-1].strftime("%Y-%m-%d")
    direction   = "▲ UP" if next_price > last_close else "▼ DOWN"

    print(f"\n    Last known close ({last_date}) : ${last_close:,.2f}")
    print(f"    Predicted next-day close       : ${next_price:,.2f}  {direction}")
    print(f"    Change                         : ${next_price - last_close:+,.2f}")

    # Step 5: Charts
    print("\n[5] Generating charts...")
    chart1 = os.path.join(OUT_DIR, "actual_vs_predicted.png")
    chart2 = os.path.join(OUT_DIR, "feature_importance.png")

    plot_actual_vs_predicted(dates_test, y_test, y_pred, metrics, chart1)
    plot_feature_importance(rf_model, feature_cols, chart2)

    print("\n" + "=" * 65)
    print(f"  All outputs saved to: {OUT_DIR}")
    print("=" * 65)

    return metrics, next_price, last_close


if __name__ == "__main__":
    main()
