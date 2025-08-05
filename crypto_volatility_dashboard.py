import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_curve, auc
)

# Page config & branding
st.set_page_config(page_title="Crypto Volatility Regime Dashboard", layout="wide")
st.markdown("""
    <div style='padding:10px 0 18px 0;border-bottom:2px solid #222'>
        <h1 style='font-size:2.8rem;color:#fafbfc;font-family:Georgia,serif;'>
            Crypto Volatility Regime Dashboard
        </h1>
        <span style='font-size:1.25rem;color:#bbb;'>Quantitative Risk & Regime Detection Toolkit</span>
    </div>
""", unsafe_allow_html=True)

# Sidebar inputs
st.sidebar.header("Configure Analysis")
symbols = st.sidebar.multiselect(
    "Select Symbol(s)",
    ["BTC-USD", "ETH-USD"],
    default=["BTC-USD"]
)
period = st.sidebar.selectbox("Data Period", ["1y", "2y", "3y", "5y"], index=2)
vol_window = st.sidebar.slider("Volatility Window (days)", 7, 30, 14)
median_window = st.sidebar.slider("Median Window (days)", 100, 500, 365)

@st.cache_data
def get_data(ticker, period):
    df = yf.download(ticker, period=period)
    return df.dropna()

def assign_regimes(df, vol_window, median_window):
    df = df.copy()
    df["Return"] = df["Close"].pct_change()
    df["Volatility"] = df["Return"].rolling(vol_window).std()
    threshold = df["Volatility"].rolling(median_window).median()
    df["Regime"] = np.where(df["Volatility"] > threshold, "High", "Low")
    df = df.dropna()
    df["Regime_Label"] = (df["Regime"] == "High").astype(int)
    return df

def find_balanced_windows(df):
    for v in range(7, 31, 2):
        for m in range(100, 501, 20):
            testdf = assign_regimes(df, v, m)
            unique = set(testdf["Regime_Label"])
            if 0 in unique and 1 in unique:
                return v, m
    return None, None

def export_csv(data):
    buffer = BytesIO()
    data.to_csv(buffer, index=True)
    buffer.seek(0)
    return buffer

for symbol in symbols:
    df = get_data(symbol, period)
    regime_df = assign_regimes(df, vol_window, median_window)

    st.subheader(f"{symbol} Volatility Regime Detection")

    # Auto-tune button
    if st.button(f"Auto-Tune Windows for {symbol}", key=f"auto_tune_{symbol}"):
        best_vol, best_median = find_balanced_windows(df)
        if best_vol and best_median:
            st.success(f"Found balanced split: Volatility={best_vol}, Median={best_median}")
            vol_window, median_window = best_vol, best_median
            regime_df = assign_regimes(df, vol_window, median_window)
        else:
            st.warning("Could not auto-tune: Try a longer period or add more data.")

    # Regime distribution
    value_counts = regime_df["Regime"].value_counts().rename("Count").to_frame()
    st.markdown("##### Regime Distribution")
    st.dataframe(value_counts)

    # Export buttons with unique keys
    exp1, exp2 = st.columns(2)
    with exp1:
        st.download_button(
            label=f"Export Regimes CSV ({symbol})",
            data=export_csv(regime_df[["Close", "Return", "Volatility", "Regime"]]),
            file_name=f"{symbol}_regimes.csv",
            mime='text/csv',
            key=f"{symbol}_regime_btn"
        )

    # Prepare features and labels
    X = regime_df[["Volatility", "Return", "Volume"]]
    y = regime_df["Regime_Label"]

    # Train model only if both regimes present
    if set(y.unique()) == {0, 1}:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, shuffle=False
        )
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        proba = model.predict_proba(X_test)
        idx_high = list(model.classes_).index(1)
        y_proba = proba[:, idx_high]

        # Classification Report & Confusion Matrix
        m1, m2 = st.columns(2)
        report = classification_report(
            y_test, y_pred,
            target_names=["Low", "High"],
            output_dict=True
        )
        df_report = pd.DataFrame(report).transpose()
        with m1:
            st.write("**Classification Report**")
            st.dataframe(
                df_report.style.format({
                    "precision": "{:.2f}",
                    "recall": "{:.2f}",
                    "f1-score": "{:.2f}"
                })
            )

        cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
        with m2:
            st.write("**Confusion Matrix**")
            fig_cm, ax_cm = plt.subplots()
            sns.heatmap(
                cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Low", "High"], yticklabels=["Low", "High"], ax=ax_cm
            )
            ax_cm.set_xlabel("Predicted")
            ax_cm.set_ylabel("Actual")
            st.pyplot(fig_cm)

        # ROC Curve
        st.write("**ROC Curve**")
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        roc_auc = auc(fpr, tpr)
        fig_roc, ax_roc = plt.subplots()
        ax_roc.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
        ax_roc.plot([0, 1], [0, 1], "--", color="gray")
        ax_roc.set_xlabel("False Positive Rate")
        ax_roc.set_ylabel("True Positive Rate")
        ax_roc.set_title("ROC Curve")
        ax_roc.legend(loc="lower right")
        st.pyplot(fig_roc)

        # Feature importances & export
        st.write("**Feature Importances**")
        importances = pd.Series(model.feature_importances_, index=["Volatility", "Return", "Volume"]).sort_values()
        fig_imp, ax_imp = plt.subplots()
        importances.plot.barh(ax=ax_imp)
        ax_imp.set_title("Feature Importances")
        st.pyplot(fig_imp)

        with exp2:
            st.download_button(
                label=f"Export Feature Importances ({symbol})",
                data=export_csv(importances.to_frame("Importance")),
                file_name=f"{symbol}_feat_imp.csv",
                mime='text/csv',
                key=f"{symbol}_featimp_btn"
            )
    else:
        st.warning(
            "Only one regime present in the selected data period/window. "
            "Adjust your period or window sliders, or try auto-tune, "
            "until you see both 'High' and 'Low' regimes."
        )

    # Price with regimes chart
    st.write(f"**{symbol} Price with Volatility Regimes**")
    fig_price, ax_price = plt.subplots(figsize=(12, 5))
    ax_price.plot(regime_df.index, regime_df["Close"], color="gray", label=f"{symbol} Price")
    highs = regime_df[regime_df["Regime"] == "High"]
    lows = regime_df[regime_df["Regime"] == "Low"]
    ax_price.scatter(highs.index, highs["Close"], color="red", s=20, label="High Volatility")
    ax_price.scatter(lows.index, lows["Close"], color="green", s=20, label="Low Volatility")
    ax_price.set_xlabel("Date")
    ax_price.set_ylabel("Close Price (USD)")
    ax_price.set_title(f"{symbol} Volatility Regimes")
    ax_price.legend(loc="upper left")
    st.pyplot(fig_price)

# About section with human-like explanation
with st.expander("❓ About / What is Volatility Regime Detection?"):
    st.markdown("""
    **Volatility regime detection** is a quantitative technique used by professional investors to identify periods of 'high' and 'low' risk in financial markets. By separating market behavior into regimes, quants can:
    - Adjust trading positions and risk exposure
    - Dynamically rebalance crypto/ETF portfolios
    - Avoid drawdowns during turbulent conditions
    - Systematically manage strategy performance

    **How does it work?**
    1. Calculate rolling price volatility (standard deviation of returns).
    2. Set a threshold (median of volatility) to split data into 'High' and 'Low' regimes.
    3. Use a machine learning model (Random Forest) to classify new periods and evaluate accuracy with confusion matrix, ROC, and feature importance.
    
    **Tips:**
    - Select longer data periods to get richer regime dynamics.
    - Use the 'Auto-Tune' button for instant window calibration.
    - Download CSV results for your own backtests and quant research.

    _This dashboard is intended for research and learning—always test with your own parameters before using in production or with real funds._
    """)
