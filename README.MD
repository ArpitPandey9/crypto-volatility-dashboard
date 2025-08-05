# Crypto Volatility Regime Dashboard

A Streamlit app for detecting and visualizing **High** vs **Low** volatility regimes in cryptocurrency price data using machine learning.

# Crypto Volatility Dashboard

A Streamlit app for crypto volatility regime detection...

---

## 🔥 Project Overview

This dashboard uses historical price data for popular cryptocurrencies (e.g., BTC, ETH) to:

- Calculate rolling volatility metrics
- Identify volatility regimes (High/Low) based on dynamic thresholds
- Train a Random Forest classifier to predict regimes
- Display interactive charts, classification reports, ROC curves, and feature importances
- Provide CSV downloads of regimes and model feature importance for offline analysis
- Support multi-symbol analysis with auto-tuning of volatility window parameters

---

## 🚀 Features

- **Multi-symbol support:** Analyze BTC and ETH, easily extendable
- **Auto-tune:** Automatically find regime window settings that split data meaningfully
- **Interactive controls:** Customize data period, volatility window, and threshold window
- **Downloadable CSV:** Export regime data and feature importance
- **Robust ML pipeline:** Train and evaluate Random Forest classifiers with rich metrics
- **Beautiful visualizations:** Interactive matplotlib and seaborn charts embedded in Streamlit

---

## 📥 How to Run Locally

### Prerequisites

- Python 3.8 or newer
- [Streamlit](https://streamlit.io)
- Required Python packages listed below

### Installation

1. Clone this repository or download the files manually.

2. (Optional but recommended) Create and activate a virtual environment:
    ```bash
    python -m venv venv
    source venv/bin/activate      # macOS/Linux
    venv\Scripts\activate         # Windows
    ```

3. Install required packages:
    ```bash
    pip install -r requirements.txt
    ```
    If `requirements.txt` is not provided, install manually:
    ```bash
    pip install streamlit yfinance scikit-learn matplotlib seaborn pandas numpy
    ```

### Run the app

```bash
streamlit run crypto_volatility_dashboard.py


