# TSA2026 Capstone: Deep Time Series Analytics & Quant Portfolio Optimization

This repository contains the quantitative research project, time-series forecasting engine, and interactive dashboard built for the IIT Guwahati Consulting & Analytics Club. It uses an ensemble of ARIMA, Prophet, XGBoost, and LSTM to forecast stock momentum, applying a Volatility-Aware SciPy SLSQP optimizer to allocate capital.

## Project Structure
- `TSA2026_Capstone_IITGuw.ipynb`: The core Jupyter Notebook containing the data pipeline, feature engineering (Weak Signal Engine), forecasting architectures, and the SciPy portfolio optimization.
- `app.py`: The interactive Streamlit dashboard for live portfolio tracking, visualization, and execution analytics.
- `requirements.txt`: Python dependencies.
- `/Images` & `/figures`: Generated visual assets and dashboard UI screenshots.

---

## Getting Started

### 1. Install Dependencies
It is recommended to use a virtual environment. Install all required packages via pip:
```bash
pip install -r requirements.txt
```

### 2. Run the Core Forecasting Engine
To fetch live data, compute the technical indicators, train the ensemble models, and generate the mathematically optimized portfolio weights, open and execute the Jupyter notebook:
```bash
jupyter notebook TSA2026_Capstone_IITGuw.ipynb
```
*Note: Run the notebook sequentially from top to bottom. It requires network access to pull data from the `yfinance` API.*

### 3. Launch the Intelligence Dashboard
To visualize the real-time execution metrics, model diagnostics, and portfolio allocations, launch the Streamlit application:
```bash
streamlit run app.py
```
This will start a local server and open the institutional-grade dashboard in your default web browser (typically at `http://localhost:8501`).
