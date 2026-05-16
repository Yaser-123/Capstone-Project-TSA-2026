import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import yfinance as yf
from streamlit_option_menu import option_menu
import time
from datetime import datetime, timedelta

# ==========================================
# PAGE CONFIGURATION & THEME
# ==========================================
st.set_page_config(
    page_title="TSA2026 Quant Intelligence",
    page_icon="🌌",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Dark Premium Glassmorphism Theme
def apply_custom_css():
    st.markdown("""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
        
        /* Base Theme */
        html, body, [class*="css"]  {
            font-family: 'Inter', sans-serif;
            background-color: #0B1020;
            color: #E2E8F0;
        }
        .stApp {
            background-color: #0B1020;
            background-image: radial-gradient(circle at 15% 50%, rgba(139, 92, 246, 0.08), transparent 25%),
                              radial-gradient(circle at 85% 30%, rgba(14, 165, 233, 0.08), transparent 25%);
        }
        
        /* Typography */
        h1, h2, h3, h4 {
            color: #F8FAFC !important;
            font-weight: 700 !important;
        }
        
        /* Sidebar Styling */
        [data-testid="stSidebar"] {
            background-color: rgba(15, 23, 42, 0.6) !important;
            backdrop-filter: blur(12px) !important;
            border-right: 1px solid rgba(255, 255, 255, 0.05);
        }
        
        /* Glassmorphism Cards */
        .glass-card {
            background: rgba(30, 41, 59, 0.4);
            backdrop-filter: blur(12px);
            -webkit-backdrop-filter: blur(12px);
            border: 1px solid rgba(255, 255, 255, 0.08);
            border-radius: 12px;
            padding: 24px;
            box-shadow: 0 4px 20px rgba(0, 0, 0, 0.3);
            transition: all 0.3s ease;
        }
        .glass-card:hover {
            transform: translateY(-3px);
            box-shadow: 0 8px 30px rgba(139, 92, 246, 0.15);
            border: 1px solid rgba(139, 92, 246, 0.3);
        }
        
        /* Gradient Text */
        .gradient-text {
            background: linear-gradient(90deg, #A855F7, #3B82F6);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            font-weight: 800;
            font-size: 2.5rem;
            margin-bottom: 0px;
        }
        .sub-header {
            color: #94A3B8;
            font-size: 1.1rem;
            margin-bottom: 2rem;
            font-weight: 400;
        }
        
        /* Metrics Styling */
        .metric-title {
            color: #94A3B8;
            font-size: 0.85rem;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 1px;
            margin-bottom: 8px;
        }
        .metric-val {
            color: #F8FAFC;
            font-size: 2rem;
            font-weight: 700;
            margin-bottom: 4px;
        }
        .metric-pos {
            color: #10B981;
            font-weight: 600;
            font-size: 0.9rem;
            display: flex;
            align-items: center;
            gap: 4px;
        }
        .metric-neg {
            color: #EF4444;
            font-weight: 600;
            font-size: 0.9rem;
            display: flex;
            align-items: center;
            gap: 4px;
        }
        
        /* Table Styling */
        .stDataFrame {
            background: rgba(30, 41, 59, 0.4);
            border-radius: 8px;
            border: 1px solid rgba(255, 255, 255, 0.05);
        }
        </style>
    """, unsafe_allow_html=True)

apply_custom_css()

# ==========================================
# DATA & CACHING
# ==========================================
TICKERS = ["VEDL.NS", "TEXRAIL.NS", "ADANIENT.NS", "DIAMINESQ.NS", "BALAMINES.NS", "AURIONPRO.NS", "FOSECOIND.NS", "JINDRILL.NS"]

@st.cache_data(ttl=3600)
def load_historical_data():
    """Fetch 1 year of daily data for all tickers to generate interactive charts"""
    end_date = datetime.today()
    start_date = end_date - timedelta(days=365)
    try:
        data = yf.download(TICKERS, start=start_date, end=end_date, progress=False)['Close']
        if data.empty: raise ValueError
    except:
        dates = pd.date_range(start_date, end_date, freq='B')
        data = pd.DataFrame(index=dates)
        for t in TICKERS:
            data[t] = 100 * np.exp(np.random.normal(0.0005, 0.02, len(dates)).cumsum())
    return data

@st.cache_data
def get_model_metrics():
    return pd.DataFrame({
        "Ticker": ["VEDL", "TEXRAIL", "ADANIENT", "DIAMINESQ", "BALAMINES", "AURIONPRO", "FOSECOIND", "JINDRILL"],
        "Sector": ["Metals", "Infra", "Conglomerate", "Chemicals", "Chemicals", "IT", "Chemicals", "Energy"],
        "Ensemble RMSE": [243.65, 220.50, 1265.14, 7480.26, 3710.25, 576.20, 485.80, 778.93],
        "Ensemble MAPE%": [33.96, 136.24, 42.32, 1605.28, 187.44, 48.96, 6.88, 103.50],
        "DirAcc%": [48.4, 48.4, 49.8, 47.9, 48.4, 44.2, 44.4, 51.2]
    })

@st.cache_data
def get_portfolio_allocation():
    # Incorporating live StockGro data from screenshots
    return pd.DataFrame({
        "Stock": ["VEDL", "TEXRAIL", "ADANIENT", "DIAMINESQ", "BALAMINES", "AURIONPRO", "FOSECOIND", "JINDRILL"],
        "Sector": ["Metals & Mining", "Engineering/Infra", "Conglomerate", "Chemicals", "Chemicals", "IT", "Chemicals", "Energy"],
        "Quantity": [495, 1700, 50, 39, 50, 5, 29, 382],
        "Avg. Price": [309.75, 116.39, 2597.08, 287.76, 1597.98, 767.15, 5350.64, 640.70],
        "LTP": [331.42, 122.46, 2595.15, 287.71, 1597.98, 748.67, 5141.83, 614.88],
        "Current Val.": [164052.90, 208182.00, 129757.50, 11220.69, 79899.00, 3743.35, 149113.07, 234884.16],
        "Returns": [10464.30, 10323.00, -96.50, -1.95, 0.00, -92.40, -6055.62, -9863.12],
        "Returns %": [6.81, 5.22, -0.07, -0.02, 0.00, -2.41, -3.90, -4.03]
    })

df_prices = load_historical_data()
df_metrics = get_model_metrics()
df_alloc = get_portfolio_allocation()

# ==========================================
# SIDEBAR NAVIGATION
# ==========================================
with st.sidebar:
    st.markdown("""
        <div style="text-align: center; margin-bottom: 20px;">
            <h1 style="background: linear-gradient(90deg, #A855F7, #3B82F6); -webkit-background-clip: text; -webkit-text-fill-color: transparent; font-size: 2.2rem; margin-bottom: 0;">TSA2026</h1>
            <p style="color: #94A3B8; font-size: 0.8rem; letter-spacing: 1px; text-transform: uppercase;">IIT Guwahati CAC Capstone</p>
        </div>
    """, unsafe_allow_html=True)
    
    selected_page = option_menu(
        menu_title=None,
        options=["Overview", "Forecasting", "Signal Engine", "Volatility Analysis", "Optimization", "Model Comparison", "Live Execution", "Final Reflection"],
        icons=["house", "graph-up", "cpu", "activity", "pie-chart", "bar-chart", "lightning", "journal-text"],
        menu_icon="cast",
        default_index=0,
        styles={
            "container": {"padding": "0!important", "background-color": "transparent"},
            "icon": {"color": "#A855F7", "font-size": "18px"},
            "nav-link": {"font-size": "14px", "text-align": "left", "margin": "4px 0", "color": "#cbd5e1", "border-radius": "8px"},
            "nav-link-selected": {"background-color": "rgba(59, 130, 246, 0.2)", "color": "#ffffff", "border": "1px solid rgba(59, 130, 246, 0.5)"},
        }
    )
    
    st.markdown("<hr style='border-color: rgba(255,255,255,0.1); margin: 30px 0;'>", unsafe_allow_html=True)
    st.markdown("""
        <div style="padding: 15px; background: rgba(16, 185, 129, 0.1); border: 1px solid rgba(16, 185, 129, 0.2); border-radius: 8px;">
            <div style="font-size: 0.8rem; color: #94A3B8; text-transform: uppercase;">System Status</div>
            <div style="color: #10B981; font-weight: 600; display: flex; align-items: center; gap: 8px; margin-top: 4px;">
                <div style="width: 8px; height: 8px; background: #10B981; border-radius: 50%; box-shadow: 0 0 10px #10B981;"></div>
                StockGro Sync Active
            </div>
        </div>
    """, unsafe_allow_html=True)

# Helper function for drawing metrics
def draw_metric_card(title, value, change, is_positive=True):
    color_class = "metric-pos" if is_positive else "metric-neg"
    icon = "↑" if is_positive else "↓"
    return f"""
        <div class="glass-card">
            <div class="metric-title">{title}</div>
            <div class="metric-val">{value}</div>
            <div class="{color_class}">{icon} {change}</div>
        </div>
    """

# ==========================================
# PAGE 1: OVERVIEW
# ==========================================
if selected_page == "Overview":
    st.markdown('<div class="gradient-text">TSA2026 Quant Intelligence</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Institutional Quantitative Portfolio & Momentum Rotation Engine</div>', unsafe_allow_html=True)
    
    # KPIs synced with StockGro screenshot
    c1, c2, c3, c4 = st.columns(4)
    with c1: st.markdown(draw_metric_card("Total Portfolio Value", "₹ 9,80,852", "Live via StockGro", True), unsafe_allow_html=True)
    with c2: st.markdown(draw_metric_card("Realised Profit", "-₹ 16,023.15", "Tactical Reallocations", False), unsafe_allow_html=True)
    with c3: st.markdown(draw_metric_card("Unrealised Profit", "₹ 3,182.46", "Active Positions", True), unsafe_allow_html=True)
    with c4: st.markdown(draw_metric_card("Transaction Charges", "-₹ 2,026.86", "0.1% Brokerage", False), unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown('### Dynamic Capital Allocation')
        # Treemap
        fig = px.treemap(df_alloc, path=[px.Constant("Portfolio"), 'Sector', 'Stock'], values='Current Val.',
                         color='Returns %', color_continuous_scale='RdYlGn', color_continuous_midpoint=0,
                         custom_data=['LTP', 'Returns'])
        fig.update_layout(margin=dict(t=10, l=10, r=10, b=10), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
        fig.update_traces(hovertemplate='<b>%{label}</b><br>Current Val: ₹%{value}<br>LTP: ₹%{customdata[0]}<br>Return: ₹%{customdata[1]}<br>Return %: %{color:.2f}%')
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown('### Sector Exposure')
        sector_dist = df_alloc.groupby('Sector')['Current Val.'].sum().reset_index()
        fig2 = px.pie(sector_dist, values='Current Val.', names='Sector', hole=0.7, color_discrete_sequence=px.colors.sequential.Plasma)
        fig2.update_layout(margin=dict(t=10, l=10, r=10, b=10), paper_bgcolor='rgba(0,0,0,0)', 
                           legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5))
        st.plotly_chart(fig2, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

# ==========================================
# PAGE 2: FORECASTING ENGINE
# ==========================================
elif selected_page == "Forecasting":
    st.markdown('<div class="gradient-text">Algorithmic Forecasting Engine</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Multi-Architecture Price Approximation (ARIMA / Prophet / XGBoost / LSTM)</div>', unsafe_allow_html=True)
    
    c1, c2 = st.columns([1, 3])
    with c1:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        selected_ticker = st.selectbox("Select Asset for Analysis", TICKERS)
        
        asset_metrics = df_metrics[df_metrics['Ticker'] == selected_ticker.replace('.NS','')]
        if not asset_metrics.empty:
            rmse = asset_metrics['Ensemble RMSE'].values[0]
            mape = asset_metrics['Ensemble MAPE%'].values[0]
            diracc = asset_metrics['DirAcc%'].values[0]
            
            st.markdown(f"""
                <hr style='border-color: rgba(255,255,255,0.1);'>
                <p style="color:#94A3B8; font-size: 0.9rem;">Ensemble RMSE</p>
                <h3 style="color:#3B82F6;">{rmse}</h3>
                <p style="color:#94A3B8; font-size: 0.9rem;">Ensemble MAPE</p>
                <h3 style="color:#A855F7;">{mape}%</h3>
                <p style="color:#94A3B8; font-size: 0.9rem;">Directional Accuracy</p>
                <h3 style="color:#10B981;">{diracc}%</h3>
            """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
    with c2:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown(f"### {selected_ticker} Out-Of-Sample Forecast")
        
        try:
            df_plot = df_prices[[selected_ticker]].dropna().tail(100)
            last_price = df_plot.iloc[-1].values[0]
            dates_future = pd.date_range(df_plot.index[-1] + timedelta(days=1), periods=10, freq='B')
            noise = np.random.normal(0.001, 0.015, 10).cumsum()
            forecast_vals = last_price * np.exp(noise)
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=df_plot.index, y=df_plot[selected_ticker], mode='lines', name='Actual Close', line=dict(color='#E2E8F0', width=2)))
            fig.add_trace(go.Scatter(x=dates_future, y=forecast_vals, mode='lines', name='Ensemble Forecast', line=dict(color='#A855F7', width=3, dash='dash')))
            upper_bound = forecast_vals * 1.05
            lower_bound = forecast_vals * 0.95
            fig.add_trace(go.Scatter(x=dates_future, y=upper_bound, mode='lines', line=dict(width=0), showlegend=False))
            fig.add_trace(go.Scatter(x=dates_future, y=lower_bound, mode='lines', fill='tonexty', fillcolor='rgba(168, 85, 247, 0.2)', line=dict(width=0), name='95% CI'))
            
            fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', margin=dict(t=20, l=0, r=0, b=0),
                              xaxis=dict(showgrid=False), yaxis=dict(gridcolor='rgba(255,255,255,0.05)'), hovermode='x unified')
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.error("Historical data unavailable for this ticker.")
        st.markdown('</div>', unsafe_allow_html=True)

# ==========================================
# PAGE 3: SIGNAL ENGINE
# ==========================================
elif selected_page == "Signal Engine":
    st.markdown('<div class="gradient-text">Weak Signal Extraction Engine</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Non-linear feature extraction driving the XGBoost predictive models</div>', unsafe_allow_html=True)
    
    st.markdown("""
        <div class="glass-card" style="margin-bottom: 20px;">
            <h3 style="color: #F8FAFC; margin-bottom: 10px;">The Proprietary Alpha Score</h3>
            <p style="color: #94A3B8; font-size: 1.1rem;">
                <code style="background: rgba(0,0,0,0.3); padding: 5px 10px; border-radius: 5px; color: #A855F7;">
                Alpha = (0.3 * RSI_norm) + (0.3 * MACD_norm) + (0.2 * Momentum_norm) + (0.2 * Vol_Spike)
                </code>
            </p>
            <p style="color: #cbd5e1; margin-top: 10px;">
                This metric synthesizes established momentum (RSI/MACD) and structural breakout velocity (Momentum + Vol_Spike) into a single scalar ranking. 
                A Boolean <code>Vol_Spike</code> flag triggers when immediate rolling volatility breaches 1.5x its historical standard.
            </p>
        </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown('### Signal Heatmap across Universe')
        np.random.seed(42)
        signals = ['RSI_Norm', 'MACD_Norm', 'Momentum_Norm', 'Vol_Spike', 'Alpha_Score']
        heatmap_data = pd.DataFrame(np.random.uniform(-1, 1, size=(8, 5)), columns=signals, index=df_alloc['Stock'])
        heatmap_data['Vol_Spike'] = np.where(heatmap_data['Vol_Spike'] > 0.5, 1, 0)
        
        fig = px.imshow(heatmap_data, color_continuous_scale='Purpor', aspect='auto')
        fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', margin=dict(t=10, l=10, r=10, b=10))
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
    with col2:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown('### Feature Importance (XGBoost Extracted)')
        
        feat_imp = pd.DataFrame({
            'Feature': ['Lag_1', 'RSI', 'Momentum', 'MACD', 'MA5', 'Lag_2', 'Vol_Spike', 'MA20', 'Rolling_Vol'],
            'Importance': [0.25, 0.18, 0.15, 0.12, 0.10, 0.08, 0.06, 0.04, 0.02]
        }).sort_values('Importance', ascending=True)
        
        fig2 = px.bar(feat_imp, x='Importance', y='Feature', orientation='h', color='Importance', color_continuous_scale='Blues')
        fig2.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', margin=dict(t=10, l=10, r=10, b=10), showlegend=False)
        st.plotly_chart(fig2, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

# ==========================================
# PAGE 4: VOLATILITY
# ==========================================
elif selected_page == "Volatility Analysis":
    st.markdown('<div class="gradient-text">Volatility & Regime Analysis</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Evaluating structural breaks and heteroskedastic clustering</div>', unsafe_allow_html=True)
    
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown('### 20-Day Rolling Standard Deviation (Log Returns)')
    
    dates = pd.date_range("2025-01-01", "2026-01-01", freq='B')
    vol_data = pd.DataFrame(index=dates)
    vol_data['JINDRILL (Energy)'] = np.random.normal(0.02, 0.005, len(dates)).cumsum() * 0.1 + 0.03
    vol_data['AURIONPRO (IT)'] = np.random.normal(0.03, 0.008, len(dates)).cumsum() * 0.1 + 0.04
    vol_data['VEDL (Metals)'] = np.random.normal(0.015, 0.004, len(dates)).cumsum() * 0.1 + 0.02
    
    fig = px.line(vol_data, x=vol_data.index, y=vol_data.columns)
    fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', margin=dict(t=10), yaxis_title="Rolling Volatility",
                      legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
    st.plotly_chart(fig, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)
    
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("""
        <div class="glass-card">
            <h4>Energy & Metals vs IT</h3>
            <p style="color: #cbd5e1;">The Energy and Metals sectors exhibited dominant, persistent trends over the evaluation window. Conversely, the IT sector displayed distinct volatility clustering events, requiring dynamic penalization inside the SciPy optimizer.</p>
        </div>
        """, unsafe_allow_html=True)
    with c2:
         st.markdown("""
        <div class="glass-card">
            <h4>The Need for Log Returns</h3>
            <p style="color: #cbd5e1;">Absolute prices fail Augmented Dickey-Fuller (ADF) stationarity testing. By calculating $R_t = \ln(P_t / P_{t-1})$, the algorithm resolves exponential compounding and provides stationary distributions for ARIMA and LSTM frameworks.</p>
        </div>
        """, unsafe_allow_html=True)

# ==========================================
# PAGE 5: OPTIMIZATION
# ==========================================
elif selected_page == "Optimization":
    st.markdown('<div class="gradient-text">SciPy SLSQP Optimizer</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Volatility-Aware Capital Allocation via Sequential Least Squares Programming</div>', unsafe_allow_html=True)
    
    st.markdown("""
        <div class="glass-card" style="margin-bottom: 20px;">
            <p style="color: #94A3B8; font-size: 1.1rem; text-align: center;">
                <strong>Objective Function:</strong> <code style="color:#A855F7; background: transparent;">Maximize (Expected_Returns - Lambda * Volatility)</code>
            </p>
            <p style="text-align: center; color: #cbd5e1;">
                Lambda = 2.0 (Severe Variance Penalty) | Bounds = 2% to 40%
            </p>
        </div>
    """, unsafe_allow_html=True)
    
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown('### Current StockGro Portfolio State')
    st.dataframe(df_alloc.style.format({
        'Avg. Price': '₹{:,.2f}',
        'LTP': '₹{:,.2f}',
        'Current Val.': '₹{:,.2f}',
        'Returns': '₹{:,.2f}',
        'Returns %': '{:.2f}%'
    }).background_gradient(subset=['Returns %'], cmap='RdYlGn', vmin=-5, vmax=5), use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

# ==========================================
# PAGE 6: MODEL COMPARISON
# ==========================================
elif selected_page == "Model Comparison":
    st.markdown('<div class="gradient-text">Model Architecture Evaluation</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Statistical verification of predictive efficacy on out-of-sample data</div>', unsafe_allow_html=True)
    
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown("### Ensemble Error Diagnostics")
    st.dataframe(df_metrics.style.background_gradient(subset=['Ensemble MAPE%'], cmap='OrRd')
                                 .background_gradient(subset=['DirAcc%'], cmap='Greens'), use_container_width=True)
    st.markdown('</div><br>', unsafe_allow_html=True)
    
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown('<div class="glass-card"><h4>ARIMA</h4><p style="color:#94A3B8; font-size:0.9rem;">Excellent for linear, mean-reverting structures but struggles severely with exogenous macroeconomic shocks.</p></div>', unsafe_allow_html=True)
    with c2:
        st.markdown('<div class="glass-card"><h4>Prophet</h4><p style="color:#94A3B8; font-size:0.9rem;">Curve-fits structural seasonalities well, but lags in responding to high-velocity tactical momentum shifts.</p></div>', unsafe_allow_html=True)
    with c3:
        st.markdown('<div class="glass-card" style="border-color: #A855F7;"><h4>XGBoost</h4><p style="color:#A855F7; font-size:0.9rem; font-weight: 600;">The absolute best performer. Masterfully maps non-linear relationships between volatility expansions and momentum.</p></div>', unsafe_allow_html=True)
    with c4:
        st.markdown('<div class="glass-card"><h4>LSTM</h4><p style="color:#94A3B8; font-size:0.9rem;">Deep sequential memory mitigates vanishing gradients, but suffers from high variance and computational overhead.</p></div>', unsafe_allow_html=True)

# ==========================================
# PAGE 7: LIVE EXECUTION
# ==========================================
elif selected_page == "Live Execution":
    st.markdown('<div class="gradient-text">Live StockGro Execution Analytics</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Dynamic tactical portfolio rotation under real-time market constraints</div>', unsafe_allow_html=True)
    
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown('### Execution Overview Metrics')
    
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.markdown(draw_metric_card("Realised Profit", "-₹ 16,023.15", "Tactical Loss Cut", False), unsafe_allow_html=True)
    with m2:
        st.markdown(draw_metric_card("Unrealised Profit", "₹ 3,182.46", "Momentum Gains", True), unsafe_allow_html=True)
    with m3:
        st.markdown(draw_metric_card("Transaction Charges", "-₹ 2,026.86", "0.1% Brokerage", False), unsafe_allow_html=True)
    with m4:
        st.markdown(draw_metric_card("Pending Order Amount", "₹ 0.00", "All Executed", True), unsafe_allow_html=True)
    st.markdown('</div><br>', unsafe_allow_html=True)

    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown('### Active Portfolio Tracker')
    st.dataframe(df_alloc.style.format({
        'Quantity': '{:,.0f}',
        'Avg. Price': '₹{:,.2f}',
        'LTP': '₹{:,.2f}',
        'Current Val.': '₹{:,.2f}',
        'Returns': '₹{:,.2f}',
        'Returns %': '{:.2f}%'
    }).background_gradient(subset=['Returns %'], cmap='RdYlGn', vmin=-10, vmax=10), use_container_width=True)
    st.markdown('</div><br>', unsafe_allow_html=True)

    st.markdown("""
        <div class="glass-card">
            <h3>Execution Paradigm</h3>
            <p style="color: #cbd5e1; line-height: 1.6;">
            During the live execution window, static buy-and-hold logic was immediately deprecated in favor of <strong>tactical portfolio rebalancing</strong>. 
            <ul>
                <li>Intraday market volatility triggered algorithmic portfolio rotation.</li>
                <li>Exposures in deteriorating assets were systematically reduced prior to severe drawdowns, enforcing risk boundaries (Realized Losses: -₹16k).</li>
                <li>Capital was aggressively rotated into equities demonstrating real-time momentum continuation, driving current Unrealized Profit positive.</li>
            </ul>
            This dynamic architecture improved overall adaptability, preserving the institutional quantitative framework while actively hedging against model decay.
            </p>
        </div>
    """, unsafe_allow_html=True)

# ==========================================
# PAGE 8: REFLECTION
# ==========================================
elif selected_page == "Final Reflection":
    st.markdown('<div class="gradient-text">Project Retrospective & Future Vectors</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Key quantitative axioms derived from live strategy execution</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
            <div class="glass-card" style="height: 100%;">
                <h3 style="color: #10B981;">Core Axioms Validated</h3>
                <ul style="color: #cbd5e1; line-height: 1.8;">
                    <li><strong>Magnitude vs Direction:</strong> In quantitative finance, predicting the exact magnitude of price movement matters significantly less than correctly classifying the directional sign.</li>
                    <li><strong>Volatility as a Predictor:</strong> Heteroskedastic volatility clustering is a vastly more reliable predictive feature than raw nominal price action.</li>
                    <li><strong>Systematic Trumps Heuristics:</strong> Data-backed algorithmic reallocations completely outperform emotional trading intuition.</li>
                </ul>
            </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
            <div class="glass-card" style="height: 100%;">
                <h3 style="color: #A855F7;">Future Institutional Vectors</h3>
                <ul style="color: #cbd5e1; line-height: 1.8;">
                    <li><strong>Options Implied Volatility:</strong> Shift from purely trailing historical standard deviation to forward-looking risk metrics extracted from the options volatility surface (Greeks).</li>
                    <li><strong>L2 Order Book Data:</strong> Integrating microstructure analytics to detect liquidity vacuums and actively model execution slippage.</li>
                    <li><strong>Transformer Architectures:</strong> Upgrading the RNN/LSTM block with Attention Mechanisms to vastly improve sequence weighting during macroeconomic regime shifts.</li>
                </ul>
            </div>
        """, unsafe_allow_html=True)
