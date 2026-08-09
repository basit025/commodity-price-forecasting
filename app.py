import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import timedelta
import os
import yfinance as yf

# Import our backend ensemble engine
from ensemble_inference import ensemble_predict
from live_data_pipeline import ASSET_TICKERS, get_live_features
import base64

@st.cache_data
def get_base64_image(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode()

# --- PAGE CONFIG ---
st.set_page_config(
    page_title="FundForge AI Terminal",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- CUSTOM CSS ---
st.markdown("""
<style>
    /* ===== FundForge Fintech Theme (Merged) ===== */
    .stApp {
        background:
            radial-gradient(circle at 15% 10%, rgba(77,230,255,0.05), transparent 24%),
            radial-gradient(circle at 85% 0%, rgba(79,107,255,0.08), transparent 24%),
            #07101E;
        color: #EAF4FF;
        font-family: Inter, "Segoe UI", sans-serif;
    }

    /* Ticker Tape Animation */
    .ticker-wrap {
        width: 100%;
        overflow: hidden;
        background: linear-gradient(180deg, rgba(19,31,51,0.96), rgba(13,24,40,0.96)) !important;
        padding-top: 4px;
        padding-bottom: 4px;
        box-sizing: content-box;
        border: 1px solid #1E3A5F !important;
        border-radius: 14px;
        margin-bottom: 2rem;
        white-space: nowrap;
        box-shadow: 0 10px 30px rgba(0,0,0,.18);
    }

    .ticker {
        display: inline-block;
        white-space: nowrap;
        padding-right: 100%;
        box-sizing: content-box;
        animation-iteration-count: infinite;
        animation-timing-function: linear;
        animation-name: ticker;
        animation-duration: 30s;
    }

    @keyframes ticker {
        0% { transform: translate3d(0, 0, 0); }
        100% { transform: translate3d(-100%, 0, 0); }
    }

    .ticker-item {
        display: inline-block;
        padding: 10px 2rem;
        font-family: 'Inter', sans-serif;
        font-size: 14px;
        font-weight: 500;
        color: #DCEBFA !important;
    }

    .pos { color: #59D8FF; font-weight: 700; }
    .neg { color: #7C5CFF; font-weight: 700; }
    
    .app-title {
        margin: 0 0 0.4rem 0;
        color: #EAF4FF !important;
        font-size: 2.4rem;
        font-weight: 800;
        letter-spacing: -0.8px;
    }

    .app-title span {
        color: #59D8FF;
    }

    /* General Streamlit text */
    h1, h2, h3, h4, h5, h6, p, label {
        color: #EAF4FF !important;
    }

    /* Custom Interactive Pills (Merged Size) */
    div[data-testid="stPills"] {
        display: flex;
        flex-direction: column;
        align-items: center;
        margin-bottom: 20px;
        width: 100%;
    }
    /* Generic Widget Labels */
    [data-testid="stWidgetLabel"] {
        display: flex !important;
        justify-content: center !important;
        text-align: center !important;
        width: 100% !important;
        margin-bottom: 12px !important;
    }
    [data-testid="stWidgetLabel"] p {
        color: #9FB4C8 !important;
        font-size: 20px !important;
        font-weight: 700 !important;
        text-transform: uppercase !important;
        letter-spacing: 1.5px !important;
        text-align: center !important;
        margin: 0 auto !important;
    }
    /* PILL / SEGMENTED CONTROL SIZING AND THEME (GENERIC ARIA ROLE TARGETING) */
    [role="radiogroup"] {
        justify-content: center !important;
        gap: 12px !important;
        flex-wrap: wrap !important;
        width: 100% !important;
        margin-bottom: 24px !important;
    }
    
    [role="radiogroup"] button, 
    [role="radiogroup"] label, 
    [role="radiogroup"] [role="radio"] {
        border-radius: 40px !important;
        background: #111C2E !important;
        border: 2px solid #29425F !important;
        transition: all 0.3s ease !important;
        padding: 0 !important;
        min-height: 48px !important;
        display: inline-flex !important;
        align-items: center !important;
        justify-content: center !important;
        margin: 4px !important;
        box-shadow: inset 0 1px 0 rgba(255,255,255,.025) !important;
    }
    
    [role="radiogroup"] button *, 
    [role="radiogroup"] label *, 
    [role="radiogroup"] [role="radio"] * {
        font-size: 24px !important;
        font-weight: 800 !important;
        color: #D9E8F7 !important;
        padding: 8px 24px !important;
        margin: 0 !important;
        line-height: 1.2 !important;
    }
    
    [role="radiogroup"] button:hover, 
    [role="radiogroup"] label:hover, 
    [role="radiogroup"] [role="radio"]:hover {
        border-color: #4DE6FF !important;
        background: #14243A !important;
        transform: translateY(-2px) !important;
    }
    
    [role="radiogroup"] button:hover *, 
    [role="radiogroup"] label:hover *, 
    [role="radiogroup"] [role="radio"]:hover * {
        color: #4DE6FF !important;
    }
    
    [role="radiogroup"] [aria-checked="true"], 
    [role="radiogroup"] [data-selected="true"], 
    [role="radiogroup"] [aria-pressed="true"],
    [role="radiogroup"] input:checked + div {
        background: linear-gradient(135deg, #435CFF 0%, #20C7E8 100%) !important;
        box-shadow: 0 8px 24px rgba(49,127,255,.35) !important;
        border-color: transparent !important;
    }
    
    [role="radiogroup"] [aria-checked="true"] *, 
    [role="radiogroup"] [data-selected="true"] *, 
    [role="radiogroup"] [aria-pressed="true"] *,
    [role="radiogroup"] input:checked + div * {
        color: #F8FDFF !important;
        font-weight: 800 !important;
    }

    /* Inputs & Metrics */
    div[data-testid="stMetric"] {
        background: #162338 !important;
        border: 1px solid #29425F !important;
        border-radius: 18px !important;
        padding: 20px !important;
        box-shadow: none !important;
        min-height: 132px;
        overflow: hidden;
        box-sizing: border-box;
        transition: transform 0.3s ease, box-shadow 0.3s ease, border-color 0.3s ease;
    }
    div[data-testid="stMetric"]:hover {
        transform: translateY(-4px);
        box-shadow: 0 12px 24px rgba(0,0,0,0.4) !important;
        border-color: rgba(77,230,255,0.4) !important;
    }

    div[data-testid="stMetricLabel"] {
        color: #9FB4C8 !important;
    }

    div[data-testid="stMetricValue"] {
        color: #F4FAFF !important;
    }

    div[data-baseweb="input"] > div,
    div[data-baseweb="select"] > div {
        background: #111C2E !important;
        border: 1px solid #2A415F !important;
        color: #EAF4FF !important;
        border-radius: 12px !important;
    }

    /* Buttons */
    .stButton > button {
        width: 100%;
        background: linear-gradient(135deg, #435CFF 0%, #20C7E8 100%) !important;
        color: #FFFFFF !important;
        border-radius: 12px !important;
        border: 1px solid rgba(77,230,255,.3) !important;
        font-weight: 800 !important;
        font-size: 24px !important;
        padding: 0.8rem !important;
        box-shadow: 0 10px 26px rgba(32,199,232,.18);
        transition: all 0.3s ease !important;
    }
    .stButton > button * {
        font-size: 24px !important;
    }

    .stButton > button:hover {
        transform: translateY(-1px);
        background: linear-gradient(135deg, #526CFF 0%, #38D4EE 100%) !important;
        box-shadow: 0 14px 30px rgba(32,199,232,.24);
    }

    /* Streamlit dialog */
    div[role="dialog"] {
        background: #091322 !important;
        border: 1px solid #345D87 !important;
        border-radius: 20px !important;
        box-shadow: 0 30px 90px rgba(0,0,0,.62);
        width: min(1120px, 94vw) !important;
        max-width: 1120px !important;
        overflow-x: hidden !important;
    }

    div[role="dialog"] * {
        font-family: Inter, "Segoe UI", sans-serif;
    }

    div[role="dialog"] h1,
    div[role="dialog"] h2,
    div[role="dialog"] h3,
    div[role="dialog"] h4 {
        color: #EAF4FF !important;
    }

    div[role="dialog"] p,
    div[role="dialog"] span,
    div[role="dialog"] label {
        color: #C7D8EA;
    }

    /* Modal Layout Elements */
    .plan-hero {
        background: linear-gradient(135deg, #15304A 0%, #162338 55%, #1A2446 100%);
        border: 1px solid #2E5D86;
        border-radius: 18px;
        padding: 22px 24px;
        margin-bottom: 16px;
        box-shadow: none;
    }
    .plan-eyebrow {
        color: #59D8FF;
        font-size: 12px;
        font-weight: 800;
        letter-spacing: .8px;
        text-transform: uppercase;
    }
    .plan-title {
        color: #F3FAFF;
        font-size: 24px;
        font-weight: 850;
        margin-top: 6px;
    }
    .plan-subtitle {
        color: #9FB4C8;
        font-size: 14px;
        margin-top: 5px;
    }
    .rec-block {
        background: #162338;
        border: 1px solid #29425F;
        border-radius: 18px;
        padding: 22px;
        margin: 14px 0 16px 0;
        box-shadow: none;
    }
    .rec-label {
        color: #9FB4C8;
        font-size: 12px;
        font-weight: 800;
        letter-spacing: .7px;
        text-transform: uppercase;
    }
    .rec-value {
        font-size: 40px;
        line-height: 1;
        font-weight: 900;
        margin-top: 8px;
        letter-spacing: -0.8px;
    }
    .gain-banner {
        background: #162338;
        border: 1px solid #29425F;
        border-radius: 18px;
        padding: 20px;
        margin: 14px 0 16px 0;
        box-shadow: none;
    }
    .gain-banner .label {
        color: #9FB4C8;
        font-size: 12px;
        font-weight: 800;
        letter-spacing: .7px;
        text-transform: uppercase;
    }
    .gain-banner .value {
        font-size: 34px;
        font-weight: 900;
        margin-top: 5px;
    }
    .section-heading {
        color: #59D8FF;
        font-weight: 800;
        font-size: 16px;
        margin: 12px 0 8px 0;
    }
</style>
""", unsafe_allow_html=True)

# --- CONFIG & STATE ---
COMMODITIES = {
    'Gold': 'gold',
    'Silver': 'silver',
    'Copper': 'copper',
    'Crude Oil': 'crude_oil',
    'Natural Gas': 'natural_gas',
    'Wheat': 'wheat'
}

HORIZONS = {
    '1D': 1,
    '7D': 7,
    '14D': 14,
    '1M': 28,
    '6W': 42,
    '2M': 60,
    '3M': 90,
    '4M': 120
}

# --- HELPER FUNCTIONS ---
@st.cache_data(ttl=3600)
def get_ticker_data():
    """Fetches live 5-day prices from yfinance for the ticker tape."""
    ticker_html = ""
    for name, key in COMMODITIES.items():
        try:
            yf_ticker = ASSET_TICKERS.get(key)
            if not yf_ticker: continue
            df = yf.download(yf_ticker, period='5d', progress=False)
            if len(df) >= 2:
                last = df['Close'].iloc[-1].item()
                prev = df['Close'].iloc[-2].item()
                pct = ((last - prev) / prev) * 100
                
                color_class = 'pos' if pct >= 0 else 'neg'
                arrow = '▲' if pct >= 0 else '▼'
                sign = '+' if pct >= 0 else ''
                
                ticker_html += f"<div class='ticker-item'>{name} ${last:.2f} <span class='{color_class}'>{arrow} {sign}{pct:.2f}%</span></div>"
        except Exception:
            pass
            
    return f"""
    <div class="ticker-wrap">
        <div class="ticker">
            {ticker_html}{ticker_html}{ticker_html}
        </div>
    </div>
    """

@st.cache_data(ttl=3600)
def get_live_chart_data(commodity_key):
    """Loads the last 90 days of live historical data for plotting."""
    try:
        yf_ticker = ASSET_TICKERS.get(commodity_key)
        df = yf.download(yf_ticker, period='90d', progress=False)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.droplevel(1)
        df = df.reset_index()
        return df
    except Exception:
        # Fallback to local CSV if yfinance fails
        file_path = os.path.join('./data', f'{commodity_key}.csv')
        df = pd.read_csv(file_path, parse_dates=['Date'])
        return df.sort_values('Date').tail(90)

@st.cache_data(ttl=3600)
def get_cached_trajectory(commodity_key, max_horizon):
    """Runs the ensemble predictions across multiple horizons efficiently."""
    all_horizons = [1, 7, 14, 28, 42, 60, 90, 120]
    target_horizons = [h for h in all_horizons if h <= max_horizon]
    results = {}
    for h in target_horizons:
        results[h] = ensemble_predict(commodity_key, h, top_n=3)
    return target_horizons, results

def generate_market_drivers(historical_features_df):
    """Generates dynamic market driver text based on the latest feature row."""
    drivers = []
    latest = historical_features_df.iloc[-1]
    
    if 'RSI_14' in historical_features_df.columns:
        if latest['RSI_14'] > 65:
            drivers.append(("RSI approaching overbought", "-"))
        elif latest['RSI_14'] < 35:
            drivers.append(("RSI heavily oversold", "+"))
            
    if 'Close_to_MA_50' in historical_features_df.columns:
        if latest['Close_to_MA_50'] > 1.0:
            drivers.append(("Price above 50-day moving average", "+"))
        else:
            drivers.append(("Price below 50-day moving average", "-"))
            
    if 'USD_Index' in historical_features_df.columns:
        dxy_5_days_ago = historical_features_df.iloc[-6]['USD_Index'] if len(historical_features_df) > 5 else latest['USD_Index']
        if latest['USD_Index'] < dxy_5_days_ago:
            drivers.append(("USD Index weakening", "+"))
        elif latest['USD_Index'] > dxy_5_days_ago:
            drivers.append(("USD Index strengthening", "-"))
            
    if 'US_10Y_Yield' in historical_features_df.columns:
        us10y_5_days_ago = historical_features_df.iloc[-6]['US_10Y_Yield'] if len(historical_features_df) > 5 else latest['US_10Y_Yield']
        if latest['US_10Y_Yield'] < us10y_5_days_ago:
            drivers.append(("Treasury yields falling", "+"))
        elif latest['US_10Y_Yield'] > us10y_5_days_ago:
            drivers.append(("Treasury yields rising", "-"))
            
    if 'VIX' in historical_features_df.columns:
        if latest['VIX'] > 20:
            drivers.append(("High macro volatility (VIX > 20)", "-"))
         
    return drivers[:3]

# --- MODALS ---
@st.dialog("AI Investment Plan", width="large")
def show_investment_plan(data: dict):
    # Dynamic calculation inside modal
    investment = st.number_input(
        "Investment Amount ($)",
        min_value=100.0, value=1000.0, step=100.0,
        key=f"investment_modal_{data['selected_name']}_{data['selected_horizon']}"
    )
    expected_profit = investment * (data['pred_move_pct'] / 100)
    future_value = investment + expected_profit
    
    direction_word = "rise" if data['pred_move_pct'] > 0 else "fall" if data['pred_move_pct'] < 0 else "stay stable"
    confidence_label = "High" if data['confidence'] >= 75 else "Moderate" if data['confidence'] >= 55 else "Low"
    gain_color = "#59D8FF" if expected_profit >= 0 else "#8A6CFF"
    
    st.markdown(
        f"""
        <div class="plan-hero">
            <div class="plan-eyebrow">AI-Powered Investment Summary</div>
            <div class="plan-title">{data['selected_name']} • {data['selected_horizon_name']} Outlook</div>
            <div class="plan-subtitle">
                The model currently expects the price to {direction_word}.
                Forecast confidence is {confidence_label.lower()} at {data['confidence']:.1f}%.
            </div>
        </div>
        """, unsafe_allow_html=True
    )

    c1, c2, c3 = st.columns(3)
    with c1: st.metric("Current Price", f"${data['current_price']:,.2f}")
    with c2: st.metric("Predicted Price", f"${data['predicted_price']:,.2f}")
    with c3: st.metric("Expected Return", f"{data['pred_move_sign']}{data['pred_move_pct']:.2f}%")

    r1, r2, r3 = st.columns(3)
    with r1: st.metric("Confidence", f"{data['confidence']:.1f}%")
    with r2: st.metric("Risk Level", data['risk_level'])
    with r3: st.metric("Risk Score", f"{data['risk_score']:.0f}/100")

    st.markdown(
        f"""
        <div class="rec-block">
            <div class="rec-label">Recommendation</div>
            <div class="rec-value" style="color:{data['rec_color']};">{data['recommendation']}</div>
            <div style="color:#9FB4C8; font-size:13px; margin-top:7px;">
                Based on the selected horizon, predicted return, confidence and quantified risk.
            </div>
        </div>
        """, unsafe_allow_html=True
    )

    st.markdown('<div class="section-heading">Investment Simulation</div>', unsafe_allow_html=True)

    s1, s2, s3 = st.columns(3)
    with s1: st.metric("Investment Amount", f"${investment:,.2f}")
    with s2: st.metric("Estimated Gain / Loss", f"${expected_profit:,.2f}", delta=f"{data['pred_move_sign']}{data['pred_move_pct']:.2f}%")
    with s3: st.metric("Estimated Future Value", f"${future_value:,.2f}")

    st.markdown(
        f"""
        <div class="gain-banner">
            <div class="label">Estimated {"Gain" if expected_profit >= 0 else "Loss"}</div>
            <div class="value" style="color:{gain_color};">${expected_profit:,.2f}</div>
        </div>
        """, unsafe_allow_html=True
    )

    st.markdown('<div class="section-heading">Risk Snapshot</div>', unsafe_allow_html=True)
    q1, q2, q3 = st.columns(3)
    with q1: st.metric("Annualised Volatility", f"{data['annualized_volatility']:.1f}%")
    with q2: st.metric("Max Drawdown", f"{data['max_drawdown_pct']:.1f}%")
    with q3: st.metric("Downside Volatility", f"{data['downside_volatility']:.1f}%")

    st.markdown('<div class="section-heading">Plan Details</div>', unsafe_allow_html=True)
    d1, d2 = st.columns(2)
    with d1:
        st.write(f"**Commodity:** {data['selected_name']}")
        st.write(f"**Horizon:** {data['selected_horizon_name']} ({data['selected_horizon']} days)")
    with d2:
        st.write(f"**Models:** {', '.join(data['models_used'])}")
        st.write(f"**Risk Level:** {data['risk_level']}")


# --- UI RENDERING ---

logo_b64 = get_base64_image("logo-small.png")
st.markdown(f'''
<div style="display: flex; align-items: center; gap: 15px; margin-bottom: 0.4rem;">
    <img src="data:image/png;base64,{logo_b64}" style="width: 50px; height: 50px; border-radius: 50%; object-fit: cover; border: 2px solid #59D8FF; box-shadow: 0 0 15px rgba(89,216,255,0.4);">
    <h1 class="app-title" style="margin: 0;">FundForge <span>AI Terminal</span></h1>
</div>
''', unsafe_allow_html=True)
st.markdown(get_ticker_data(), unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)
col_spacer, col_btn = st.columns([7, 3])
with col_btn:
    simulator_btn_placeholder = st.empty()
st.markdown("<br>", unsafe_allow_html=True)

selected_name = st.pills(
    "Select Asset",
    options=list(COMMODITIES.keys()),
    default=list(COMMODITIES.keys())[0],
    selection_mode="single",
    label_visibility="visible"
)
if not selected_name:
    selected_name = list(COMMODITIES.keys())[0]
selected_key = COMMODITIES[selected_name]

selected_horizon_name = st.pills(
    "Investment Horizon",
    options=list(HORIZONS.keys()),
    default='2M',
    selection_mode="single",
    label_visibility="visible"
)
if not selected_horizon_name:
    selected_horizon_name = '2M'
selected_horizon = HORIZONS[selected_horizon_name]

st.markdown("<br><br>", unsafe_allow_html=True)

# 3. Main Logic
with st.spinner("AI Models Computing Consensus..."):
    try:
        # Load historical graph data (live)
        historical_df = get_live_chart_data(selected_key)
        last_date = historical_df['Date'].iloc[-1]
        hist_dates = historical_df['Date'].tolist()
        hist_prices = historical_df['Close'].tolist()
        
        # Load features for Market Drivers
        df_live_seq, current_price_live = get_live_features(selected_key)
        
        # Generate predictions path
        target_horizons, trajectory_results = get_cached_trajectory(selected_key, selected_horizon)
        
        path_dates = [last_date]
        path_prices = [hist_prices[-1]]
        
        for h in target_horizons:
            res = trajectory_results[h]
            f_date = last_date + timedelta(days=int(h * 1.4))
            path_dates.append(f_date)
            path_prices.append(res['predicted_price'])
            
        final_result = trajectory_results[selected_horizon]
        
        # Signal Formatting
        is_up = final_result['predicted_return'] > 0
        signal_color = "#4DE6FF" if is_up else "#7C5CFF"
        
        col1, col2 = st.columns([7, 3], gap="large")
        
        with col1:
            st.subheader(f"{selected_name} Trajectory Projection")
            
            fig = go.Figure()
            # Historical Area
            fig.add_trace(go.Scatter(
                x=hist_dates, y=hist_prices,
                fill='tozeroy',
                mode='lines',
                line=dict(color='#4F6BFF', width=2),
                fillcolor='rgba(79, 107, 255, 0.10)',
                name='Historical Price'
            ))
            # Multi-Horizon Projection Line
            fig.add_trace(go.Scatter(
                x=path_dates,
                y=path_prices,
                mode='lines+markers',
                line=dict(color=signal_color, width=3, dash='dash', shape='spline'), 
                marker=dict(size=8, color=signal_color),
                name='AI Multi-Horizon Path'
            ))
            
            fig.update_layout(
                plot_bgcolor='#07101E',
                paper_bgcolor='#07101E',
                font=dict(color='#9FB4C8'),
                xaxis=dict(showgrid=False, title=''),
                yaxis=dict(showgrid=True, gridcolor='#243954', title='Price (USD)'),
                margin=dict(l=0, r=0, t=20, b=0),
                height=500,
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )
            st.plotly_chart(fig, use_container_width=True)
            st.caption(f"**Models inside current ensemble:** {', '.join(final_result['models_used'])}")
        
        with col2:
            st.subheader("Actionable Telemetry")
            
            # --- MARKET DRIVERS ---
            drivers = generate_market_drivers(df_live_seq)
            driver_html = ""
            for driver_text, sentiment in drivers:
                color = "#4DE6FF" if sentiment == "+" else "#7C5CFF"
                driver_html += f'''
                <div class="driver-item">
                    <span style="color: #E2E8F0; font-weight: 500;">{driver_text}</span>
                    <span style="color: {color}; font-weight: 900; font-size: 18px; line-height: 1;">{sentiment}</span>
                </div>
                '''
                
            badge_bg = "rgba(77, 230, 255, 0.10)" if is_up else "rgba(124, 92, 255, 0.12)"
            badge_color = "#4DE6FF" if is_up else "#7C5CFF"
            badge_icon = "↗ Bullish" if is_up else "↘ Bearish"
            
            p_min = final_result.get('predicted_min', final_result['predicted_price'] * 0.98)
            p_max = final_result.get('predicted_max', final_result['predicted_price'] * 1.02)
            c_price = final_result['current_price']
            
            visual_min = min(p_min, c_price)
            visual_max = max(p_max, c_price)
            if visual_max == visual_min: visual_max += 0.01
            
            marker_pos = ((final_result['predicted_price'] - visual_min) / (visual_max - visual_min)) * 100
            marker_pos = max(0, min(100, marker_pos))
            
            confidence = final_result.get('confidence_pct', 72.0)
            pred_move_pct = final_result['predicted_return'] * 100
            pred_move_sign = "+" if pred_move_pct >= 0 else ""

            # --- DYNAMIC MARKET RISK ENGINE ---
            daily_returns = historical_df['Close'].pct_change().dropna()
            annualized_volatility = daily_returns.std() * np.sqrt(252) * 100
            negative_returns = daily_returns[daily_returns < 0]
            downside_volatility = (negative_returns.std() * np.sqrt(252) * 100 if len(negative_returns) > 1 else 0.0)
            rolling_peak = historical_df['Close'].cummax()
            drawdown_series = (historical_df['Close'] / rolling_peak) - 1
            max_drawdown_pct = abs(drawdown_series.min()) * 100

            vol_pts = 35 if annualized_volatility >= 50 else 28 if annualized_volatility >= 35 else 18 if annualized_volatility >= 20 else 9 if annualized_volatility >= 10 else 4
            dd_pts = 25 if max_drawdown_pct >= 20 else 18 if max_drawdown_pct >= 12 else 11 if max_drawdown_pct >= 7 else 6 if max_drawdown_pct >= 3 else 2
            conf_pts = 20 if confidence < 50 else 15 if confidence < 60 else 10 if confidence < 70 else 5 if confidence < 80 else 2
            abs_move = abs(pred_move_pct)
            f_pts = 15 if abs_move >= 12 else 12 if abs_move >= 8 else 8 if abs_move >= 5 else 4 if abs_move >= 2 else 1
            ds_pts = 5 if downside_volatility >= 40 else 3 if downside_volatility >= 25 else 1

            risk_score = min(100, vol_pts + dd_pts + conf_pts + f_pts + ds_pts)
            if risk_score < 30:
                risk_level = "LOW"
                risk_color ="#4DE6FF"
            elif risk_score < 60:
                risk_level = "MEDIUM"
                risk_color = "#4F6BFF"
            else:
                risk_level = "HIGH"
                risk_color = "#7C5CFF"

            if pred_move_pct >= 1.0 and confidence >= 60:
                recommendation = "BUY"
                rec_color = "#4DE6FF"
                rec_icon = "↗"
            elif pred_move_pct <= -1.0 and confidence >= 60:
                recommendation = "AVOID"
                rec_color = "#7C5CFF"
                rec_icon = "↘"
            else:
                recommendation = "HOLD"
                rec_color = "#4F6BFF"
                rec_icon = "•"
            
            card_html = f"""
            <style>
                .premium-card {{
                    background: #132238;
                    border-radius: 16px;
                    padding: 24px;
                    border: 1px solid #29425F;
                    box-shadow: 0 10px 30px rgba(0,0,0,0.5);
                    font-family: 'Inter', sans-serif;
                    transition: transform 0.3s ease, box-shadow 0.3s ease;
                }}
                .premium-card:hover {{
                    transform: translateY(-5px);
                    box-shadow: 0 15px 40px rgba(0,0,0,0.7);
                    border: 1px solid rgba(255, 255, 255, 0.1);
                }}
                .glow-marker {{
                    position: absolute; top: -8px; height: 22px; width: 4px; 
                    background-color: {signal_color}; 
                    box-shadow: 0 0 12px {signal_color}, 0 0 24px {signal_color}; 
                    border-radius: 2px;
                    transition: left 1s cubic-bezier(0.4, 0, 0.2, 1);
                    cursor: pointer;
                }}
                .glow-marker::after {{
                    content: attr(data-price);
                    position: absolute; bottom: 30px; left: 50%; transform: translateX(-50%);
                    background: #252525; color: white; padding: 6px 10px; border-radius: 6px;
                    font-size: 13px; font-weight: bold; white-space: nowrap; opacity: 0;
                    pointer-events: none; transition: opacity 0.2s ease, bottom 0.2s ease;
                    border: 1px solid rgba(255,255,255,0.1); box-shadow: 0 4px 12px rgba(0,0,0,0.5); z-index: 10;
                }}
                .glow-marker:hover::after {{ opacity: 1; bottom: 35px; }}
                .sub-card {{
                    flex: 1; background: rgba(0,0,0,0.3); padding: 18px; border-radius: 12px;
                    border: 1px solid rgba(255,255,255,0.02); transition: background 0.3s ease;
                }}
                .sub-card:hover {{ background: rgba(0,0,0,0.5); }}
            </style>
            
            <div class="premium-card">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 4px;">
                    <h2 style="margin: 0; color: white; font-size: 24px; font-weight: 700; letter-spacing: -0.5px;">{selected_name}</h2>
                    <div style="background-color: {badge_bg}; color: {badge_color}; padding: 6px 14px; border-radius: 20px; font-size: 13px; font-weight: bold; box-shadow: 0 0 10px {badge_bg};">
                        {badge_icon}
                    </div>
                </div>
                <div style="color: #9FB4C8; font-size: 14px; margin-bottom: 24px; font-weight: 500;">{selected_horizon}-Day AI Forecast</div>
                
                <div style="display: flex; align-items: baseline; margin-bottom: 30px;">
                    <span style="font-size: 42px; font-weight: 800; color: white; letter-spacing: -1px;">${final_result['current_price']:,.2f}</span>
                    <span style="color: #9FB4C8; font-size: 15px; margin-left: 10px; font-weight: 500;">current close</span>
                </div>
                
                <div style="display: flex; justify-content: space-between; color: #9FB4C8; font-size: 13px; margin-bottom: 10px; font-weight: 600;">
                    <span>${visual_min:,.2f}</span>
                    <span style="text-transform: uppercase; letter-spacing: 1px; font-size: 11px;">Predicted Range</span>
                    <span>${visual_max:,.2f}</span>
                </div>
                
                <div style="position: relative; width: 100%; height: 6px; background-color: #2D3748; border-radius: 4px; margin-bottom: 30px; overflow: visible;">
                    <div style="position: absolute; left: 10%; right: 10%; height: 100%; background: linear-gradient(90deg, transparent, rgba(255,255,255,0.1), transparent); border-radius: 4px;"></div>
                    <div class="glow-marker" style="left: {marker_pos}%;" data-price="${final_result['predicted_price']:,.2f}"></div>
                </div>
                
                <div style="display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 12px; margin-bottom: 18px;">
                    <div class="sub-card">
                        <div style="color: #9FB4C8; font-size: 12px; margin-bottom: 6px; font-weight: 600; text-transform: uppercase; letter-spacing: 0.5px;">Confidence</div>
                        <div style="color: white; font-size: 22px; font-weight: 800;">{confidence:.1f}%</div>
                    </div>
                    <div class="sub-card">
                        <div style="color: #9FB4C8; font-size: 12px; margin-bottom: 6px; font-weight: 600; text-transform: uppercase; letter-spacing: 0.5px;">Proj. Move</div>
                        <div style="color: {signal_color}; font-size: 22px; font-weight: 800;">{pred_move_sign}{pred_move_pct:.2f}%</div>
                    </div>
                    <div class="sub-card">
                        <div style="color: #9FB4C8; font-size: 12px; margin-bottom: 6px; font-weight: 600; text-transform: uppercase; letter-spacing: 0.5px;">Risk</div>
                        <div style="color: {risk_color}; font-size: 16px; font-weight: 800;">{risk_level}</div>
                    </div>
                </div>

                <div style="background:#162338; padding:18px; border-radius:14px; margin-bottom:20px; border:1px solid #29425F;">
                    <div style="color:#9FB4C8; font-size:12px; text-transform:uppercase; font-weight:700; letter-spacing:0.7px;">Recommendation</div>
                    <div style="font-size:30px; font-weight:900; color:{rec_color}; margin-top:6px;">{rec_icon} {recommendation}</div>
                </div>
                
                <hr style="border: none; border-top: 1px solid rgba(255,255,255,0.05); margin-bottom: 20px;">
                
                {f'''<!-- Top Drivers -->
                <div style="color: #9FB4C8; font-size: 13px; margin-bottom: 16px; font-weight: 600; text-transform: uppercase; letter-spacing: 0.5px;">Primary Market Drivers</div>
                {driver_html}''' if driver_html else ""}
            </div>
            """
            
            clean_html = "\n".join([line.lstrip() for line in card_html.split("\n")])
            st.markdown(clean_html, unsafe_allow_html=True)
            st.markdown("<br>", unsafe_allow_html=True)

            plan_data = {
                "selected_name": selected_name,
                "selected_horizon_name": selected_horizon_name,
                "selected_horizon": selected_horizon,
                "pred_move_pct": pred_move_pct,
                "confidence": confidence,
                "current_price": final_result['current_price'],
                "predicted_price": final_result['predicted_price'],
                "pred_move_sign": pred_move_sign,
                "risk_level": risk_level,
                "risk_score": risk_score,
                "recommendation": recommendation,
                "rec_color": rec_color,
                "annualized_volatility": annualized_volatility,
                "max_drawdown_pct": max_drawdown_pct,
                "downside_volatility": downside_volatility,
                "models_used": final_result['models_used']
            }
            
        with simulator_btn_placeholder:
            if st.button(f"LAUNCH INVESTMENT SIMULATOR", use_container_width=True):
                show_investment_plan(plan_data)

    except Exception as e:
        st.error(f"Backend Engine Error: {str(e)}")
        st.info("Check if models and data pipeline are functioning correctly.")
