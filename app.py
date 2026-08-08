import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import timedelta
import os
import yfinance as yf

# Import our backend ensemble engine
from ensemble_inference import ensemble_predict

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
    /* Premium Dark Theme Overrides */
    .stApp {
        background-color: #0E1117;
        color: #FAFAFA;
    }
    
    /* Ticker Tape Animation */
    .ticker-wrap {
        width: 100%;
        overflow: hidden;
        background-color: #1A1C23;
        padding-left: 100%;
        box-sizing: content-box;
        border-top: 1px solid #333;
        border-bottom: 1px solid #333;
        margin-bottom: 2rem;
        white-space: nowrap;
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
        color: #E2E8F0;
    }
    .pos { color: #00FF7F; font-weight: bold; }
    .neg { color: #FF4136; font-weight: bold; }
    
    /* Mock CTA Button styling */
    .stButton>button {
        width: 100%;
        background-color: #2962FF !important;
        color: white !important;
        font-size: 18px !important;
        font-weight: bold !important;
        padding: 1rem !important;
        border-radius: 8px !important;
        border: none !important;
        transition: all 0.3s ease !important;
    }
    .stButton>button:hover {
        background-color: #1E4BD8 !important;
        box-shadow: 0px 4px 15px rgba(41, 98, 255, 0.4) !important;
    }
    
    /* Metrics Box Styling */
    .metric-container {
        background-color: #1A1C23;
        border-radius: 12px;
        padding: 20px;
        margin-bottom: 15px;
        border: 1px solid #2D3748;
    }
    .metric-title {
        font-size: 14px;
        color: #A0AEC0;
        margin-bottom: 5px;
    }
    .metric-value {
        font-size: 28px;
        font-weight: bold;
        color: #FFFFFF;
    }
    
    /* Custom Interactive Pills */
    div[data-testid="stPills"] {
        display: flex;
        justify-content: center;
        flex-wrap: wrap;
        margin-bottom: 10px;
    }
    div[data-testid="stPills"] label {
        display: none !important;
    }
    div[data-testid="stPills"] div[role="radiogroup"] {
        justify-content: center;
        gap: 12px;
    }
    div[data-testid="stPills"] button {
        padding: 12px 28px !important;
        font-size: 16px !important;
        font-weight: 700 !important;
        border-radius: 30px !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        background-color: rgba(255, 255, 255, 0.03) !important;
        color: #E2E8F0 !important;
        transition: all 0.3s ease !important;
        letter-spacing: 0.5px !important;
    }
    div[data-testid="stPills"] button:hover {
        background-color: rgba(255, 255, 255, 0.1) !important;
        border-color: rgba(255, 255, 255, 0.2) !important;
    }
    div[data-testid="stPills"] button[aria-checked="true"], 
    div[data-testid="stPills"] button[data-checked="true"] {
        background: linear-gradient(135deg, #00E676 0%, #00B85C 100%) !important;
        color: #0E1117 !important;
        box-shadow: 0 4px 15px rgba(0, 230, 118, 0.3) !important;
        border: none !important;
    }
</style>
""", unsafe_allow_html=True)

# --- CONFIG & STATE ---
COMMODITIES = {
    'Gold (GC=F)': 'gold',
    'Silver (SI=F)': 'silver',
    'Copper (HG=F)': 'copper',
    'Crude Oil (CL=F)': 'crude_oil',
    'Natural Gas (NG=F)': 'natural_gas',
    'Wheat (ZW=F)': 'wheat'
}

HORIZONS = {
    '1D': 1,
    '7D': 7,
    '14D': 14,
    '1M': 28,
    '6W': 42,
    '2M': 60,
    '1Q': 90,
    '4M': 120
}

# --- HELPER FUNCTIONS ---
@st.cache_data(ttl=3600)
def get_ticker_data():
    """Fetches latest prices from local CSV files for the ticker tape."""
    ticker_html = ""
    for name, key in COMMODITIES.items():
        file_path = os.path.join('./data', f'{key}.csv')
        if os.path.exists(file_path):
            df = pd.read_csv(file_path)
            last = df['Close'].iloc[-1]
            prev = df['Close'].iloc[-2]
            pct = ((last - prev) / prev) * 100
            
            color_class = 'pos' if pct >= 0 else 'neg'
            arrow = '▲' if pct >= 0 else '▼'
            sign = '+' if pct >= 0 else ''
            
            name_short = name.split(' ')[0]
            ticker_html += f"<div class='ticker-item'>{name_short} ${last:.2f} <span class='{color_class}'>{arrow} {sign}{pct:.2f}%</span></div>"
            
    return f"""
    <div class="ticker-wrap">
        <div class="ticker">
            {ticker_html}{ticker_html}{ticker_html}
        </div>
    </div>
    """

def get_chart_data(commodity_key):
    """Loads the last 90 days of historical data for plotting."""
    file_path = os.path.join('./data', f'{commodity_key}.csv')
    df = pd.read_csv(file_path, parse_dates=['Date'])
    df = df.sort_values('Date').tail(90)
    return df

def generate_market_drivers(historical_df):
    """Generates dynamic market driver text based on the latest data row."""
    drivers = []
    latest = historical_df.iloc[-1]
    
    if 'RSI_14' in historical_df.columns:
        if latest['RSI_14'] > 65:
            drivers.append(("RSI approaching overbought", "-"))
        elif latest['RSI_14'] < 35:
            drivers.append(("RSI heavily oversold", "+"))
            
    if 'Close_to_MA_50' in historical_df.columns:
        if latest['Close_to_MA_50'] > 1.0:
            drivers.append(("Price above 50-day moving average", "+"))
        else:
            drivers.append(("Price below 50-day moving average", "-"))
            
    if 'USD_Index' in historical_df.columns:
        dxy_5_days_ago = historical_df.iloc[-6]['USD_Index'] if len(historical_df) > 5 else latest['USD_Index']
        if latest['USD_Index'] < dxy_5_days_ago:
            drivers.append(("USD Index weakening", "+"))
        elif latest['USD_Index'] > dxy_5_days_ago:
            drivers.append(("USD Index strengthening", "-"))
            
    if 'US_10Y_Yield' in historical_df.columns:
        us10y_5_days_ago = historical_df.iloc[-6]['US_10Y_Yield'] if len(historical_df) > 5 else latest['US_10Y_Yield']
        if latest['US_10Y_Yield'] < us10y_5_days_ago:
            drivers.append(("Treasury yields falling", "+"))
        elif latest['US_10Y_Yield'] > us10y_5_days_ago:
            drivers.append(("Treasury yields rising", "-"))
            
    if 'VIX' in historical_df.columns:
        if latest['VIX'] > 20:
            drivers.append(("High macro volatility (VIX > 20)", "-"))
         
    # Take top 3
    return drivers[:3]

# --- UI RENDERING ---

# 1. Header & Ticker Tape
st.title("⚡ FundForge AI Terminal")
st.markdown(get_ticker_data(), unsafe_allow_html=True)

# 2. Main Terminal Controls
st.markdown("<br>", unsafe_allow_html=True)

st.markdown("<div style='text-align: center; color: #8B949E; font-size: 13px; font-weight: 600; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 8px;'>Select Asset</div>", unsafe_allow_html=True)
selected_name = st.pills(
    "Asset",
    options=list(COMMODITIES.keys()),
    default=list(COMMODITIES.keys())[0],
    selection_mode="single",
    label_visibility="collapsed"
)
if not selected_name:
    selected_name = list(COMMODITIES.keys())[0]
selected_key = COMMODITIES[selected_name]

st.markdown("<div style='text-align: center; color: #8B949E; font-size: 13px; font-weight: 600; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 8px; margin-top: 25px;'>Investment Horizon</div>", unsafe_allow_html=True)
selected_horizon_name = st.pills(
    "Horizon",
    options=list(HORIZONS.keys()),
    default='2M',
    selection_mode="single",
    label_visibility="collapsed"
)
if not selected_horizon_name:
    selected_horizon_name = '2M'
selected_horizon = HORIZONS[selected_horizon_name]

st.markdown("<br><br>", unsafe_allow_html=True)

# 3. Main Logic (Instant Execution)
with st.spinner("AI Models Computing Consensus..."):
    # Run backend ensemble inference
    try:
        # Collect multi-horizon path
        all_horizons = [1, 7, 14, 28, 42, 60, 90, 120]
        target_horizons = [h for h in all_horizons if h <= selected_horizon]
        
        historical_df = get_chart_data(selected_key)
        last_date = historical_df['Date'].iloc[-1]
        hist_dates = historical_df['Date'].tolist()
        hist_prices = historical_df['Close'].tolist()
        
        path_dates = [last_date]
        path_prices = [hist_prices[-1]]
        
        final_result = None
        # Run inference across the timeline to build the curve
        for h in target_horizons:
            res = ensemble_predict(selected_key, h, top_n=3)
            f_date = last_date + timedelta(days=int(h * 1.4))
            path_dates.append(f_date)
            path_prices.append(res['predicted_price'])
            if h == selected_horizon:
                final_result = res
                
        result = final_result
        
        # Determine Signal Formatting
        is_up = result['predicted_return'] > 0
        signal_text = "🟢 STRONG BUY" if is_up else "🔴 STRONG SELL"
        signal_color = "#00FF7F" if is_up else "#FF4136"
        
        # --- ASYMMETRICAL LAYOUT ---
        col1, col2 = st.columns([7, 3])
        
        with col1:
            st.subheader(f"{selected_name.split(' ')[0]} Trajectory Projection")
            
            # Interactive Plotly Chart
            fig = go.Figure()
            
            # Historical Area
            fig.add_trace(go.Scatter(
                x=hist_dates, y=hist_prices,
                fill='tozeroy',
                mode='lines',
                line=dict(color='#2962FF', width=2),
                fillcolor='rgba(41, 98, 255, 0.1)',
                name='Historical Price'
            ))
            
            # Multi-Horizon Projection Line (The Curve)
            fig.add_trace(go.Scatter(
                x=path_dates,
                y=path_prices,
                mode='lines+markers',
                line=dict(color=signal_color, width=3, dash='dash', shape='spline'), # Spline for smooth curve
                marker=dict(size=8, color=signal_color),
                name='AI Multi-Horizon Path'
            ))
            
            fig.update_layout(
                plot_bgcolor='#0E1117',
                paper_bgcolor='#0E1117',
                font=dict(color='#A0AEC0'),
                xaxis=dict(showgrid=False, title=''),
                yaxis=dict(showgrid=True, gridcolor='#2D3748', title='Price (USD)'),
                margin=dict(l=0, r=0, t=20, b=0),
                height=500,
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            st.caption(f"**Models inside current ensemble:** {', '.join(result['models_used'])}")
            
        with col2:
            st.subheader("Actionable Telemetry")
            
            import textwrap
            
            # Generate Dynamic Drivers
            drivers = generate_market_drivers(historical_df)
            driver_html = ""
            for driver_text, sentiment in drivers:
                color = "#00FF7F" if sentiment == "+" else "#FF4136"
                driver_html += f'''
<div class="driver-item">
    <span style="color: #E2E8F0; font-weight: 500;">{driver_text}</span>
    <span style="color: {color}; font-weight: 900; font-size: 18px; line-height: 1;">{sentiment}</span>
</div>
'''
                
            # Dynamic Pill Badge
            badge_bg = "rgba(0, 255, 127, 0.1)" if is_up else "rgba(255, 65, 54, 0.1)"
            badge_color = "#00FF7F" if is_up else "#FF4136"
            badge_icon = "📈 Bullish" if is_up else "📉 Bearish"
            
            # Range logic
            p_min = result.get('predicted_min', result['predicted_price'] * 0.98)
            p_max = result.get('predicted_max', result['predicted_price'] * 1.02)
            c_price = result['current_price']
            
            # Ensure p_min is the absolute minimum and p_max is absolute maximum
            visual_min = min(p_min, c_price)
            visual_max = max(p_max, c_price)
            
            # Prevent div by zero
            if visual_max == visual_min:
                visual_max += 0.01
                
            # Calculate marker position (0 to 100%)
            marker_pos = ((result['predicted_price'] - visual_min) / (visual_max - visual_min)) * 100
            marker_pos = max(0, min(100, marker_pos)) # clamp between 0 and 100
            
            confidence = result.get('confidence_pct', 72.0)
            pred_move_pct = result['predicted_return'] * 100
            pred_move_sign = "+" if pred_move_pct >= 0 else ""
            
            # Custom Premium Card HTML
            card_html = f"""
            <style>
                .premium-card {{
                    background: linear-gradient(145deg, #1A1C23 0%, #111318 100%);
                    border-radius: 16px;
                    padding: 24px;
                    border: 1px solid rgba(255, 255, 255, 0.05);
                    box-shadow: 0 10px 30px rgba(0,0,0,0.5);
                    font-family: 'Inter', sans-serif;
                    transition: transform 0.3s ease, box-shadow 0.3s ease;
                }}
                .premium-card:hover {{
                    transform: translateY(-5px);
                    box-shadow: 0 15px 40px rgba(0,0,0,0.7);
                    border: 1px solid rgba(255, 255, 255, 0.1);
                }}
                .driver-item {{
                    display: flex; justify-content: space-between; align-items: center; 
                    margin-bottom: 10px; padding: 12px 16px; 
                    background: rgba(255,255,255,0.03); 
                    border-radius: 8px; transition: all 0.2s ease;
                    font-size: 14px;
                }}
                .driver-item:hover {{
                    background: rgba(255,255,255,0.08);
                    transform: scale(1.02);
                    border-left: 3px solid {signal_color};
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
                    position: absolute;
                    bottom: 30px;
                    left: 50%;
                    transform: translateX(-50%);
                    background: #252525;
                    color: white;
                    padding: 6px 10px;
                    border-radius: 6px;
                    font-size: 13px;
                    font-weight: bold;
                    white-space: nowrap;
                    opacity: 0;
                    pointer-events: none;
                    transition: opacity 0.2s ease, bottom 0.2s ease;
                    border: 1px solid rgba(255,255,255,0.1);
                    box-shadow: 0 4px 12px rgba(0,0,0,0.5);
                    z-index: 10;
                }}
                .glow-marker:hover::after {{
                    opacity: 1;
                    bottom: 35px;
                }}
                .sub-card {{
                    flex: 1; background: rgba(0,0,0,0.3); padding: 18px; border-radius: 12px;
                    border: 1px solid rgba(255,255,255,0.02);
                    transition: background 0.3s ease;
                }}
                .sub-card:hover {{
                    background: rgba(0,0,0,0.5);
                }}
            </style>
            
            <div class="premium-card">
                <!-- Header -->
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 4px;">
                    <h2 style="margin: 0; color: white; font-size: 24px; font-weight: 700; letter-spacing: -0.5px;">{selected_name}</h2>
                    <div style="background-color: {badge_bg}; color: {badge_color}; padding: 6px 14px; border-radius: 20px; font-size: 13px; font-weight: bold; box-shadow: 0 0 10px {badge_bg};">
                        {badge_icon}
                    </div>
                </div>
                <div style="color: #8B949E; font-size: 14px; margin-bottom: 24px; font-weight: 500;">{selected_horizon}-Day AI Forecast</div>
                
                <!-- Current Price -->
                <div style="display: flex; align-items: baseline; margin-bottom: 30px;">
                    <span style="font-size: 42px; font-weight: 800; color: white; letter-spacing: -1px;">${result['current_price']:,.2f}</span>
                    <span style="color: #8B949E; font-size: 15px; margin-left: 10px; font-weight: 500;">current close</span>
                </div>
                
                <!-- Predicted Range Visualizer -->
                <div style="display: flex; justify-content: space-between; color: #8B949E; font-size: 13px; margin-bottom: 10px; font-weight: 600;">
                    <span>${visual_min:,.2f}</span>
                    <span style="text-transform: uppercase; letter-spacing: 1px; font-size: 11px;">Predicted Range</span>
                    <span>${visual_max:,.2f}</span>
                </div>
                
                <div style="position: relative; width: 100%; height: 6px; background-color: #2D3748; border-radius: 4px; margin-bottom: 30px; overflow: visible;">
                    <div style="position: absolute; left: 10%; right: 10%; height: 100%; background: linear-gradient(90deg, transparent, rgba(255,255,255,0.1), transparent); border-radius: 4px;"></div>
                    <div class="glow-marker" style="left: {marker_pos}%;" data-price="${result['predicted_price']:,.2f}"></div>
                </div>
                
                <!-- Sub-cards -->
                <div style="display: flex; gap: 16px; margin-bottom: 28px;">
                    <div class="sub-card">
                        <div style="color: #8B949E; font-size: 13px; margin-bottom: 6px; font-weight: 600; text-transform: uppercase; letter-spacing: 0.5px;">Confidence</div>
                        <div style="color: white; font-size: 24px; font-weight: 800;">{confidence:.1f}%</div>
                    </div>
                    <div class="sub-card" style="border-bottom: 3px solid {signal_color};">
                        <div style="color: #8B949E; font-size: 13px; margin-bottom: 6px; font-weight: 600; text-transform: uppercase; letter-spacing: 0.5px;">Projected Move</div>
                        <div style="color: {signal_color}; font-size: 24px; font-weight: 800;">{pred_move_sign}{pred_move_pct:.2f}%</div>
                    </div>
                </div>
                
                <hr style="border: none; border-top: 1px solid rgba(255,255,255,0.05); margin-bottom: 20px;">
                
                {f'''<!-- Top Drivers -->
                <div style="color: #8B949E; font-size: 13px; margin-bottom: 16px; font-weight: 600; text-transform: uppercase; letter-spacing: 0.5px;">Primary Market Drivers</div>
                {driver_html}''' if driver_html else ""}
            </div>
            """
            
            # Force remove all leading whitespace to completely prevent markdown code block rendering
            clean_html = "\n".join([line.lstrip() for line in card_html.split("\n")])
            
            st.markdown(clean_html, unsafe_allow_html=True)
            
            st.markdown("<br>", unsafe_allow_html=True)
            
            # Mock CTA Button
            button_label = f"EXECUTE {selected_name.split(' ')[0]} TRADE"
            st.button(button_label, use_container_width=True)
            st.caption("Note: This is a simulation environment. No real capital will be allocated.")

    except Exception as e:
        st.error(f"Backend Engine Error: {str(e)}")
        st.info("Check if models for this specific horizon and commodity exist in the /models directory.")
