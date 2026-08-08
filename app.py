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
    initial_sidebar_state="expanded"
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
        border-bottom: 1px solid #333;
        margin-top: -2rem;
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
    '1 Day (Tomorrow)': 1,
    '7 Days (1 Week)': 7,
    '14 Days (2 Weeks)': 14,
    '28 Days (1 Month)': 28,
    '42 Days (1.5 Months)': 42,
    '60 Days (2 Months)': 60,
    '90 Days (1 Quarter)': 90,
    '120 Days (Half Year)': 120
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
    
    # Check if Macro columns exist, fallback if not
    if 'RSI_14' in historical_df.columns:
        if latest['RSI_14'] > 70:
            drivers.append(("RSI approaching overbought", "-"))
        elif latest['RSI_14'] < 30:
            drivers.append(("RSI heavily oversold", "+"))
            
    if 'Close' in historical_df.columns and 'SMA_50' in historical_df.columns:
        if latest['Close'] > latest['SMA_50']:
            drivers.append(("Price above 50-day moving average", "+"))
        else:
            drivers.append(("Price below 50-day moving average", "-"))
            
    if 'Macro_DXY' in historical_df.columns:
        dxy_5_days_ago = historical_df.iloc[-6]['Macro_DXY'] if len(historical_df) > 5 else latest['Macro_DXY']
        if latest['Macro_DXY'] < dxy_5_days_ago:
            drivers.append(("USD Index weakening", "+"))
        elif latest['Macro_DXY'] > dxy_5_days_ago:
            drivers.append(("USD Index strengthening", "-"))
            
    if 'Macro_US10Y' in historical_df.columns:
        us10y_5_days_ago = historical_df.iloc[-6]['Macro_US10Y'] if len(historical_df) > 5 else latest['Macro_US10Y']
        if latest['Macro_US10Y'] < us10y_5_days_ago:
            drivers.append(("Real 10Y yield falling", "+"))
        elif latest['Macro_US10Y'] > us10y_5_days_ago:
            drivers.append(("Real 10Y yield rising", "-"))
            
    # Default fallbacks if no specific conditions triggered
    if len(drivers) == 0:
         drivers.append(("Normal market volatility", "+"))
         
    # Take top 3
    return drivers[:3]

# --- UI RENDERING ---

# 1. Ticker Tape Header
st.markdown(get_ticker_data(), unsafe_allow_html=True)
st.title("⚡ FundForge AI Terminal")

# 2. Sidebar Controls
with st.sidebar:
    st.header("Terminal Controls")
    
    selected_name = st.selectbox("Select Asset", list(COMMODITIES.keys()), index=0)
    selected_key = COMMODITIES[selected_name]
    
    selected_horizon_name = st.selectbox("Select Investment Horizon", list(HORIZONS.keys()), index=5) # Default 60d
    selected_horizon = HORIZONS[selected_horizon_name]
    
    st.markdown("---")
    st.markdown("""
    ### System Status
    🟢 Deep Learning Backend: **Online**  
    🟢 Machine Learning Backend: **Online**  
    🟢 Ensemble Engine: **Active**
    """)
    st.caption("v2.1.0 Production")

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
<div style="display: flex; justify-content: space-between; margin-bottom: 8px; font-size: 14px;">
    <span style="color: #E2E8F0; font-weight: 500;">{driver_text}</span>
    <span style="color: {color}; font-weight: bold;">{sentiment}</span>
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
            
            # Custom Premium Card HTML (dedented)
            card_html = textwrap.dedent(f"""
            <div style="background-color: #1E1E1E; border-radius: 16px; padding: 24px; border: 1px solid #333; font-family: 'Inter', sans-serif;">
                <!-- Header -->
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 4px;">
                    <h2 style="margin: 0; color: white; font-size: 22px;">{selected_name}</h2>
                    <div style="background-color: {badge_bg}; color: {badge_color}; padding: 4px 12px; border-radius: 20px; font-size: 12px; font-weight: bold;">
                        {badge_icon}
                    </div>
                </div>
                <div style="color: #A0AEC0; font-size: 14px; margin-bottom: 24px;">{selected_horizon}-day forecast</div>
                
                <!-- Current Price -->
                <div style="display: flex; align-items: baseline; margin-bottom: 24px;">
                    <span style="font-size: 36px; font-weight: bold; color: white;">${result['current_price']:,.2f}</span>
                    <span style="color: #A0AEC0; font-size: 14px; margin-left: 8px;">current close</span>
                </div>
                
                <!-- Predicted Range Visualizer -->
                <div style="display: flex; justify-content: space-between; color: #A0AEC0; font-size: 12px; margin-bottom: 8px;">
                    <span>${visual_min:,.2f}</span>
                    <span>Predicted range</span>
                    <span>${visual_max:,.2f}</span>
                </div>
                
                <div style="position: relative; width: 100%; height: 8px; background-color: #2D3748; border-radius: 4px; margin-bottom: 24px;">
                    <div style="position: absolute; left: 10%; right: 10%; height: 100%; background-color: rgba(0, 255, 127, 0.2); border-radius: 4px;"></div>
                    <div style="position: absolute; left: {marker_pos}%; top: -6px; height: 20px; width: 3px; background-color: {signal_color}; box-shadow: 0 0 8px {signal_color}; border-radius: 2px;"></div>
                </div>
                
                <!-- Sub-cards -->
                <div style="display: flex; gap: 12px; margin-bottom: 24px;">
                    <div style="flex: 1; background-color: #252525; padding: 16px; border-radius: 12px;">
                        <div style="color: #A0AEC0; font-size: 12px; margin-bottom: 4px;">Confidence</div>
                        <div style="color: white; font-size: 20px; font-weight: bold;">{confidence:.1f}%</div>
                    </div>
                    <div style="flex: 1; background-color: #252525; padding: 16px; border-radius: 12px;">
                        <div style="color: #A0AEC0; font-size: 12px; margin-bottom: 4px;">Model used</div>
                        <div style="color: white; font-size: 20px; font-weight: bold;">Ensemble</div>
                    </div>
                </div>
                
                <hr style="border: none; border-top: 1px solid #333; margin-bottom: 16px;">
                
                <!-- Top Drivers -->
                <div style="color: #A0AEC0; font-size: 14px; margin-bottom: 16px;">Top drivers this week</div>
                {driver_html}
            </div>
            """)
            
            st.markdown(card_html, unsafe_allow_html=True)
            
            st.markdown("<br>", unsafe_allow_html=True)
            
            # Mock CTA Button
            button_label = f"EXECUTE {selected_name.split(' ')[0]} TRADE"
            st.button(button_label, use_container_width=True)
            st.caption("Note: This is a simulation environment. No real capital will be allocated.")

    except Exception as e:
        st.error(f"Backend Engine Error: {str(e)}")
        st.info("Check if models for this specific horizon and commodity exist in the /models directory.")
