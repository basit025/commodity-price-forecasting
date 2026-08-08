import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import yfinance as yf
from inference import run_inference, fetch_live_data, COMMODITY_TICKERS

st.set_page_config(page_title="FundForge AI", layout="wide", page_icon="⚡")

# --- 1. Ticker Tape Data Fetching ---
@st.cache_data(ttl=300)
def get_market_summary():
    tickers = list(COMMODITY_TICKERS.values())
    try:
        data = yf.download(tickers, period="5d", progress=False)
        summary = []
        for name, ticker in COMMODITY_TICKERS.items():
            if ticker in data['Close']:
                closes = data['Close'][ticker].dropna()
                if len(closes) >= 2:
                    current = closes.iloc[-1]
                    prev = closes.iloc[-2]
                    pct_change = ((current - prev) / prev) * 100
                    summary.append({
                        'name': name.replace('_', ' ').title(),
                        'price': current,
                        'change': pct_change
                    })
        return summary
    except Exception as e:
        return []

market_summary = get_market_summary()

# --- 2. Dynamic Theme Dictionary ---
THEMES = {
    'gold': {'grad': 'linear-gradient(45deg, #FFD700, #FFA500)', 'btn_grad': 'linear-gradient(90deg, #FFD700 0%, #FFA500 100%)', 'btn_hover': 'linear-gradient(90deg, #FFA500 0%, #FFD700 100%)', 'g1': 'rgba(255, 215, 0, 0.08)', 'g2': 'rgba(255, 165, 0, 0.08)', 'line': '#FFD700', 'fill': 'rgba(255, 215, 0, 0.1)', 'text': '#111'},
    'silver': {'grad': 'linear-gradient(45deg, #E0E0E0, #9E9E9E)', 'btn_grad': 'linear-gradient(90deg, #E0E0E0 0%, #9E9E9E 100%)', 'btn_hover': 'linear-gradient(90deg, #9E9E9E 0%, #E0E0E0 100%)', 'g1': 'rgba(224, 224, 224, 0.08)', 'g2': 'rgba(158, 158, 158, 0.08)', 'line': '#E0E0E0', 'fill': 'rgba(224, 224, 224, 0.1)', 'text': '#111'},
    'copper': {'grad': 'linear-gradient(45deg, #CD7F32, #8B4513)', 'btn_grad': 'linear-gradient(90deg, #CD7F32 0%, #8B4513 100%)', 'btn_hover': 'linear-gradient(90deg, #8B4513 0%, #CD7F32 100%)', 'g1': 'rgba(205, 127, 50, 0.08)', 'g2': 'rgba(139, 69, 19, 0.08)', 'line': '#CD7F32', 'fill': 'rgba(205, 127, 50, 0.1)', 'text': '#FFF'},
    'natural_gas': {'grad': 'linear-gradient(45deg, #00C6FF, #0072FF)', 'btn_grad': 'linear-gradient(90deg, #00C6FF 0%, #0072FF 100%)', 'btn_hover': 'linear-gradient(90deg, #0072FF 0%, #00C6FF 100%)', 'g1': 'rgba(0, 198, 255, 0.08)', 'g2': 'rgba(0, 114, 255, 0.08)', 'line': '#0072FF', 'fill': 'rgba(0, 114, 255, 0.1)', 'text': '#FFF'},
    'crude_oil': {'grad': 'linear-gradient(45deg, #8E2DE2, #4A00E0)', 'btn_grad': 'linear-gradient(90deg, #8E2DE2 0%, #4A00E0 100%)', 'btn_hover': 'linear-gradient(90deg, #4A00E0 0%, #8E2DE2 100%)', 'g1': 'rgba(142, 45, 226, 0.08)', 'g2': 'rgba(74, 0, 224, 0.08)', 'line': '#8E2DE2', 'fill': 'rgba(142, 45, 226, 0.1)', 'text': '#FFF'},
    'wheat': {'grad': 'linear-gradient(45deg, #F6D365, #FDA085)', 'btn_grad': 'linear-gradient(90deg, #F6D365 0%, #FDA085 100%)', 'btn_hover': 'linear-gradient(90deg, #FDA085 0%, #F6D365 100%)', 'g1': 'rgba(246, 211, 101, 0.08)', 'g2': 'rgba(253, 160, 133, 0.08)', 'line': '#F6D365', 'fill': 'rgba(246, 211, 101, 0.1)', 'text': '#111'}
}

# Sidebar Controls for frictionless interaction
with st.sidebar:
    st.markdown("<h2 style='text-align: center; color: white;'>Platform Settings</h2>", unsafe_allow_html=True)
    commodity = st.selectbox("🎯 Target Asset", list(COMMODITY_TICKERS.keys()), format_func=lambda x: x.replace('_', ' ').title())
    model_type = st.selectbox("🧠 AI Engine", ["LSTM", "Transformer", "XGBoost", "LightGBM"], index=0)
    chart_period = st.selectbox("📅 Chart History", ["3 Months", "6 Months", "1 Year", "2 Years", "5 Years"], index=1)
    period_map = {"3 Months": 90, "6 Months": 180, "1 Year": 365, "2 Years": 730, "5 Years": 1825}
    
    st.markdown("<br><hr style='border: 1px solid #232833;'><p style='text-align: center; color: #8A9BB1; font-size: 0.9rem;'>Powered by FundForge AI</p>", unsafe_allow_html=True)

t = THEMES[commodity]

# --- 3. Dynamic CSS Injection ---
st.markdown(f"""
<style>
    .stApp {{
        background-color: #0B0E14;
        background-image: 
            radial-gradient(circle at 15% 50%, {t['g1']}, transparent 25%),
            radial-gradient(circle at 85% 30%, {t['g2']}, transparent 25%),
            linear-gradient(rgba(255, 255, 255, 0.015) 1px, transparent 1px),
            linear-gradient(90deg, rgba(255, 255, 255, 0.015) 1px, transparent 1px);
        background-size: 100% 100%, 100% 100%, 40px 40px, 40px 40px;
        font-family: 'Inter', sans-serif;
    }}
    .gradient-text {{
        background: {t['grad']};
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 3rem !important;
        font-weight: 900;
        margin-bottom: 0rem;
        letter-spacing: -1px;
    }}
    /* Ticker Tape Animation */
    .ticker-wrap {{
        width: 100%;
        overflow: hidden;
        background-color: rgba(21, 25, 35, 0.9);
        border-bottom: 1px solid #232833;
        border-top: 1px solid #232833;
        padding-left: 20px;
        box-sizing: border-box;
        display: flex;
        align-items: center;
        height: 50px;
        margin-bottom: 2rem;
        margin-top: -2rem; /* Pull up to offset default streamlit padding */
    }}
    .ticker-content {{
        display: flex;
        white-space: nowrap;
        animation: ticker 40s linear infinite;
    }}
    .ticker-content:hover {{
        animation-play-state: paused;
    }}
    @keyframes ticker {{
        0%   {{ transform: translateX(0); }}
        100% {{ transform: translateX(-50%); }}
    }}
    .ticker-item {{
        display: inline-block;
        padding: 0 40px;
        font-size: 15px;
        color: #8A9BB1;
        font-weight: 600;
    }}
    .ticker-item span.up {{ color: #00FF00; font-weight: 800; }}
    .ticker-item span.down {{ color: #FF0000; font-weight: 800; }}
    
    div[data-testid="metric-container"] {{
        background-color: #151923;
        border-radius: 12px;
        padding: 16px;
        box-shadow: 0 5px 15px rgba(0,0,0,0.4);
        border: 1px solid #232833;
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }}
    div[data-testid="metric-container"]:hover {{
        transform: translateY(-4px);
        box-shadow: 0 10px 20px {t['g1']};
        border: 1px solid {t['line']};
    }}
    
    .cta-btn {{
        background: {t['btn_grad']};
        color: {t['text']};
        border: none;
        padding: 18px 24px;
        text-align: center;
        text-decoration: none;
        display: inline-block;
        font-size: 16px;
        font-weight: 900;
        margin-top: 1rem;
        border-radius: 8px;
        transition: all 0.3s ease 0s;
        box-shadow: 0px 8px 15px rgba(0, 0, 0, 0.3);
        width: 100%;
        text-transform: uppercase;
        letter-spacing: 1px;
        cursor: pointer;
    }}
    .cta-btn:hover {{
        background: {t['btn_hover']};
        box-shadow: 0px 15px 20px {t['g2']};
        transform: translateY(-2px);
    }}
</style>
""", unsafe_allow_html=True)

# --- 4. Render Ticker Tape ---
if market_summary:
    # We duplicate the items to make the seamless infinite scroll work properly
    items_html = ""
    for item in market_summary:
        color_class = "up" if item['change'] >= 0 else "down"
        arrow = "▲" if item['change'] >= 0 else "▼"
        items_html += f"<div class='ticker-item'>{item['name']} <span style='color: white; margin-left: 5px;'>${item['price']:,.2f}</span> <span class='{color_class}' style='margin-left: 5px;'>{arrow} {abs(item['change']):.2f}%</span></div>"
    
    ticker_html = f"<div class='ticker-wrap'><div class='ticker-content'>{items_html}{items_html}</div></div>"
    st.markdown(ticker_html, unsafe_allow_html=True)

# --- 5. Main Dashboard Header ---
st.markdown(f'<h1 class="gradient-text">FundForge AI : {commodity.replace("_", " ").title()}</h1>', unsafe_allow_html=True)
st.markdown("<p style='color: #8A9BB1; font-size: 1.1rem; margin-bottom: 2rem; font-weight: 500;'>Institutional Trading Signals & Predictive Analytics</p>", unsafe_allow_html=True)

# --- 6. Instant Execution Logic ---
try:
    with st.spinner(f"Aggregating {model_type} intelligence for {commodity.upper()}..."):
        result = run_inference(commodity, model_type)
        df = fetch_live_data(commodity, days=period_map[chart_period])
        
        # --- 7. Side-by-Side Layout ---
        col_chart, col_telemetry = st.columns([7, 3])
        
        with col_chart:
            # Build Plotly Chart
            fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                                vertical_spacing=0.03, row_heights=[0.8, 0.2])
            
            # Area Chart
            fig.add_trace(go.Scatter(
                x=df.index, y=df['Close'],
                mode='lines',
                line=dict(color=t['line'], width=2.5),
                fill='tozeroy',
                fillcolor=t['fill'],
                name='Price'
            ), row=1, col=1)
            
            # Volume Bars
            colors = ['#00FF00' if df['Close'].iloc[i] > df['Open'].iloc[i] else '#FF0000' for i in range(len(df))]
            fig.add_trace(go.Bar(
                x=df.index, y=df['Volume'],
                marker_color=colors,
                opacity=0.6,
                name='Volume'
            ), row=2, col=1)
            
            # Projection
            next_date = df.index[-1] + pd.Timedelta(days=1)
            direction_color = "#00FF00" if result['Direction'] == "Up" else "#FF0000"
            
            fig.add_trace(go.Scatter(
                x=[df.index[-1], next_date],
                y=[df['Close'].iloc[-1], result['Predicted_Price']],
                mode='lines+markers',
                marker=dict(color=direction_color, size=14, symbol='circle', line=dict(color='white', width=2)),
                line=dict(color=direction_color, width=3, dash='dash'),
                name='AI Target'
            ), row=1, col=1)
            
            # MAs
            df['MA20'] = df['Close'].rolling(20).mean()
            df['MA50'] = df['Close'].rolling(50).mean()
            
            fig.add_trace(go.Scatter(x=df.index, y=df['MA20'], line=dict(color='#FFFFFF', width=1.2, dash='dot'), name='20-Day MA'))
            fig.add_trace(go.Scatter(x=df.index, y=df['MA50'], line=dict(color='#8A9BB1', width=1.2, dash='dash'), name='50-Day MA'))
            
            fig.update_layout(
                template="plotly_dark",
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                xaxis_rangeslider_visible=False,
                height=600,
                margin=dict(l=0, r=0, t=10, b=0),
                legend=dict(
                    orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0.01,
                    bgcolor='rgba(21, 25, 35, 0.8)', bordercolor='#232833', borderwidth=1
                ),
                xaxis=dict(gridcolor='#232833', showline=True, linewidth=1, linecolor='#232833'),
                yaxis=dict(gridcolor='#232833', showline=True, linewidth=1, linecolor='#232833'),
                xaxis2=dict(gridcolor='#232833', showline=True, linewidth=1, linecolor='#232833'),
                yaxis2=dict(showgrid=False, showticklabels=False)
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
        with col_telemetry:
            # Telemetry Metrics Stacked Vertically on the right
            st.markdown("<h3 style='color: white; margin-top: 0;'>Market Telemetry</h3>", unsafe_allow_html=True)
            
            c1, c2 = st.columns(2)
            c1.metric("Date", result['Last_Date'])
            c2.metric("Engine", model_type)
            
            st.metric("Current Market Price", f"${result['Last_Close']:,.2f}")
            
            delta_val = f"{result['Predicted_Return_Pct']}% Expected Move"
            st.metric("AI Price Target (1D)", f"${result['Predicted_Price']:,.2f}", delta_val)
            
            # CTA Signal Box
            signal_text = "STRONG BUY" if result['Direction'] == "Up" else "STRONG SELL"
            arrow = "⇡" if result['Direction'] == "Up" else "⇣"
            
            st.markdown(f"""
                <div style='background-color: #151923; border-radius: 12px; padding: 24px; border: 1px solid #232833; text-align: center; box-shadow: 0 10px 20px rgba(0,0,0,0.4); margin-top: 1.5rem;'>
                    <p style='color: #8A9BB1; font-size: 14px; margin: 0; font-weight: bold; text-transform: uppercase;'>Algorithmic Signal</p>
                    <p style='color: {direction_color}; font-size: 36px; font-weight: 900; margin: 0; text-shadow: 0 0 15px {direction_color}60;'>{arrow} {signal_text}</p>
                </div>
            """, unsafe_allow_html=True)
            
            # Interactive CTA Button
            st.markdown(f"""
                <a href="#" style="text-decoration: none;">
                    <button class="cta-btn">Execute Trade At Market</button>
                </a>
            """, unsafe_allow_html=True)
            
except Exception as e:
    st.error(f"System Error: {str(e)}")
