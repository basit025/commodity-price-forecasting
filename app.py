import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
from inference import run_inference, fetch_live_data, COMMODITY_TICKERS

st.set_page_config(page_title="FundForge AI", layout="wide", page_icon="⚡")

# 1. Title (will be styled by the dynamic CSS injected later)
st.markdown('<h1 class="gradient-text" style="text-align: center; margin-bottom: 3rem;">FundForge AI</h1>', unsafe_allow_html=True)

# 2. Controls
col_spacer1, col_controls, col_spacer2 = st.columns([1, 2.5, 1])

with col_controls:
    col_a, col_b, col_c = st.columns(3)
    with col_a:
        commodity = st.selectbox("🎯 Target Asset", list(COMMODITY_TICKERS.keys()), format_func=lambda x: x.replace('_', ' ').title())
    with col_b:
        model_type = st.selectbox("🧠 Inference Engine", ["XGBoost", "LightGBM", "LSTM", "Transformer"])
    with col_c:
        chart_period = st.selectbox("📅 Chart History", ["3 Months", "6 Months", "1 Year", "2 Years", "5 Years"], index=1)
        period_map = {"3 Months": 90, "6 Months": 180, "1 Year": 365, "2 Years": 730, "5 Years": 1825}

# 3. Dynamic Theme Dictionary
THEMES = {
    'gold': {'grad': 'linear-gradient(45deg, #FFD700, #FFA500)', 'btn_grad': 'linear-gradient(90deg, #FFD700 0%, #FFA500 100%)', 'btn_hover': 'linear-gradient(90deg, #FFA500 0%, #FFD700 100%)', 'g1': 'rgba(255, 215, 0, 0.08)', 'g2': 'rgba(255, 165, 0, 0.08)', 'line': '#FFD700', 'fill': 'rgba(255, 215, 0, 0.1)', 'text': '#111'},
    'silver': {'grad': 'linear-gradient(45deg, #E0E0E0, #9E9E9E)', 'btn_grad': 'linear-gradient(90deg, #E0E0E0 0%, #9E9E9E 100%)', 'btn_hover': 'linear-gradient(90deg, #9E9E9E 0%, #E0E0E0 100%)', 'g1': 'rgba(224, 224, 224, 0.08)', 'g2': 'rgba(158, 158, 158, 0.08)', 'line': '#E0E0E0', 'fill': 'rgba(224, 224, 224, 0.1)', 'text': '#111'},
    'copper': {'grad': 'linear-gradient(45deg, #CD7F32, #8B4513)', 'btn_grad': 'linear-gradient(90deg, #CD7F32 0%, #8B4513 100%)', 'btn_hover': 'linear-gradient(90deg, #8B4513 0%, #CD7F32 100%)', 'g1': 'rgba(205, 127, 50, 0.08)', 'g2': 'rgba(139, 69, 19, 0.08)', 'line': '#CD7F32', 'fill': 'rgba(205, 127, 50, 0.1)', 'text': '#FFF'},
    'natural_gas': {'grad': 'linear-gradient(45deg, #00C6FF, #0072FF)', 'btn_grad': 'linear-gradient(90deg, #00C6FF 0%, #0072FF 100%)', 'btn_hover': 'linear-gradient(90deg, #0072FF 0%, #00C6FF 100%)', 'g1': 'rgba(0, 198, 255, 0.08)', 'g2': 'rgba(0, 114, 255, 0.08)', 'line': '#0072FF', 'fill': 'rgba(0, 114, 255, 0.1)', 'text': '#FFF'},
    'crude_oil': {'grad': 'linear-gradient(45deg, #8E2DE2, #4A00E0)', 'btn_grad': 'linear-gradient(90deg, #8E2DE2 0%, #4A00E0 100%)', 'btn_hover': 'linear-gradient(90deg, #4A00E0 0%, #8E2DE2 100%)', 'g1': 'rgba(142, 45, 226, 0.08)', 'g2': 'rgba(74, 0, 224, 0.08)', 'line': '#8E2DE2', 'fill': 'rgba(142, 45, 226, 0.1)', 'text': '#FFF'},
    'wheat': {'grad': 'linear-gradient(45deg, #F6D365, #FDA085)', 'btn_grad': 'linear-gradient(90deg, #F6D365 0%, #FDA085 100%)', 'btn_hover': 'linear-gradient(90deg, #FDA085 0%, #F6D365 100%)', 'g1': 'rgba(246, 211, 101, 0.08)', 'g2': 'rgba(253, 160, 133, 0.08)', 'line': '#F6D365', 'fill': 'rgba(246, 211, 101, 0.1)', 'text': '#111'}
}
t = THEMES[commodity]

# 4. Inject Dynamic CSS
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
        font-size: 4.5rem !important;
        font-weight: 900;
        margin-bottom: 0rem;
        letter-spacing: -1px;
    }}
    div[data-testid="metric-container"] {{
        background-color: #151923;
        border-radius: 16px;
        padding: 24px;
        box-shadow: 0 10px 20px rgba(0,0,0,0.4);
        border: 1px solid #232833;
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }}
    div[data-testid="metric-container"]:hover {{
        transform: translateY(-8px);
        box-shadow: 0 15px 30px {t['g1']};
        border: 1px solid {t['line']};
    }}
    .stButton>button {{
        background: {t['btn_grad']};
        color: {t['text']};
        border: none;
        padding: 15px 32px;
        text-align: center;
        text-decoration: none;
        display: inline-block;
        font-size: 18px;
        font-weight: 800;
        margin: 4px 2px;
        border-radius: 50px;
        transition: all 0.3s ease 0s;
        box-shadow: 0px 8px 15px rgba(0, 0, 0, 0.3);
        width: 100%;
        text-transform: uppercase;
        letter-spacing: 1.5px;
    }}
    .stButton>button:hover {{
        background: {t['btn_hover']};
        box-shadow: 0px 15px 20px {t['g2']};
        transform: translateY(-3px);
        color: {t['text']};
    }}
    .stButton>button:focus {{
        color: {t['text']};
    }}
    .block-container {{
        padding-top: 2rem !important;
    }}
</style>
""", unsafe_allow_html=True)

# 5. Button
with col_controls:
    st.markdown("<br>", unsafe_allow_html=True)
    b_col1, b_col2, b_col3 = st.columns([1, 2, 1])
    with b_col2:
        run_btn = st.button("Initiate Forecast Sequence 🔮")

st.markdown("<hr style='border: 1px solid #232833; margin-top: 2rem; margin-bottom: 2rem;'>", unsafe_allow_html=True)

# 6. Main Execution Logic
if run_btn:
    with st.spinner(f"Initiating {model_type} quantum inference for {commodity.upper()}..."):
        try:
            result = run_inference(commodity, model_type)
            df = fetch_live_data(commodity, days=period_map[chart_period])
            
            st.markdown("<h3 style='color: white;'>Real-Time Telemetry</h3>", unsafe_allow_html=True)
            col1, col2, col3, col4 = st.columns(4)
            
            col1.metric("Market Date", result['Last_Date'])
            col2.metric("Last Traded Price", f"${result['Last_Close']:,.2f}")
            
            delta_val = f"{result['Predicted_Return_Pct']}%"
            col3.metric("Projected Price (+1 Day)", f"${result['Predicted_Price']:,.2f}", delta_val)
            
            direction_color = "#00FF00" if result['Direction'] == "Up" else "#FF0000"
            arrow = "⇡" if result['Direction'] == "Up" else "⇣"
            
            col4.markdown(f"""
                <div style='background-color: #151923; border-radius: 16px; padding: 24px; border: 1px solid #232833; text-align: center; box-shadow: 0 10px 20px rgba(0,0,0,0.4);'>
                    <p style='color: #8A9BB1; font-size: 14px; margin: 0; font-weight: bold; text-transform: uppercase;'>Signal</p>
                    <p style='color: {direction_color}; font-size: 38px; font-weight: 900; margin: 0; text-shadow: 0 0 10px {direction_color}40;'>{arrow} {result['Direction'].upper()}</p>
                </div>
            """, unsafe_allow_html=True)
            
            st.markdown("<br>", unsafe_allow_html=True)
            
            st.markdown("<h3 style='color: white;'>Live Market Trajectory</h3>", unsafe_allow_html=True)
            
            fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                                vertical_spacing=0.03, row_heights=[0.8, 0.2])
            
            # Use dynamic theme for the chart
            fig.add_trace(go.Scatter(
                x=df.index, y=df['Close'],
                mode='lines',
                line=dict(color=t['line'], width=2),
                fill='tozeroy',
                fillcolor=t['fill'],
                name='Price Action'
            ), row=1, col=1)
            
            colors = ['#00FF00' if df['Close'].iloc[i] > df['Open'].iloc[i] else '#FF0000' for i in range(len(df))]
            fig.add_trace(go.Bar(
                x=df.index, y=df['Volume'],
                marker_color=colors,
                opacity=0.6,
                name='Volume'
            ), row=2, col=1)
            
            next_date = df.index[-1] + pd.Timedelta(days=1)
            fig.add_trace(go.Scatter(
                x=[df.index[-1], next_date],
                y=[df['Close'].iloc[-1], result['Predicted_Price']],
                mode='lines+markers',
                marker=dict(color=direction_color, size=14, symbol='circle', line=dict(color='white', width=2)),
                line=dict(color=direction_color, width=3, dash='dash'),
                name='AI Projection'
            ), row=1, col=1)
            
            # Use white/gray for MAs so they don't clash with the dynamic theme
            df['MA20'] = df['Close'].rolling(20).mean()
            df['MA50'] = df['Close'].rolling(50).mean()
            
            fig.add_trace(go.Scatter(x=df.index, y=df['MA20'], line=dict(color='#FFFFFF', width=1.2, dash='dot'), name='20-Day MA'))
            fig.add_trace(go.Scatter(x=df.index, y=df['MA50'], line=dict(color='#8A9BB1', width=1.2, dash='dash'), name='50-Day MA'))
            
            fig.update_layout(
                template="plotly_dark",
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                xaxis_rangeslider_visible=False,
                height=700,
                margin=dict(l=20, r=20, t=40, b=20),
                legend=dict(
                    orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1,
                    bgcolor='rgba(21, 25, 35, 0.8)', bordercolor='#232833', borderwidth=1
                ),
                xaxis=dict(gridcolor='#232833', showline=True, linewidth=1, linecolor='#232833'),
                yaxis=dict(gridcolor='#232833', showline=True, linewidth=1, linecolor='#232833'),
                xaxis2=dict(gridcolor='#232833', showline=True, linewidth=1, linecolor='#232833'),
                yaxis2=dict(showgrid=False, showticklabels=False)
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
        except Exception as e:
            st.error(f"System Error: {str(e)}")
else:
    st.markdown("""
        <div style="text-align: center; margin-top: 80px; padding: 40px; border-radius: 16px; background-color: rgba(21, 25, 35, 0.5); border: 1px dashed #3A4354;">
            <h2 style="color: #8A9BB1; font-weight: 300; letter-spacing: 2px;">SYSTEM STANDBY</h2>
            <p style="color: #3A4354; font-size: 1.1rem;">All Neural Networks loaded. Awaiting parameter configuration...</p>
        </div>
    """, unsafe_allow_html=True)
    
    st.markdown("<br><br>", unsafe_allow_html=True)
    
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("<div style='height: 120px; background-color: rgba(21, 25, 35, 0.3); border-radius: 16px; border: 1px solid rgba(35, 40, 51, 0.5);'></div>", unsafe_allow_html=True)
    with c2:
        st.markdown("<div style='height: 120px; background-color: rgba(21, 25, 35, 0.3); border-radius: 16px; border: 1px solid rgba(35, 40, 51, 0.5);'></div>", unsafe_allow_html=True)
    with c3:
        st.markdown("<div style='height: 120px; background-color: rgba(21, 25, 35, 0.3); border-radius: 16px; border: 1px solid rgba(35, 40, 51, 0.5);'></div>", unsafe_allow_html=True)
