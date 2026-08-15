import os
import json
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px

# --- Streamlit Config ---
st.set_page_config(page_title="Institutional Crypto Forecaster", layout="wide", initial_sidebar_state="expanded")

RESULTS_DIR = "results/crypto"

@st.cache_data
def load_data():
    try:
        preds = pd.read_csv(f"{RESULTS_DIR}/final_predictions.csv")
        equity = pd.read_csv(f"{RESULTS_DIR}/equity_curve.csv")
        return preds, equity
    except Exception as e:
        st.error(f"Data missing! Have you run Phase 4 and Phase 6? Error: {e}")
        return None, None

preds_df, equity_df = load_data()

if preds_df is not None:
    # --- Sidebar ---
    st.sidebar.title("🧠 Antigravity Engine")
    st.sidebar.markdown("---")
    
    selected_asset = st.sidebar.selectbox("Select Asset", preds_df['Asset'].unique())
    
    # Dynamically filter horizons available for this specific asset
    available_horizons = preds_df[preds_df['Asset'] == selected_asset]['Horizon'].unique()
    selected_horizon = st.sidebar.selectbox("Select Horizon", available_horizons)
    
    st.sidebar.markdown("---")
    st.sidebar.info("Data Powered by: RTX 6000 Ada")
    
    # --- Main Content ---
    st.title(f"🚀 {selected_asset} Macro Forecast")
    
    # Filter Data
    asset_data = preds_df[(preds_df['Asset'] == selected_asset) & (preds_df['Horizon'] == selected_horizon)].iloc[0]
    
    # --- Top KPIs ---
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Current Price", f"${asset_data['Current_Price']:,.4f}")
    col2.metric("Predicted Target", f"${asset_data['Point_Prediction_Price']:,.4f}", f"{asset_data['Point_Prediction_Price'] - asset_data['Current_Price']:+,.4f}")
    col3.metric("Global Sentiment", f"{asset_data['Sentiment_Score']:+.4f}")
    
    action_color = "normal" if "HOLD" in asset_data['Action'] else "inverse" if "SELL" in asset_data['Action'] else "normal"
    col4.metric("AI Action", asset_data['Action'], delta_color=action_color)
    
    st.markdown("---")
    
    # --- Tabbed UI ---
    tab1, tab2, tab3 = st.tabs(["📊 Prediction Cone", "🧠 NLP Sentiment", "📈 Historical Backtest"])
    
    with tab1:
        st.subheader("Price Prediction Range")
        st.markdown("This cone represents the mathematical variance between the 9 Neural Network and Tree architectures.")
        
        # Create a simple line connecting Current Price to Predicted Prices
        fig = go.Figure()
        
        # Current Price Point
        fig.add_trace(go.Scatter(x=[0], y=[asset_data['Current_Price']], mode='markers+text', name='Today', text=['Today'], textposition="bottom center", marker=dict(size=12, color='blue')))
        
        # Target Cone
        fig.add_trace(go.Scatter(
            x=[1, 1], 
            y=[asset_data['Min_Prediction_Price'], asset_data['Max_Prediction_Price']], 
            mode='lines', 
            name='Confidence Interval', 
            line=dict(color='rgba(0,100,255,0.2)', width=30)
        ))
        
        # Point Prediction
        fig.add_trace(go.Scatter(x=[1], y=[asset_data['Point_Prediction_Price']], mode='markers+text', name='AI Target', text=[f"${asset_data['Point_Prediction_Price']:,.4f}"], textposition="top center", marker=dict(size=15, color='green' if asset_data['Point_Prediction_Price'] > asset_data['Current_Price'] else 'red')))
        
        fig.update_layout(
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis_title="USD Price",
            height=400,
            template="plotly_dark"
        )
        st.plotly_chart(fig, use_container_width=True)
        
    with tab2:
        st.subheader("FinBERT Market Emotion (Neural Network Input)")
        st.markdown(f"The Deep Learning models mathematically learned the historical relationship between this live score and price crashes.")
        
        score = asset_data['Sentiment_Score']
        
        fig2 = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = score,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': "Live Fear & Greed Index"},
            gauge = {
                'axis': {'range': [-1, 1]},
                'bar': {'color': "white"},
                'steps' : [
                    {'range': [-1.0, -0.4], 'color': "darkred"},
                    {'range': [-0.4, 0.1], 'color': "gray"},
                    {'range': [0.1, 1.0], 'color': "darkgreen"}
                ],
                'threshold' : {'line': {'color': "red", 'width': 4}, 'thickness': 0.75, 'value': score}
            }
        ))
        fig2.update_layout(height=400, template="plotly_dark")
        st.plotly_chart(fig2, use_container_width=True)
        
    with tab3:
        st.subheader("Simulated Equity Curve (Past 365 Days)")
        st.markdown("Proving the Engine's performance vs simply holding the asset.")
        
        fig3 = px.line(equity_df, x='Date', y=['Strategy_Equity', 'Buy_Hold_Equity'], title="AI Engine vs Buy & Hold")
        fig3.update_layout(yaxis_title="USD ($)", hovermode="x unified", template="plotly_dark")
        st.plotly_chart(fig3, use_container_width=True)
