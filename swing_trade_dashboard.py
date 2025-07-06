import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta

# Configure page settings
st.set_page_config(
    page_title="Swing Trade Dashboard",
    page_icon="📈",
    layout="wide"
)

# Custom styling
st.markdown("""
<style>
    .big-font { font-size:24px !important; font-weight:bold; }
    .recommendation { padding:15px; border-radius:10px; margin:10px 0; }
    .buy { background-color:#e6f7e6; border-left:5px solid #4CAF50; }
    .sell { background-color:#ffe6e6; border-left:5px solid #f44336; }
    .hold { background-color:#e6f0ff; border-left:5px solid #2196F3; }
    .positive { color: #4CAF50; }
    .negative { color: #f44336; }
    .card { padding:15px; border-radius:10px; box-shadow:0 4px 6px rgba(0,0,0,0.1); margin-bottom:15px; }
</style>
""", unsafe_allow_html=True)

# SIDEBAR CONTROLS
st.sidebar.header("📊 Trading Controls")
symbol = st.sidebar.text_input("Stock Symbol", "AAPL").strip().upper()
period = st.sidebar.selectbox("Analysis Period", ["1M", "3M", "6M", "1Y"], index=1)
risk_level = st.sidebar.selectbox("Risk Tolerance", ["Low", "Medium", "High"], index=1)

# Convert period to days
period_map = {"1M": 30, "3M": 90, "6M": 180, "1Y": 365}
days = period_map[period]

# TECHNICAL ANALYSIS FUNCTIONS
def calculate_support_resistance(df):
    """Calculate meaningful support and resistance levels"""
    if len(df) < 20:
        return float(df['Close'].min()), float(df['Close'].max())
    
    # Get scalar values
    high = float(df['High'].max())
    low = float(df['Low'].min())
    diff = high - low
    
    fib_382 = high - diff * 0.382
    fib_618 = high - diff * 0.618
    
    # Get last values as scalars
    last_high = float(df['High'].iloc[-1])
    last_low = float(df['Low'].iloc[-1])
    last_close = float(df['Close'].iloc[-1])
    
    # Calculate pivot points
    pivot = (last_high + last_low + last_close) / 3
    s1 = 2 * pivot - last_high
    r1 = 2 * pivot - last_low
    
    # Get min/max as scalars
    min_close = float(df['Close'].tail(20).min())
    max_close = float(df['Close'].tail(20).max())
    
    # Calculate support using explicit comparison
    support = fib_382
    if s1 < support:
        support = s1
    if min_close < support:
        support = min_close
    
    # Calculate resistance using explicit comparison
    resistance = fib_618
    if r1 > resistance:
        resistance = r1
    if max_close > resistance:
        resistance = max_close
    
    # Ensure levels are reasonable relative to current price
    current = last_close
    if support < current * 0.7:  # If too low
        # Use the highest of the short-term values
        support = s1
        if fib_382 > support:
            support = fib_382
        short_min = float(df['Close'].tail(10).min())
        if short_min > support:
            support = short_min
    
    if resistance > current * 1.3:  # If too high
        # Use the lowest of the short-term values
        resistance = r1
        if fib_618 < resistance:
            resistance = fib_618
        short_max = float(df['Close'].tail(10).max())
        if short_max < resistance:
            resistance = short_max
    
    return support, resistance

def calculate_volatility(df):
    """Calculate average true range (ATR) for volatility"""
    if len(df) < 14:
        return 0.0
    
    high_low = df['High'] - df['Low']
    high_close = np.abs(df['High'] - df['Close'].shift())
    low_close = np.abs(df['Low'] - df['Close'].shift())
    
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return float(true_range.tail(14).mean())

def generate_recommendation(df, symbol, risk_level):
    """Generate professional trading recommendation"""
    current_price = float(df['Close'].iloc[-1])
    support, resistance = calculate_support_resistance(df)
    volatility = calculate_volatility(df)
    
    # Calculate price position relative to levels
    support_dist = (current_price - support) / current_price * 100
    resistance_dist = (resistance - current_price) / current_price * 100
    
    # Recommendation logic
    action = "HOLD"
    reason = "Price between key levels"
    confidence = "Medium"
    
    if support_dist < 3:  # Within 3% of support
        action = "BUY"
        reason = "Price near strong support"
        confidence = "High" if risk_level != "Low" else "Medium"
    elif resistance_dist < 3:  # Within 3% of resistance
        action = "SELL"
        reason = "Price near strong resistance"
        confidence = "High" if risk_level != "Low" else "Medium"
    
    # Calculate realistic trading ranges
    risk_factor = 1 if risk_level == "Low" else 1.5 if risk_level == "Medium" else 2
    buy_range = f"${max(support, current_price - volatility):.2f}-${current_price * 1.01:.2f}"
    sell_range = f"${current_price * 0.99:.2f}-${min(resistance, current_price + volatility):.2f}"
    
    # Calculate key metrics
    prev_close = float(df['Close'].iloc[-2]) if len(df) > 1 else current_price
    daily_change = ((current_price - prev_close) / prev_close) * 100
    
    return {
        "symbol": symbol,
        "current_price": current_price,
        "support": support,
        "resistance": resistance,
        "volatility": volatility,
        "daily_change": daily_change,
        "action": action,
        "reason": reason,
        "confidence": confidence,
        "buy_range": buy_range,
        "sell_range": sell_range
    }

# DATA FUNCTIONS
@st.cache_data
def get_stock_data(symbol, days):
    try:
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        data = yf.download(symbol, start=start_date, end=end_date)
        return data
    except Exception as e:
        st.error(f"Error fetching data: {e}")
        return pd.DataFrame()

# MAIN DASHBOARD
st.title("📊 Professional Swing Trading Dashboard")
st.markdown("Analyze stocks using advanced technical indicators for realistic trading recommendations")

if not symbol:
    st.warning("Please enter a stock symbol in the sidebar")
    st.stop()

stock_data = get_stock_data(symbol, days)

if stock_data.empty:
    st.error(f"No data found for {symbol}")
    st.stop()

try:
    recommendation = generate_recommendation(stock_data, symbol, risk_level)
    current_price = recommendation['current_price']

    # DASHBOARD LAYOUT
    col1, col2 = st.columns([1, 2])

    with col1:
        st.markdown(f"### Current Price: ${current_price:.2f}")
        
        # Key metrics cards
        st.markdown("#### Key Levels")
        col1a, col1b = st.columns(2)
        with col1a:
            st.markdown(f"""
                <div class="card">
                    <div>Support</div>
                    <div class="big-font">${recommendation['support']:.2f}</div>
                </div>
            """, unsafe_allow_html=True)
        with col1b:
            st.markdown(f"""
                <div class="card">
                    <div>Resistance</div>
                    <div class="big-font">${recommendation['resistance']:.2f}</div>
                </div>
            """, unsafe_allow_html=True)
        
        st.markdown("#### Market Data")
        col2a, col2b = st.columns(2)
        with col2a:
            change_class = "positive" if recommendation['daily_change'] >= 0 else "negative"
            st.markdown(f"""
                <div class="card">
                    <div>Daily Change</div>
                    <div class="big-font {change_class}">{recommendation['daily_change']:.2f}%</div>
                </div>
            """, unsafe_allow_html=True)
        with col2b:
            st.markdown(f"""
                <div class="card">
                    <div>Volatility (ATR)</div>
                    <div class="big-font">${recommendation['volatility']:.2f}</div>
                </div>
            """, unsafe_allow_html=True)
        
        # Trading recommendation
        st.markdown("#### Trading Recommendation")
        action_class = recommendation["action"].lower()
        st.markdown(f"""
            <div class="recommendation {action_class}">
                <div class="big-font">{recommendation['action']} {symbol}</div>
                <div>Confidence: {recommendation['confidence']}</div>
                <div>Reason: {recommendation['reason']}</div>
                <br>
                <div>Recommended Buy Range: {recommendation['buy_range']}</div>
                <div>Recommended Sell Range: {recommendation['sell_range']}</div>
            </div>
        """, unsafe_allow_html=True)
        
        # Additional metrics - FIXED: Convert to scalars before formatting
        st.markdown("#### Technical Indicators")
        st.metric("52-Week High", f"${float(stock_data['High'].max()):.2f}")
        st.metric("30-Day Average Volume", f"{float(stock_data['Volume'].tail(30).mean()):,.0f}")

    with col2:
        # Price chart with support/resistance
        st.markdown("### Price Analysis")
        fig = go.Figure()
        
        # Candlesticks
        fig.add_trace(go.Candlestick(
            x=stock_data.index,
            open=stock_data['Open'],
            high=stock_data['High'],
            low=stock_data['Low'],
            close=stock_data['Close'],
            name='Price'
        ))
        
        # Support and resistance
        fig.add_hline(y=recommendation['support'], line_dash="dash", line_color="green",
                    annotation_text=f"Support: ${recommendation['support']:.2f}", 
                    annotation_position="bottom right")
        fig.add_hline(y=recommendation['resistance'], line_dash="dash", line_color="red",
                    annotation_text=f"Resistance: ${recommendation['resistance']:.2f}", 
                    annotation_position="top right")
        
        fig.update_layout(
            title=f"{symbol} Price Chart",
            xaxis_title="Date",
            yaxis_title="Price",
            height=500,
            template="plotly_white",
            showlegend=False
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Volume chart
        st.markdown("### Volume Analysis")
        vol_fig = go.Figure()
        
        # Create color array for volume bars
        colors = np.where(stock_data['Close'] > stock_data['Open'], '#4CAF50', '#f44336')
        
        vol_fig.add_trace(go.Bar(
            x=stock_data.index,
            y=stock_data['Volume'],
            marker_color=colors,
            name='Volume'
        ))
        vol_fig.update_layout(
            height=300,
            template="plotly_white",
            showlegend=False,
            yaxis_title="Volume"
        )
        st.plotly_chart(vol_fig, use_container_width=True)

except Exception as e:
    st.error(f"An error occurred: {str(e)}")
    import traceback
    st.text(traceback.format_exc())
    st.write("Please try a different stock symbol or timeframe")

# STRATEGY EXPLANATION
st.markdown("---")
st.subheader("Trading Strategy Explanation")
st.markdown("""
**Professional Swing Trading Approach:**

1. **Support & Resistance Calculation**:
   - Uses Fibonacci retracement levels (38.2% and 61.8%)
   - Combines with traditional pivot points
   - Ensures levels are realistic relative to current price
   - Adjusts for extreme volatility cases

2. **Volatility-Based Trading Ranges**:
   - Uses Average True Range (ATR) to determine realistic entry/exit zones
   - Buy range: Support level to just above current price
   - Sell range: Just below current price to resistance level

3. **Realistic Recommendations**:
   - Buy signals only when near support (within 3%)
   - Sell signals only when near resistance (within 3%)
   - Hold recommendation with clear explanation when between levels

4. **Risk Management**:
   - Daily volatility measurement
   - Realistic profit targets
   - Clear entry/exit zones
""")

# DISCLAIMER
st.markdown("---")
st.caption("""
**Disclaimer**: This is for educational purposes only. Past performance is not indicative of future results. 
Always conduct your own research and consider consulting a financial advisor before trading.
""")