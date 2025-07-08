import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta

# Configure page settings
st.set_page_config(
    page_title="Professional Swing Trading Dashboard",
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
    .narrative { background-color:#f8f9fa; padding:20px; border-radius:10px; border-left:5px solid #6c757d; }
    .warning { background-color:#fff3cd; padding:10px; border-radius:5px; border-left:5px solid #ffc107; }
    .range-info { margin-top: 10px; }
    .trade-levels { margin-top: 10px; }
</style>
""", unsafe_allow_html=True)

# SIDEBAR CONTROLS
st.sidebar.header("📊 Trading Controls")
symbol = st.sidebar.text_input("Stock Symbol", "AAPL").strip().upper()
timeframe = st.sidebar.selectbox("Chart Timeframe", 
                                ["15m", "30m", "1h", "4h", "Daily"], 
                                index=0)
risk_level = st.sidebar.selectbox("Risk Tolerance", ["Low", "Medium", "High"], index=1)
risk_reward = st.sidebar.slider("Risk:Reward Ratio", 1.0, 5.0, 2.0, 0.5)

# TECHNICAL ANALYSIS FUNCTIONS
def calculate_support_resistance(df):
    """Calculate meaningful support and resistance levels"""
    if len(df) < 20:
        return float(df['Close'].min()), float(df['Close'].max())
    
    # Get scalar values
    high = float(df['High'].max())
    low = float(df['Low'].min())
    diff = high - low
    
    fib_382 = float(high - diff * 0.382)
    fib_618 = float(high - diff * 0.618)
    
    # Get last values as scalars
    last_high = float(df['High'].iloc[-1])
    last_low = float(df['Low'].iloc[-1])
    last_close = float(df['Close'].iloc[-1])
    
    # Calculate pivot points
    pivot = float((last_high + last_low + last_close) / 3)
    s1 = float(2 * pivot - last_high)
    r1 = float(2 * pivot - last_low)
    
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
        # Convert all values to floats before comparison
        support = max(float(s1), float(fib_382), float(df['Close'].tail(10).min()))
    
    if resistance > current * 1.3:  # If too high
        # Convert all values to floats before comparison
        resistance = min(float(r1), float(fib_618), float(df['Close'].tail(10).max()))
    
    return float(support), float(resistance)

def calculate_volatility(df):
    """Calculate average true range (ATR) for volatility"""
    if len(df) < 14:
        return 0.0
    
    high_low = df['High'] - df['Low']
    high_close = np.abs(df['High'] - df['Close'].shift())
    low_close = np.abs(df['Low'] - df['Close'].shift())
    
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return float(true_range.tail(14).mean())

def calculate_rsi(df, period=14):
    """Calculate Relative Strength Index (RSI)"""
    if len(df) < period:
        return 0.0
    
    delta = df['Close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    
    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()
    
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return float(rsi.iloc[-1])

def calculate_vwap(df):
    """Calculate Volume Weighted Average Price (VWAP)"""
    if len(df) < 1:
        return 0.0
    
    typical_price = (df['High'] + df['Low'] + df['Close']) / 3
    vwap = (typical_price * df['Volume']).cumsum() / df['Volume'].cumsum()
    return float(vwap.iloc[-1])

def generate_recommendation(df, symbol, risk_level):
    """Generate professional trading recommendation"""
    current_price = float(df['Close'].iloc[-1])
    support, resistance = calculate_support_resistance(df)
    volatility = calculate_volatility(df)
    rsi = calculate_rsi(df)
    vwap = calculate_vwap(df)
    
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
    
    # Context-aware trading ranges
    if action == "BUY":
        buy_range = f"${max(support, current_price - volatility):.2f}-${min(current_price * 1.01, resistance):.2f}"
        sell_range = f"${min(current_price * 1.02, resistance):.2f}-${min(current_price * 1.05, resistance * 1.03):.2f}"
    elif action == "SELL":
        buy_range = f"${max(support * 0.97, current_price * 0.95):.2f}-${max(support, current_price * 0.98):.2f}"
        sell_range = f"${max(current_price * 0.99, support):.2f}-${min(resistance, current_price * 1.01):.2f}"
    else:  # HOLD
        buy_range = f"${max(support, current_price - volatility):.2f}-${current_price:.2f}"
        sell_range = f"${current_price:.2f}-${min(resistance, current_price + volatility):.2f}"
    
    # Calculate key metrics
    if len(df) > 1:
        prev_close = float(df['Close'].iloc[-2])
        daily_change = float(((current_price - prev_close) / prev_close) * 100)
    else:
        daily_change = 0.0
    
    return {
        "symbol": symbol,
        "current_price": float(current_price),
        "support": float(support),
        "resistance": float(resistance),
        "volatility": float(volatility),
        "rsi": float(rsi),
        "vwap": float(vwap),
        "daily_change": float(daily_change),
        "action": action,
        "reason": reason,
        "confidence": confidence,
        "buy_range": buy_range,
        "sell_range": sell_range
    }

def generate_narrative(recommendation, stock_data):
    """Generate beginner-friendly narrative analysis of price action"""
    try:
        # Check for sufficient data
        if len(stock_data) < 20:
            return f"**Beginner-Friendly Analysis of {recommendation['symbol']}**\n\n⚠️ **Warning**: Insufficient data for detailed analysis ({len(stock_data)} bars available). Requires at least 20 bars."
        
        # Safely extract all values as scalars
        current_price = float(recommendation['current_price'])
        support = float(recommendation['support'])
        resistance = float(recommendation['resistance'])
        volatility = float(recommendation['volatility'])
        daily_change = float(recommendation['daily_change'])
        rsi = float(recommendation['rsi'])
        vwap = float(recommendation['vwap'])
        
        # Safely calculate trend direction
        short_term = float(stock_data['Close'].tail(5).mean())
        medium_term = float(stock_data['Close'].tail(20).mean())
        
        # Determine trend with explicit comparisons
        trend = "sideways/range-bound"
        if current_price > medium_term and medium_term > short_term:
            trend = "strong uptrend"
        elif current_price > medium_term:
            trend = "moderate uptrend"
        elif current_price < medium_term and medium_term < short_term:
            trend = "strong downtrend"
        elif current_price < medium_term:
            trend = "moderate downtrend"
        
        # Initialize gap variables
        gap_info = ""
        gap_target = ""
        prev_close_value = None
        
        # Safely check for gaps
        if len(stock_data) >= 2:
            try:
                # Get values using iloc for position-based access
                prev_close_value = float(stock_data['Close'].iloc[-2])
                current_open_value = float(stock_data['Open'].iloc[-1])
                
                # Calculate gap percentage
                gap_percent = ((current_open_value - prev_close_value) / prev_close_value) * 100
                
                if gap_percent > 2.0:
                    gap_info = f"with a significant gap up ({gap_percent:.1f}%)"
                    gap_target = f"Gap fill target: ${prev_close_value:.2f}"
                elif gap_percent < -2.0:
                    gap_info = f"with a significant gap down ({abs(gap_percent):.1f}%)"
                    gap_target = f"Gap fill target: ${prev_close_value:.2f}"
                elif gap_percent > 0:
                    gap_info = "with a small gap up"
                elif gap_percent < 0:
                    gap_info = "with a small gap down"
            except Exception as e:
                gap_info = f"(gap analysis error: {str(e)})"
        
        # Price position analysis
        position = "trading in the middle of its recent range"
        price_range = resistance - support
        
        if price_range > 0:  # Ensure valid range
            support_zone = support + price_range * 0.3
            resistance_zone = resistance - price_range * 0.3
            
            if current_price < support_zone:
                position = "trading near support levels"
            elif current_price > resistance_zone:
                position = "trading near resistance levels"
        
        # Volatility context
        vol_context = "low volatility - smaller price moves expected"
        volatility_ratio = volatility / current_price if current_price > 0 else 0
        
        if volatility_ratio > 0.03:
            vol_context = "high volatility - expect larger price swings"
        elif volatility_ratio > 0.015:
            vol_context = "moderate volatility - normal price movements"
        
        # RSI interpretation
        rsi_context = "RSI shows neutral momentum"
        if rsi > 70:
            rsi_context = "RSI indicates overbought conditions"
        elif rsi < 30:
            rsi_context = "RSI indicates oversold conditions"
        
        # Create narrative
        narrative = f"""
**Beginner-Friendly Analysis of {recommendation['symbol']}**

- **Overall Trend**: The stock is in a **{trend}** {gap_info}
- **Price Position**: Currently **{position}** (Support: ${support:.2f}, Resistance: ${resistance:.2f})
- **Key Indicators**:
  - Today's change: **{daily_change:.2f}%** ({vol_context})
  - RSI: **{rsi:.1f}** ({rsi_context})
  - VWAP: **${vwap:.2f}** (volume-weighted average)
- **Important Levels**:
  - Upside target: ${resistance:.2f}
  - Downside target: ${support:.2f}
  - Volatility range: ±${volatility:.2f} from current price
  {f"- {gap_target}" if gap_target else ""}

**Beginner Trading Tips**:
1. In **{trend.split()[0]} trends**, trade in the trend direction
2. Place stop-loss orders below support for buy positions, above resistance for sell positions
3. Take partial profits near key levels to lock in gains
4. {f"Watch for gap fill at ${prev_close_value:.2f}" if gap_target else "No significant gap to fill"}
"""
        return narrative
        
    except Exception as e:
        import traceback
        return f"**Error generating narrative:** {str(e)}\n\n```{traceback.format_exc()}```"

# DATA FUNCTIONS
@st.cache_data
def get_stock_data(symbol, timeframe):
    try:
        period_map = {
            "15m": "60d",
            "30m": "60d",
            "1h": "60d",
            "4h": "60d",
            "Daily": "1y"
        }
        
        interval_map = {
            "15m": "15m",
            "30m": "30m",
            "1h": "60m",
            "4h": "240m",
            "Daily": "1d"
        }
        
        data = yf.download(
            symbol, 
            period=period_map[timeframe],
            interval=interval_map[timeframe],
            progress=False
        )
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

stock_data = get_stock_data(symbol, timeframe)

if stock_data.empty:
    st.error(f"No data found for {symbol}")
    st.stop()

# Clean and validate data
stock_data = stock_data.dropna()

# Check data sufficiency
if len(stock_data) < 2:
    st.error(f"⚠️ Insufficient historical data ({len(stock_data)} bars available). Requires at least 2 bars.")
    st.stop()

try:
    recommendation = generate_recommendation(stock_data, symbol, risk_level)
    current_price = float(recommendation['current_price'])
    volatility = float(recommendation['volatility'])
    
    # Calculate stop-loss and take-profit
    support_val = float(recommendation['support'])
    resistance_val = float(recommendation['resistance'])
    
    if recommendation['action'] == "BUY":
        stop_loss = max(support_val, current_price - volatility * risk_reward)
        take_profit = min(resistance_val, current_price + volatility * risk_reward * 2)
    elif recommendation['action'] == "SELL":
        stop_loss = min(resistance_val, current_price + volatility * risk_reward)
        take_profit = max(support_val, current_price - volatility * risk_reward * 2)
    else:
        stop_loss = current_price - volatility * risk_reward
        take_profit = current_price + volatility * risk_reward * 2

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
        
        # Create the recommendation using simple HTML without complex formatting
        rec_html = f"""
        <div class="recommendation {action_class}">
            <div class="big-font">{recommendation['action']} {symbol}</div>
            <div>Confidence: {recommendation['confidence']}</div>
            <div>Reason: {recommendation['reason']}</div>
            <br>
        """
        
        if recommendation["action"] == "BUY":
            rec_html += f"<div>Entry Zone: {recommendation['buy_range']}</div>"
            rec_html += f"<div>Profit Target: {recommendation['sell_range']}</div>"
        elif recommendation["action"] == "SELL":
            rec_html += f"<div>Exit Zone: {recommendation['sell_range']}</div>"
            rec_html += f"<div>Re-entry Zone: {recommendation['buy_range']}</div>"
        else:  # HOLD
            rec_html += f"<div>Accumulation Zone: {recommendation['buy_range']}</div>"
            rec_html += f"<div>Distribution Zone: {recommendation['sell_range']}</div>"
            
        rec_html += f"""
            <div style="margin-top:10px">
                <div>Stop-Loss: <strong>${stop_loss:.2f}</strong></div>
                <div>Take-Profit: <strong>${take_profit:.2f}</strong></div>
                <div>Risk-Reward: <strong>1:{risk_reward:.1f}</strong></div>
            </div>
        </div>
        """
        
        st.markdown(rec_html, unsafe_allow_html=True)
        
        # Beginner narrative
        st.markdown("#### Beginner Analysis")
        narrative = generate_narrative(recommendation, stock_data)
        st.markdown(f'<div class="narrative">{narrative}</div>', unsafe_allow_html=True)

    with col2:
        # Price chart with technical indicators
        st.markdown(f"### {symbol} Price Analysis ({timeframe})")
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
        
        # VWAP (for intraday charts)
        if timeframe != "Daily":
            vwap = calculate_vwap(stock_data)
            fig.add_hline(y=vwap, line_dash="dot", line_color="blue",
                        annotation_text=f"VWAP: ${vwap:.2f}",
                        annotation_position="bottom left")
        
        fig.update_layout(
            height=500,
            template="plotly_white",
            showlegend=False,
            xaxis_rangeslider_visible=False
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Technical indicators row
        st.markdown("### Technical Indicators")
        col_ind1, col_ind2 = st.columns(2)
        
        with col_ind1:
            # RSI chart
            rsi_values = []
            if len(stock_data) > 14:
                delta = stock_data['Close'].diff()
                gain = delta.where(delta > 0, 0)
                loss = -delta.where(delta < 0, 0)
                avg_gain = gain.rolling(14).mean()
                avg_loss = loss.rolling(14).mean()
                rs = avg_gain / avg_loss
                rsi_values = 100 - (100 / (1 + rs))
            
            rsi_fig = go.Figure()
            if len(rsi_values) > 0:
                rsi_fig.add_trace(go.Scatter(
                    x=stock_data.index,
                    y=rsi_values,
                    line=dict(color='purple', width=2),
                    name='RSI'
                ))
            rsi_fig.add_hline(y=30, line_dash="dash", line_color="green")
            rsi_fig.add_hline(y=70, line_dash="dash", line_color="red")
            rsi_fig.update_layout(
                height=300,
                title="RSI (14-period)",
                yaxis_range=[0,100],
                template="plotly_white"
            )
            st.plotly_chart(rsi_fig, use_container_width=True)
            
        with col_ind2:
            # Volume chart
            vol_fig = go.Figure()
            if len(stock_data) > 0:
                colors = np.where(stock_data['Close'] > stock_data['Open'], '#4CAF50', '#f44336')
                
                vol_fig.add_trace(go.Bar(
                    x=stock_data.index,
                    y=stock_data['Volume'],
                    marker_color=colors,
                    name='Volume'
                ))
            vol_fig.update_layout(
                height=300,
                title="Volume",
                template="plotly_white",
                showlegend=False
            )
            st.plotly_chart(vol_fig, use_container_width=True)

except Exception as e:
    st.error(f"An error occurred: {str(e)}")
    st.stop()

# STRATEGY EXPLANATION
st.markdown("---")
st.subheader("Trading Strategy Explanation")
st.markdown("""
**Professional Swing Trading Approach:**

1. **Multi-Timeframe Analysis**:
   - Uses 15min to daily charts to identify trade setups
   - Combines short-term entries with daily trend direction

2. **Key Technical Tools**:
   - Volume Weighted Average Price (VWAP) for intraday bias
   - RSI for momentum and overbought/oversold conditions
   - Support/resistance based on Fibonacci + pivot points

3. **Risk-Managed Trading**:
   - Stop-loss based on volatility (ATR) and risk tolerance
   - Take-profit levels aligned with key technical zones
   - Clear 1:2+ risk-reward ratios

4. **Beginner-Friendly Guidance**:
   - Narrative explanations of price action
   - Gap analysis and fill targets
   - Trend identification and trading tips
""")

# DISCLAIMER
st.markdown("---")
st.caption("""
**Disclaimer**: This is for educational purposes only. Past performance is not indicative of future results. 
Always conduct your own research and consider consulting a financial advisor before trading.
""")
