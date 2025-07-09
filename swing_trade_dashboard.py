import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import pytz

# Configure page settings
st.set_page_config(
    page_title="Professional Trading Dashboard",
    page_icon="📈",
    layout="wide"
)

# Custom styling
st.markdown("""
<style>
    :root {
        --primary-color: #4CAF50;
        --secondary-color: #2196F3;
        --positive-color: #27ae60;
        --negative-color: #e74c3c;
        --warning-color: #e67e22;
    }
    
    .header { 
        font-size: 28px !important; 
        font-weight: bold; 
        padding-bottom: 10px;
        border-bottom: 2px solid var(--primary-color);
        margin-bottom: 20px;
        color: #2c3e50;
    }
    .section-header { 
        font-size: 20px !important; 
        font-weight: bold; 
        background-color: #f0f5ff;
        padding: 12px;
        border-radius: 5px;
        margin-top: 25px;
        margin-bottom: 15px;
        color: #2c3e50;
        border-left: 4px solid var(--secondary-color);
    }
    .metric-label { 
        font-weight: bold; 
        font-size: 15px;
    }
    .metric-value { 
        font-size: 15px;
    }
    .source-row { 
        font-size: 12px; 
        color: #666; 
        margin-top: -10px;
        margin-bottom: 15px;
    }
    .takeaway-box { 
        background-color: #eaf7ff; 
        padding: 15px; 
        border-radius: 5px; 
        border-left: 4px solid #3498db;
        margin: 15px 0;
        font-size: 15px;
    }
    .indicator-row { 
        margin-bottom: 8px;
        font-size: 15px;
    }
    .indicator-name { 
        font-weight: bold; 
        font-size: 15px;
    }
    .trading-idea { 
        background-color: #e6f7e6; 
        padding: 12px; 
        border-radius: 5px; 
        border-left: 4px solid var(--primary-color);
        margin: 10px 0;
        font-size: 15px;
    }
    .news-item { 
        margin-bottom: 8px;
        font-size: 15px;
    }
    .positive { color: var(--positive-color); }
    .negative { color: var(--negative-color); }
    .warning { color: var(--warning-color); }
    .volatility-high { color: var(--warning-color); font-weight: bold; }
    .recommendation { 
        padding: 20px; 
        border-radius: 10px; 
        margin: 15px 0;
        font-size: 16px;
    }
    .buy { 
        background-color: rgba(39, 174, 96, 0.15); 
        border-left: 5px solid var(--positive-color); 
    }
    .sell { 
        background-color: rgba(231, 76, 60, 0.15); 
        border-left: 5px solid var(--negative-color); 
    }
    .hold { 
        background-color: rgba(52, 152, 219, 0.15); 
        border-left: 5px solid var(--secondary-color); 
    }
    .narrative { 
        background-color: #f8f9fa; 
        padding: 20px; 
        border-radius: 10px; 
        border-left: 5px solid #6c757d;
        font-size: 15px;
    }
    .commentary { 
        font-size: 14px;
        color: #555;
        font-style: italic;
        margin-bottom: 10px;
    }
    .extended-hours-container {
        display: flex;
        justify-content: space-between;
        margin-bottom: 20px;
    }
    .extended-hours-box {
        width: 48%;
        padding: 15px;
        border-radius: 8px;
        background-color: #f9f9f9;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .data-point {
        margin-bottom: 8px;
        display: flex;
    }
    .data-label {
        font-weight: bold;
        min-width: 100px;
    }
    .data-value {
        flex-grow: 1;
    }
    .price-change {
        font-weight: bold;
        padding: 2px 5px;
        border-radius: 3px;
    }
</style>
""", unsafe_allow_html=True)

# SIDEBAR CONTROLS
st.sidebar.header("📊 Trading Controls")
symbol = st.sidebar.text_input("Stock Symbol", "AAPL").strip().upper()
timeframe = st.sidebar.selectbox("Chart Timeframe", 
                                ["Daily", "15m", "30m", "1h", "4h"], 
                                index=0)  # Set Daily as default

# TECHNICAL ANALYSIS FUNCTIONS
def calculate_support_resistance(df):
    """Calculate meaningful support and resistance levels"""
    if len(df) < 10:
        return float(df['Close'].min()), float(df['Close'].max())
    
    # Use recent price action (last 20 periods)
    recent_df = df.tail(20)
    high = float(recent_df['High'].max())
    low = float(recent_df['Low'].min())
    
    # Calculate Fibonacci levels
    diff = high - low
    fib_382 = float(high - diff * 0.382)
    fib_618 = float(high - diff * 0.618)
    
    # Calculate pivot points
    last_high = float(df['High'].iloc[-1])
    last_low = float(df['Low'].iloc[-1])
    last_close = float(df['Close'].iloc[-1])
    pivot = float((last_high + last_low + last_close) / 3)
    s1 = float(2 * pivot - last_high)
    r1 = float(2 * pivot - last_low)
    
    # Identify recent support and resistance
    min_close = float(recent_df['Close'].min())
    max_close = float(recent_df['Close'].max())
    
    # Combine indicators to determine key levels
    support_candidates = [low, fib_382, s1, min_close]
    resistance_candidates = [high, fib_618, r1, max_close]
    
    # Filter out unreasonable values
    current_price = last_close
    valid_supports = [x for x in support_candidates if x < current_price * 0.99]
    valid_resistances = [x for x in resistance_candidates if x > current_price * 1.01]
    
    # Use the most significant levels
    support = min(valid_supports) if valid_supports else low
    resistance = max(valid_resistances) if valid_resistances else high
    
    # Final sanity checks
    if support > current_price * 0.95:
        support = min(low, min_close)
    if resistance < current_price * 1.05:
        resistance = max(high, max_close)
    
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

def calculate_daily_pivot_points(symbol):
    """Calculate pivot points using daily data and classic formula"""
    try:
        daily_data = yf.download(symbol, period='5d', interval='1d')
        if len(daily_data) < 2:
            return 0.0, 0.0, 0.0
        
        prev_day = daily_data.iloc[-2]
        high = float(prev_day['High'])
        low = float(prev_day['Low'])
        close = float(prev_day['Close'])
        
        pivot = (high + low + close) / 3
        s1 = (2 * pivot) - high
        r1 = (2 * pivot) - low
        
        return s1, pivot, r1
    except Exception as e:
        st.error(f"Error calculating pivot points: {e}")
        return 0.0, 0.0, 0.0

def generate_recommendation(df, symbol):
    """Generate professional trading recommendation"""
    current_price = float(df['Close'].iloc[-1])
    support, resistance = calculate_support_resistance(df)
    volatility = calculate_volatility(df)
    rsi = calculate_rsi(df)
    
    action = "HOLD"
    reason = "Price between key levels"
    confidence = "Medium"
    
    # Calculate distance to support and resistance
    support_dist = ((current_price - support) / current_price * 100) if current_price != 0 else 0
    resistance_dist = ((resistance - current_price) / current_price * 100) if current_price != 0 else 0
    
    # Determine RSI condition
    rsi_condition = ""
    if rsi > 70:
        rsi_condition = "overbought"
    elif rsi < 30:
        rsi_condition = "oversold"
    
    # Generate recommendation based on price position and RSI
    if support_dist < 5 and current_price > support:
        if rsi_condition == "oversold":
            action = "BUY"
            reason = "Price near strong support with oversold conditions"
            confidence = "High"
        else:
            action = "BUY"
            reason = "Price near strong support"
    elif resistance_dist < 5 and current_price < resistance:
        if rsi_condition == "overbought":
            action = "SELL"
            reason = "Price near strong resistance with overbought conditions"
            confidence = "High"
        else:
            action = "SELL"
            reason = "Price near strong resistance"
    elif rsi_condition == "overbought":
        action = "SELL"
        reason = "Overbought conditions"
        confidence = "Medium"
    elif rsi_condition == "oversold":
        action = "BUY"
        reason = "Oversold conditions"
        confidence = "Medium"
    
    # Special case for strong trends
    if len(df) > 20:
        ma20 = float(df['Close'].tail(20).mean())
        ma50 = float(df['Close'].tail(50).mean())
        
        if current_price > ma20 > ma50 and action == "BUY":
            confidence = "High"
        elif current_price < ma20 < ma50 and action == "SELL":
            confidence = "High"
    
    return {
        "symbol": symbol,
        "current_price": current_price,
        "support": support,
        "resistance": resistance,
        "volatility": volatility,
        "rsi": rsi,
        "action": action,
        "reason": reason,
        "confidence": confidence
    }

def generate_narrative(recommendation, stock_data):
    """Generate beginner-friendly narrative analysis"""
    try:
        current_price = float(recommendation['current_price'])
        support = float(recommendation['support'])
        resistance = float(recommendation['resistance'])
        volatility = float(recommendation['volatility'])
        rsi = float(recommendation['rsi'])
        
        short_term = float(stock_data['Close'].tail(5).mean())
        medium_term = float(stock_data['Close'].tail(20).mean())
        
        trend = "sideways/range-bound"
        if current_price > medium_term and medium_term > short_term:
            trend = "strong uptrend"
        elif current_price > medium_term:
            trend = "moderate uptrend"
        elif current_price < medium_term and medium_term < short_term:
            trend = "strong downtrend"
        elif current_price < medium_term:
            trend = "moderate downtrend"
        
        gap_info = ""
        if len(stock_data) >= 2:
            prev_close = float(stock_data['Close'].iloc[-2])
            gap_percent = ((current_price - prev_close) / prev_close) * 100
            if abs(gap_percent) > 2:
                direction = "up" if gap_percent > 0 else "down"
                gap_info = f" with a significant gap {direction} ({abs(gap_percent):.1f}%)"
        
        narrative = f"""
**Beginner-Friendly Analysis of {symbol}**

- **Overall Trend**: The stock is in a **{trend}**{gap_info}
- **Price Position**: Currently trading at **${current_price:.2f}** between support (${support:.2f}) and resistance (${resistance:.2f})
- **Key Indicators**:
  - RSI: **{rsi:.1f}** ({'overbought' if rsi > 70 else 'oversold' if rsi < 30 else 'neutral'})
  - Volatility: ±${volatility:.2f} from current price
- **Trading Tips**:
  - In **{trend.split()[0]} trends**, trade in the trend direction
  - Place stop-loss orders below support for buy positions
  - Take partial profits near resistance levels
"""
        return narrative
        
    except Exception as e:
        return f"**Error generating narrative:** {str(e)}"

# DATA FUNCTIONS
@st.cache_data
def get_stock_data(symbol, timeframe):
    try:
        period_map = {
            "15m": "7d",
            "30m": "60d",
            "1h": "60d",
            "4h": "120d",  # Increased period for 4h data
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
        
        # Special handling for 4h data if empty
        if timeframe == "4h" and data.empty:
            data = yf.download(
                symbol, 
                period="180d",
                interval="240m",
                progress=False
            )
        
        return data
    except Exception as e:
        st.error(f"Error fetching data: {e}")
        return pd.DataFrame()

# MAIN DASHBOARD
st.markdown(f'<div class="header">📊 Professional Trading Dashboard: {symbol} Analysis</div>', unsafe_allow_html=True)

if not symbol:
    st.warning("Please enter a stock symbol in the sidebar")
    st.stop()

stock_data = get_stock_data(symbol, timeframe)

if stock_data.empty:
    st.error(f"No data found for {symbol} with {timeframe} timeframe")
    st.stop()

if len(stock_data) > 0:
    stock_data = stock_data.dropna()

if len(stock_data) < 2:
    st.error(f"⚠️ Insufficient historical data ({len(stock_data)} bars available). Requires at least 2 bars.")
    st.stop()

# Get accurate pre-market and post-market data
ticker = yf.Ticker(symbol)
ny_tz = pytz.timezone('America/New_York')
today = datetime.now(ny_tz).date()
full_day_data = ticker.history(period="1d", interval="1m", prepost=True)

premarket_session = full_day_data.between_time("04:00", "09:30")
postmarket_session = full_day_data.between_time("16:00", "20:00")

# Pre-market data
premarket_open = None
premarket_close = None
premarket_high = None
premarket_low = None
premarket_volume = None
premarket_change = 0

if not premarket_session.empty:
    premarket_open = float(premarket_session['Open'].iloc[0]) if len(premarket_session) > 0 else None
    premarket_close = float(premarket_session['Close'].iloc[-1]) if len(premarket_session) > 0 else None
    premarket_high = float(premarket_session['High'].max()) if len(premarket_session) > 0 else None
    premarket_low = float(premarket_session['Low'].min()) if len(premarket_session) > 0 else None
    premarket_volume = float(premarket_session['Volume'].sum()) if len(premarket_session) > 0 else None
    if premarket_open and premarket_close:
        premarket_change = ((premarket_close - premarket_open) / premarket_open * 100) if premarket_open != 0 else 0

# Post-market data
postmarket_open = None
postmarket_close = None
postmarket_high = None
postmarket_low = None
postmarket_volume = None
postmarket_change = 0

if not postmarket_session.empty:
    postmarket_open = float(postmarket_session['Open'].iloc[0]) if len(postmarket_session) > 0 else None
    postmarket_close = float(postmarket_session['Close'].iloc[-1]) if len(postmarket_session) > 0 else None
    postmarket_high = float(postmarket_session['High'].max()) if len(postmarket_session) > 0 else None
    postmarket_low = float(postmarket_session['Low'].min()) if len(postmarket_session) > 0 else None
    postmarket_volume = float(postmarket_session['Volume'].sum()) if len(postmarket_session) > 0 else None
    if postmarket_open and postmarket_close:
        postmarket_change = ((postmarket_close - postmarket_open) / postmarket_open * 100) if postmarket_open != 0 else 0

# Calculate metrics
current_price = float(stock_data['Close'].iloc[-1]) if len(stock_data) > 0 else 0
prev_close = float(stock_data['Close'].iloc[-2]) if len(stock_data) > 1 else current_price
daily_change = ((current_price - prev_close) / prev_close) * 100 if prev_close != 0 else 0
support, resistance = calculate_support_resistance(stock_data)
volatility = calculate_volatility(stock_data)
vwap = calculate_vwap(stock_data)
s1, pivot, r1 = calculate_daily_pivot_points(symbol)
rsi = calculate_rsi(stock_data)

# Volume metrics
if len(stock_data) > 63:
    vol_avg_3mo = float(stock_data['Volume'].tail(63).mean())
elif len(stock_data) > 0:
    vol_avg_3mo = float(stock_data['Volume'].mean())
else:
    vol_avg_3mo = 0

# Gap filling analysis
gap_fill_commentary = ""
if len(stock_data) >= 2:
    prev_close_val = float(stock_data['Close'].iloc[-2])
    current_open = float(stock_data['Open'].iloc[-1])
    gap_percent = ((current_open - prev_close_val) / prev_close_val * 100)
    
    if abs(gap_percent) > 1:  # Significant gap
        gap_direction = "up" if gap_percent > 0 else "down"
        
        # Check if gap has been filled
        low_since_open = float(stock_data['Low'].iloc[-1:].min())
        high_since_open = float(stock_data['High'].iloc[-1:].max())
        
        gap_filled = False
        if gap_percent > 0 and low_since_open <= prev_close_val:
            gap_filled = True
        elif gap_percent < 0 and high_since_open >= prev_close_val:
            gap_filled = True
            
        gap_fill_commentary = f"<strong>{abs(gap_percent):.1f}% gap {gap_direction}</strong> "
        gap_fill_commentary += f"({'filled' if gap_filled else 'not filled'})"
        
        # Trading implications
        if gap_filled:
            gap_fill_commentary += " - Gap fill complete, trend continuation likely"
        else:
            if gap_percent > 0:
                gap_fill_commentary += f" - Support at ${prev_close_val:.2f} (gap fill level)"
            else:
                gap_fill_commentary += f" - Resistance at ${prev_close_val:.2f} (gap fill level)"

# SECTION 1: EXTENDED HOURS PERFORMANCE
st.markdown('<div class="section-header">🌅 Extended Hours Performance</div>', unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    st.markdown("### Pre-Market Trading")
    if premarket_open:
        pm_color = "positive" if premarket_change > 0 else "negative" if premarket_change < 0 else ""
        st.markdown(f"<div class='data-point'><div class='data-label'>Open:</div><div class='data-value'>${premarket_open:.2f}</div></div>", unsafe_allow_html=True)
        st.markdown(f"<div class='data-point'><div class='data-label'>Close:</div><div class='data-value'>${premarket_close:.2f} <span class='price-change {pm_color}'>({premarket_change:+.2f}%)</span></div></div>", unsafe_allow_html=True)
        st.markdown(f"<div class='data-point'><div class='data-label'>High:</div><div class='data-value'>${premarket_high:.2f}</div></div>", unsafe_allow_html=True)
        st.markdown(f"<div class='data-point'><div class='data-label'>Low:</div><div class='data-value'>${premarket_low:.2f}</div></div>", unsafe_allow_html=True)
        st.markdown(f"<div class='data-point'><div class='data-label'>Volume:</div><div class='data-value'>{premarket_volume/1e6:.2f}M</div></div>", unsafe_allow_html=True)
        
        # Premarket commentary
        if abs(premarket_change) > 1.5:
            pm_comment = "Strong momentum heading into regular session"
        elif premarket_volume > vol_avg_3mo * 1.5:
            pm_comment = "High premarket volume indicates significant interest"
        else:
            pm_comment = "Normal premarket activity"
        st.markdown(f"<div class='commentary'>{pm_comment}</div>", unsafe_allow_html=True)
    else:
        st.markdown("<div class='commentary'>No pre-market data available</div>", unsafe_allow_html=True)

with col2:
    st.markdown("### Post-Market Trading")
    if postmarket_open:
        pom_color = "positive" if postmarket_change > 0 else "negative" if postmarket_change < 0 else ""
        st.markdown(f"<div class='data-point'><div class='data-label'>Open:</div><div class='data-value'>${postmarket_open:.2f}</div></div>", unsafe_allow_html=True)
        st.markdown(f"<div class='data-point'><div class='data-label'>Close:</div><div class='data-value'>${postmarket_close:.2f} <span class='price-change {pom_color}'>({postmarket_change:+.2f}%)</span></div></div>", unsafe_allow_html=True)
        st.markdown(f"<div class='data-point'><div class='data-label'>High:</div><div class='data-value'>${postmarket_high:.2f}</div></div>", unsafe_allow_html=True)
        st.markdown(f"<div class='data-point'><div class='data-label'>Low:</div><div class='data-value'>${postmarket_low:.2f}</div></div>", unsafe_allow_html=True)
        st.markdown(f"<div class='data-point'><div class='data-label'>Volume:</div><div class='data-value'>{postmarket_volume/1e6:.2f}M</div></div>", unsafe_allow_html=True)
        
        # Postmarket commentary
        if abs(postmarket_change) > 2:
            pom_comment = "Significant after-hours movement - watch for next day follow-through"
        elif postmarket_volume > vol_avg_3mo * 1.2:
            pom_comment = "Elevated after-hours volume suggests continued interest"
        else:
            pom_comment = "Typical postmarket activity"
        st.markdown(f"<div class='commentary'>{pom_comment}</div>", unsafe_allow_html=True)
    else:
        st.markdown("<div class='commentary'>No post-market data available</div>", unsafe_allow_html=True)

# SECTION 2: REGULAR SESSION PERFORMANCE
st.markdown('<div class="section-header">💰 Regular Session Performance</div>', unsafe_allow_html=True)

# Calculate intraday metrics
current_open = float(stock_data['Open'].iloc[-1]) if len(stock_data) > 0 else current_price
intraday_high = current_price
intraday_low = current_price
high_volume = 0
low_volume = 0

if len(stock_data) > 0:
    today_mask = stock_data.index.date == today
    today_data = stock_data[today_mask]
    
    if len(today_data) > 0:
        intraday_high = float(today_data['High'].max())
        intraday_low = float(today_data['Low'].min())
        
        # Find volume at high and low
        high_row = today_data[today_data['High'] == intraday_high]
        low_row = today_data[today_data['Low'] == intraday_low]
        
        if len(high_row) > 0:
            high_volume = float(high_row['Volume'].iloc[0])
        if len(low_row) > 0:
            low_volume = float(low_row['Volume'].iloc[0])

vol_range = ((intraday_high - intraday_low) / intraday_low) * 100 if intraday_low != 0 else 0

# Regular session data with commentary
daily_color = "positive" if daily_change > 0 else "negative" if daily_change < 0 else ""
st.markdown(f"<div class='data-point'><div class='data-label'>Open:</div><div class='data-value'>${current_open:.2f}</div></div>", unsafe_allow_html=True)
st.markdown(f"<div class='data-point'><div class='data-label'>Close:</div><div class='data-value'>${current_price:.2f} <span class='price-change {daily_color}'>({daily_change:+.2f}%)</span></div></div>", unsafe_allow_html=True)
st.markdown(f"<div class='data-point'><div class='data-label'>High:</div><div class='data-value'>${intraday_high:.2f} (Vol: {high_volume/1e6:.2f}M)</div></div>", unsafe_allow_html=True)
st.markdown(f"<div class='data-point'><div class='data-label'>Low:</div><div class='data-value'>${intraday_low:.2f} (Vol: {low_volume/1e6:.2f}M)</div></div>", unsafe_allow_html=True)
st.markdown(f"<div class='data-point'><div class='data-label'>Volatility:</div><div class='data-value'>{vol_range:.1f}%</div></div>", unsafe_allow_html=True)

intraday_comment = ""
if vol_range > 10:
    intraday_comment = "Extreme volatility with wide price swings"
elif vol_range > 5:
    intraday_comment = "Elevated volatility during trading session"
elif high_volume and high_volume > low_volume * 1.5:
    intraday_comment = "Volume concentrated at highs suggests strong buying pressure"
elif low_volume and low_volume > high_volume * 1.5:
    intraday_comment = "Volume concentrated at lows suggests selling pressure"

if intraday_comment:
    st.markdown(f"<div class='commentary'>{intraday_comment}</div>", unsafe_allow_html=True)
    
# Source citations
st.markdown('<div class="source-row">Data Sources: Yahoo Finance • stocktitan.net • investing.com • chartmill.com</div>', unsafe_allow_html=True)

# Key takeaways
takeaway = ""
if premarket_open and premarket_change > 1.5 and daily_change > 0:
    takeaway = "Strong pre-market momentum carried into regular session"
elif premarket_open and premarket_change < -1.5 and daily_change < 0:
    takeaway = "Pre-market weakness continued through regular session"
elif intraday_comment:
    takeaway = intraday_comment
else:
    takeaway = "Typical trading session with normal price action"

st.markdown(f"""
<div class="takeaway-box">
    <strong>Key takeaways:</strong> {takeaway}
</div>
""", unsafe_allow_html=True)

# SECTION 3: TECHNICAL INDICATORS
st.markdown('<div class="section-header">🔍 Technical Indicators</div>', unsafe_allow_html=True)

# Calculate MA status
ma_periods = [5, 10, 20, 50, 100, 200]
ma_bullish = True
ma_values = []
for period in ma_periods:
    if len(stock_data) >= period:
        ma = float(stock_data['Close'].rolling(period).mean().iloc[-1])
        ma_values.append(f"{period}-day: ${ma:.2f}")
        if current_price < ma:
            ma_bullish = False
ma_text = " | ".join(ma_values)

# RSI status
rsi_status = "neutral"
if rsi > 70:
    rsi_status = "overbought"
elif rsi < 30:
    rsi_status = "oversold"

# Display indicators
st.markdown(f"""
<div class="indicator-row">
    <div class="indicator-name">Moving Averages:</div>
    <div>Key MAs are <strong>{'bullish' if ma_bullish else 'mixed'}</strong>; {ma_text}</div>
</div>

<div class="indicator-row">
    <div class="indicator-name">RSI(14):</div>
    <div>~{rsi:.0f} — {rsi_status}</div>
</div>

<div class="indicator-row">
    <div class="indicator-name">Gap Analysis:</div>
    <div>{gap_fill_commentary if gap_fill_commentary else "No significant gap today"}</div>
</div>

<div class="indicator-row">
    <div class="indicator-name">Volatility (ATR):</div>
    <div>${volatility:.2f}</div>
</div>

<div class="indicator-row">
    <div class="indicator-name">VWAP:</div>
    <div>${vwap:.2f}</div>
</div>

<div class="indicator-row">
    <div class="indicator-name">Pivot levels:</div>
    <div>Support ~${s1:.2f}, pivot ~${pivot:.2f}, resistance ~${r1:.2f}</div>
</div>
""", unsafe_allow_html=True)

# Summary
summary_text = f"Indicators are {'bullish' if ma_bullish else 'mixed'}. "
if rsi_status == "overbought":
    summary_text += "RSI suggests caution for overbought conditions."
elif rsi_status == "oversold":
    summary_text += "RSI suggests potential buying opportunity."
else:
    summary_text += "RSI is neutral."

st.markdown(f"""
<div class="takeaway-box">
    <strong>Summary:</strong> {summary_text}
</div>
""", unsafe_allow_html=True)

# SECTION 4: TRADING RECOMMENDATION
st.markdown('<div class="section-header">🚦 Trading Recommendation</div>', unsafe_allow_html=True)

# Generate trading recommendation
recommendation = generate_recommendation(stock_data, symbol)
action_class = recommendation["action"].lower()

# Calculate percentage distances
current_price_val = recommendation['current_price']
support_val = recommendation['support']
resistance_val = recommendation['resistance']

# Calculate support distance percentage
if current_price_val != 0:
    support_dist_pct = ((current_price_val - support_val) / current_price_val) * 100
else:
    support_dist_pct = 0

# Calculate resistance distance percentage
if current_price_val != 0:
    resistance_dist_pct = ((resistance_val - current_price_val) / current_price_val) * 100
else:
    resistance_dist_pct = 0

rec_html = f"""
<div class="recommendation {action_class}">
    <div style="font-size:24px;font-weight:bold;">{recommendation['action']} {symbol}</div>
    <div><strong>Confidence:</strong> {recommendation['confidence']}</div>
    <div><strong>Reason:</strong> {recommendation['reason']}</div>
    <div style="margin-top:15px">
        <div><strong>Current Price:</strong> ${current_price_val:.2f}</div>
        <div><strong>Support:</strong> ${support_val:.2f} ({support_dist_pct:.1f}% below)</div>
        <div><strong>Resistance:</strong> ${resistance_val:.2f} ({resistance_dist_pct:.1f}% above)</div>
        <div><strong>RSI:</strong> {recommendation['rsi']:.1f} ({'Overbought' if recommendation['rsi'] > 70 else 'Oversold' if recommendation['rsi'] < 30 else 'Neutral'})</div>
    </div>
</div>
"""

st.markdown(rec_html, unsafe_allow_html=True)

# SECTION 5: BEGINNER ANALYSIS
st.markdown('<div class="section-header">🧠 Beginner Analysis</div>', unsafe_allow_html=True)

narrative = generate_narrative(recommendation, stock_data)
st.markdown(f'<div class="narrative">{narrative}</div>', unsafe_allow_html=True)

# SECTION 6: BUY/SELL IDEAS
st.markdown('<div class="section-header">💼 Buy/Sell Ideas (Beginner-Friendly)</div>', unsafe_allow_html=True)

# Trading ideas
st.markdown(f"""
<div class="trading-idea">
    <strong>Aggressive entry:</strong> For short swing, consider late-day pullback into ${recommendation['support']-1:.2f}–${recommendation['support']+1:.2f} (support zone).
</div>

<div class="trading-idea">
    <strong>More cautious:</strong> Wait for RSI cool down (below 60) and retest near pivot (${pivot:.2f}).
</div>

<div class="trading-idea">
    <strong>Targets:</strong> Partial profit near ${r1:.2f} (next resistance), with room to ${r1*1.1:.2f}+ on momentum continuation; trail stops tight.
</div>

<div class="trading-idea">
    <strong>Stops:</strong> Below support—break under ${recommendation['support']-2:.2f} invalidates setup.
</div>

<div class="trading-idea">
    <strong>Risk/Reward:</strong> At ${recommendation['support']:.2f} entry, stop at ${recommendation['support']-3:.2f} (risk ~$3), target ${r1:.2f} → ~2.7:1 R:R.
</div>

<div class="trading-idea">
    <strong>Position sizing:</strong> Keep small – this is volatile. Use under 1–2% of capital per swing.
</div>
""", unsafe_allow_html=True)

# DISCLAIMER
st.markdown("---")
st.caption("""
**Disclaimer**: This is for educational purposes only. Past performance is not indicative of future results. 
Pre-market data may be limited or delayed. Always conduct your own research and consider consulting 
a financial advisor before trading.
""")
