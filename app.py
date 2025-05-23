import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime, timedelta
import traceback
import os
import json
import requests
import base64
import tempfile
from dotenv import load_dotenv

# load env
load_dotenv()

# get api key
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
SITE_URL = os.getenv("SITE_URL", "http://localhost:8501")
SITE_NAME = os.getenv("SITE_NAME", "Stock Analysis Dashboard")

# page config - light theme
st.set_page_config(layout="wide", page_title="AI-Powered Technical Stock Analysis Dashboard")

# title
st.title("AI-Powered Technical Stock Analysis Dashboard")

# sidebar config
with st.sidebar:
    st.header("Configuration")
    
    # add language selection
    st.subheader("Language / 语言")
    language = st.radio(
        "",
        options=["English", "中文"],
        horizontal=True
    )
    
    # set label text
    labels = {
        "stock_selection": "Stock Selection" if language == "English" else "选择股票",
        "chart_settings": "Chart Settings" if language == "English" else "图表设置",
        "indicator_settings": "Indicator Settings" if language == "English" else "指标设置",
        "analysis_settings": "AI Analysis Settings" if language == "English" else "AI分析设置",
        "fetch_data": "Fetch Data" if language == "English" else "获取数据",
        "debug_mode": "Debug Mode" if language == "English" else "调试模式",
        "select_group": "1. Select Stock Group:" if language == "English" else "1. 选择股票组:",
        "select_stocks": "2. Select Specific Stocks from" if language == "English" else "2. 从以下组选择股票:",
        "custom_input": "Enter Custom Tickers (comma-separated):" if language == "English" else "输入自定义股票代码(逗号分隔):",
        "selected": "Selected" if language == "English" else "已选择",
        "please_select": "Please select at least one stock" if language == "English" else "请至少选择一只股票",
        "analyzing": "Analyzing {ticker}..." if language == "English" else "正在进行{ticker}的AI分析...",
        "suggestions": "Suggestions" if language == "English" else "推荐",
        "no_api_key": "OpenRouter API key not configured. Please set OPENROUTER_API_KEY in the .env file." if language == "English" else "OpenRouter API密钥未配置。请在.env文件中设置OPENROUTER_API_KEY。",
        "run_analysis": "Run AI Analysis" if language == "English" else "运行AI分析",
        "recommendation": "Recommendation" if language == "English" else "推荐",
        "justification": "Analysis Justification" if language == "English" else "分析理由",
        "overall_summary": "Overall Summary" if language == "English" else "总体摘要",
        "overall_recommendations": "Overall Structured Recommendations" if language == "English" else "整体推荐结果",
        "run_for_summary": "Run AI Analysis on individual stock tabs to see recommendations summarized here." if language == "English" else "在各股票标签页运行AI分析以在此处查看汇总推荐。"
    }
    
    # modify stock selection logic
    st.subheader(labels["stock_selection"])
    
    # define stock groups and corresponding stocks
    stock_groups = {
        "FAANG": ["META", "AAPL", "AMZN", "NFLX", "GOOG"],
        "Tech Giants": ["AAPL", "MSFT", "GOOG", "AMZN"],
        "Chinese Tech": ["BABA", "BIDU", "JD", "PDD"],
        "EV Leaders": ["TSLA", "NIO", "XPEV", "LI"],
        "Semiconductors": ["NVDA", "AMD", "INTC", "TSM"],
        "Custom": []
    }
    
    # first step: select stock group
    selected_group = st.selectbox(labels["select_group"], list(stock_groups.keys()))
    
    # second step: select specific stocks from the selected group
    available_stocks = stock_groups[selected_group]
    
    if selected_group == "Custom":
        # add more common stocks and their company names for search suggestions
        common_stocks_dict = {
            "AAPL": "Apple Inc.",
            "MSFT": "Microsoft Corporation",
            "GOOG": "Alphabet Inc. (Google)",
            "GOOGL": "Alphabet Inc. (Google)",
            "AMZN": "Amazon.com Inc.",
            "META": "Meta Platforms Inc. (Facebook)",
            "TSLA": "Tesla Inc.",
            "NVDA": "NVIDIA Corporation",
            "JPM": "JPMorgan Chase & Co.",
            "V": "Visa Inc.",
            "JNJ": "Johnson & Johnson",
            "WMT": "Walmart Inc.",
            "MA": "Mastercard Inc.",
            "PG": "Procter & Gamble Co.",
            "UNH": "UnitedHealth Group Inc.",
            "HD": "Home Depot Inc.",
            "BAC": "Bank of America Corp.",
            "BABA": "Alibaba Group Holding Ltd.",
            "DIS": "Walt Disney Co.",
            "NFLX": "Netflix Inc.",
            "PYPL": "PayPal Holdings Inc.",
            "INTC": "Intel Corporation",
            "VZ": "Verizon Communications Inc.",
            "ADBE": "Adobe Inc.",
            "CSCO": "Cisco Systems Inc.",
            "CMCSA": "Comcast Corporation",
            "PFE": "Pfizer Inc.",
            "ORCL": "Oracle Corporation",
            "KO": "Coca-Cola Co.",
            "NKE": "Nike Inc.",
            "PEP": "PepsiCo Inc.",
            "T": "AT&T Inc.",
            "TMUS": "T-Mobile US Inc.",
            "MRK": "Merck & Co. Inc.",
            "AMD": "Advanced Micro Devices Inc.",
            "QCOM": "Qualcomm Inc.",
            "COST": "Costco Wholesale Corporation",
            "ABT": "Abbott Laboratories",
            "AVGO": "Broadcom Inc.",
            "TMO": "Thermo Fisher Scientific Inc.",
            "TXN": "Texas Instruments Inc.",
            "SBUX": "Starbucks Corporation",
            "BIDU": "Baidu Inc.",
            "JD": "JD.com Inc.",
            "PDD": "PDD Holdings Inc. (Pinduoduo)",
            "NIO": "NIO Inc.",
            "XPEV": "XPeng Inc.",
            "LI": "Li Auto Inc.",
            "TSM": "Taiwan Semiconductor Manufacturing Co. Ltd."
        }
        
        # if custom is selected, show input box and provide suggestions
        custom_input = st.text_input(
            labels["custom_input"], 
            value=st.session_state.get('custom_ticker_input', "")
        )
        
        # create search suggestions
        if custom_input:
            # get the last word being input
            parts = custom_input.split(',')
            current_input = parts[-1].strip().upper()
            
            if current_input:
                # match stock code and company name
                suggestions = []
                for ticker, company in common_stocks_dict.items():
                    if current_input in ticker or current_input.lower() in company.lower():
                        suggestions.append(f"{ticker}: {company}")
                
                # show suggestions
                if suggestions:
                    st.markdown(f"### {labels['suggestions']}:")
                    suggestion_cols = st.columns(min(len(suggestions[:5]), 3))
                    for i, suggestion in enumerate(suggestions[:5]):  # limit display number
                        ticker = suggestion.split(':')[0].strip()
                        with suggestion_cols[i % 3]:
                            if st.button(suggestion, key=f"suggest_{ticker}"):
                                # when user clicks suggestion, update input box
                                if len(parts) > 1:
                                    new_input = ','.join(parts[:-1]) + ',' + ticker
                                else:
                                    new_input = ticker
                                # use session state to store new input
                                st.session_state.custom_ticker_input = new_input
                                st.rerun()
        
        tickers = [ticker.strip().upper() for ticker in custom_input.split(",") if ticker.strip()]
    else:
        # otherwise show multi-select box
        selected_stocks = st.multiselect(
            f"{labels['select_stocks']} {selected_group}:",
            available_stocks,
            default=available_stocks[0] if available_stocks else None
        )
        tickers = selected_stocks
    
    # show selected stocks
    if tickers:
        st.success(f"{labels['selected']}: {', '.join(tickers)}")
    else:
        st.warning(labels["please_select"])
    
    # set date range
    end_date_default = datetime.today()
    start_date_default = end_date_default - timedelta(days=365)
    start_date = st.date_input("Start Date", value=start_date_default)
    end_date = st.date_input("End Date", value=end_date_default)
    
    # add chart settings
    st.subheader(labels["chart_settings"])
    chart_height = st.slider("Chart Height", 400, 1000, 600, 50)
    chart_theme = st.selectbox(
        "Chart Theme",
        ["Light", "Dark", "Professional"],
        index=0
    )
    show_volume = st.checkbox("Show Volume Chart", value=False)
    
    # add indicator settings
    st.subheader(labels["indicator_settings"])
    # allow custom technical indicator parameters
    with st.expander("Customize Indicators"):
        sma_period = st.slider("SMA Period", 5, 200, 20, 1)
        ema_period = st.slider("EMA Period", 5, 200, 20, 1)
        bb_period = st.slider("Bollinger Bands Period", 5, 50, 20, 1)
        bb_std = st.slider("Bollinger Bands Standard Deviation", 1.0, 4.0, 2.0, 0.1)
    
    selected_indicators = st.multiselect(
        "Select Indicators:",
        options=[f"{sma_period}-Day SMA", f"{ema_period}-Day EMA", f"{bb_period}-Day Bollinger Bands", "VWAP"],
        default=[f"{sma_period}-Day SMA", f"{ema_period}-Day EMA", f"{bb_period}-Day Bollinger Bands", "VWAP"]
    )
    
    # add AI model selection
    st.subheader(labels["analysis_settings"])
    selected_model = st.selectbox(
        "Select AI Model:",
        ["deepseek/deepseek-r1:free", "qwen/qwen3-235b-a22b:free", "meta-llama/llama-4-maverick:free"],
        index=0
    )
    
    # add a debug mode checkbox
    debug_mode = st.checkbox(labels["debug_mode"], value=False)
    
    fetch_button = st.button(labels["fetch_data"])

# Function to clear session state for stock data
def clear_stock_data_session():
    if "stock_data" in st.session_state:
        del st.session_state["stock_data"]
    if "analysis_results" in st.session_state:
        del st.session_state["analysis_results"]

# define function to clean and prepare data      
def clean_prepare_data(data, ticker_symbol):
    # Handle MultiIndex columns (if any)
    if isinstance(data.columns, pd.MultiIndex):
        # Check if we can extract the first level of column names
        try:
            # Get the first level of the MultiIndex (which should be 'Open', 'High', etc.)
            column_level = 0
            data.columns = data.columns.get_level_values(column_level)
            if debug_mode:
                st.write(f"Converted MultiIndex columns for {ticker_symbol} to single level:")
                st.write(f"New columns: {list(data.columns)}")
        except Exception as e:
            if debug_mode:
                st.error(f"Error converting MultiIndex columns for {ticker_symbol}: {str(e)}")
                st.error(traceback.format_exc())
    
    # Check for essential columns
    expected_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
    
    # Case-insensitive column check
    col_mapping = {}
    for expected in expected_cols:
        found = False
        # First try exact match
        if expected in data.columns:
            col_mapping[expected] = expected
            found = True
        else:
            # Try case-insensitive match
            for col in data.columns:
                if isinstance(col, str) and col.upper() == expected.upper():
                    col_mapping[expected] = col
                    found = True
                    break
    
    missing_cols = [col for col in expected_cols if col not in col_mapping]
    
    if missing_cols:
        if debug_mode:
            st.error(f"Data for {ticker_symbol} is missing essential columns: {', '.join(missing_cols)}.")
            st.write(f"Expected columns: {expected_cols}")
            st.write(f"Actual columns: {list(data.columns)}")
            st.write(f"Column mapping: {col_mapping}")
        return None
    
    # Rename columns to expected names if needed
    if not all(col_mapping[col] == col for col in expected_cols):
        data = data.rename(columns={col_mapping[col]: col for col in expected_cols})
    
    # Data Cleaning and Preparation
    data.dropna(subset=expected_cols, inplace=True)
    data = data[data['Volume'] > 0]

    # Ensure index is DatetimeIndex and clean
    if not isinstance(data.index, pd.DatetimeIndex):
        data.index = pd.to_datetime(data.index, errors='coerce')
        # Drop rows where conversion failed (index became NaT)
        data = data[~data.index.isna()]
    
    # Sort by date
    data.sort_index(inplace=True)
    
    return data if not data.empty else None

# modify create_chart_with_indicators function to support custom parameters
def create_chart_with_indicators(data, ticker, indicators, chart_settings):
    data_for_display = data.copy()
    fig = go.Figure()
    
    # parse chart settings
    height = chart_settings.get("height", 600)
    theme = chart_settings.get("theme", "Light")
    show_volume = chart_settings.get("show_volume", False)
    sma_period = chart_settings.get("sma_period", 20)
    ema_period = chart_settings.get("ema_period", 20)
    bb_period = chart_settings.get("bb_period", 20)
    bb_std = chart_settings.get("bb_std", 2.0)
    
    # set theme colors
    if theme == "Light":
        bg_color = "white"
        grid_color = "LightGray"
        text_color = "black"
    elif theme == "Dark":
        bg_color = "#121212"
        grid_color = "#333333"
        text_color = "white"
    else:  # Professional
        bg_color = "#F5F5F5"
        grid_color = "#E0E0E0"
        text_color = "#333333"
    
    ohlc_cols = ['Open', 'High', 'Low', 'Close']
    all_ohlc_numeric = True
    for col in ohlc_cols:
        data_for_display[col] = pd.to_numeric(data_for_display[col], errors='coerce')
        if data_for_display[col].isnull().all():
            all_ohlc_numeric = False
            if debug_mode:
                st.warning(f"Column '{col}' for {ticker} contains no valid numeric data after conversion.")
            break 
              
    if all_ohlc_numeric:
        data_for_display.dropna(subset=ohlc_cols, inplace=True)
        if not data_for_display.empty:
            fig.add_trace(
                go.Candlestick(
                    x=data_for_display.index,
                    open=data_for_display["Open"],
                    high=data_for_display["High"],
                    low=data_for_display["Low"],
                    close=data_for_display["Close"],
                    name="Candlestick",
                    increasing_line_color="#26A69A",
                    increasing_fillcolor="#26A69A",
                    decreasing_line_color="#EF5350",
                    decreasing_fillcolor="#EF5350"
                )
            )
    
    calculated_cols = []

    # add volume chart
    if show_volume and 'Volume' in data_for_display.columns:
        # create volume chart
        colors = []
        for i in range(len(data_for_display.Close)):
            if i != 0:
                if data_for_display.Close[i] > data_for_display.Close[i-1]:
                    colors.append('#26A69A')  # green for up
                else:
                    colors.append('#EF5350')  # red for down
            else:
                colors.append('#888888')  # gray for the first point
        
        fig.add_trace(
            go.Bar(
                x=data_for_display.index,
                y=data_for_display.Volume,
                name="Volume",
                marker_color=colors,
                opacity=0.7,
                yaxis="y2"
            )
        )

    for indicator in indicators:
        if f"{sma_period}-Day SMA" in indicator and "Close" in data_for_display.columns and not data_for_display["Close"].isnull().all():
            # extract user-defined SMA period
            period = int(indicator.split("-")[0])
            data_for_display[f'SMA_{period}'] = data_for_display["Close"].rolling(window=period, min_periods=1).mean()
            fig.add_trace(go.Scatter(
                x=data_for_display.index, 
                y=data_for_display[f'SMA_{period}'], 
                name=f"SMA ({period})", 
                line=dict(color="#2196F3", width=1.5)
            ))
            calculated_cols.append(f'SMA_{period}')

        elif f"{ema_period}-Day EMA" in indicator and "Close" in data_for_display.columns and not data_for_display["Close"].isnull().all():
            # extract user-defined EMA period
            period = int(indicator.split("-")[0])
            data_for_display[f'EMA_{period}'] = data_for_display["Close"].ewm(span=period, adjust=False, min_periods=1).mean()
            fig.add_trace(go.Scatter(
                x=data_for_display.index, 
                y=data_for_display[f'EMA_{period}'], 
                name=f"EMA ({period})", 
                line=dict(color="#FFC107", width=1.5)
            ))
            calculated_cols.append(f'EMA_{period}')

        elif f"{bb_period}-Day Bollinger Bands" in indicator and "Close" in data_for_display.columns and not data_for_display["Close"].isnull().all():
            # extract user-defined Bollinger Bands parameters
            period = int(indicator.split("-")[0])
            sma_bb = data_for_display["Close"].rolling(window=period, min_periods=1).mean()
            std_bb = data_for_display["Close"].rolling(window=period, min_periods=1).std().fillna(0)
            data_for_display[f'Upper_BB_{period}'] = sma_bb + (std_bb * bb_std)
            data_for_display[f'Lower_BB_{period}'] = sma_bb - (std_bb * bb_std)
            fig.add_trace(go.Scatter(
                x=data_for_display.index, 
                y=data_for_display[f'Upper_BB_{period}'], 
                name=f"BB Upper ({period}, {bb_std}σ)", 
                line=dict(color="#FF5722", width=1.5)
            ))
            fig.add_trace(go.Scatter(
                x=data_for_display.index, 
                y=data_for_display[f'Lower_BB_{period}'], 
                name=f"BB Lower ({period}, {bb_std}σ)", 
                line=dict(color="#9C27B0", width=1.5)
            ))
            calculated_cols.extend([f'Upper_BB_{period}', f'Lower_BB_{period}'])

        elif "VWAP" in indicator and all(col in data_for_display.columns for col in ['High', 'Low', 'Close', 'Volume']):
            is_volume_valid = pd.api.types.is_numeric_dtype(data_for_display['Volume']) and data_for_display['Volume'].gt(0).any()
            are_ohlc_numeric = all(pd.api.types.is_numeric_dtype(data_for_display[col]) for col in ['High', 'Low', 'Close'])

            if is_volume_valid and are_ohlc_numeric:
                data_for_display['Volume'] = pd.to_numeric(data_for_display['Volume'], errors='coerce').fillna(0.00001)
                tp = (data_for_display['High'] + data_for_display['Low'] + data_for_display['Close']) / 3
                volume_for_vwap = data_for_display['Volume'].replace(0, 0.00001)
                data_for_display['VWAP'] = (tp * volume_for_vwap).cumsum() / volume_for_vwap.cumsum()
                fig.add_trace(go.Scatter(
                    x=data_for_display.index, 
                    y=data_for_display['VWAP'], 
                    name="VWAP", 
                    line=dict(color="#4CAF50", width=1.5)
                ))
                calculated_cols.append('VWAP')
    
    if len(fig.data) > 0:
        # set layout
        layout = dict(
            title=f"{ticker} Stock Chart",
            xaxis_title="Date", 
            yaxis_title="Price (USD)", 
            height=height,
            template="plotly_white" if theme == "Light" else "plotly_dark",
            plot_bgcolor=bg_color,
            paper_bgcolor=bg_color,
            xaxis_rangeslider_visible=False,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, font=dict(color=text_color)),
            font=dict(color=text_color)
        )
        
        # if volume is included, add second Y-axis
        if show_volume and 'Volume' in data_for_display.columns:
            layout.update(
                yaxis=dict(
                    domain=[0.3, 1.0],
                    showgrid=True,
                    gridwidth=1,
                    gridcolor=grid_color,
                    tickfont=dict(color=text_color),
                    title_font=dict(color=text_color)
                ),
                yaxis2=dict(
                    domain=[0, 0.2],
                    showgrid=True,
                    gridwidth=1,
                    gridcolor=grid_color,
                    tickfont=dict(color=text_color),
                    title_font=dict(color=text_color),
                    title="Volume"
                )
            )
        else:
            layout.update(
                yaxis=dict(
                    showgrid=True,
                    gridwidth=1,
                    gridcolor=grid_color,
                    tickfont=dict(color=text_color),
                    title_font=dict(color=text_color)
                )
            )
        
        fig.update_layout(layout)
        fig.update_xaxes(
            showgrid=True, 
            gridwidth=1, 
            gridcolor=grid_color, 
            tickfont=dict(color=text_color), 
            title_font=dict(color=text_color)
        )
    
    return fig, data_for_display, calculated_cols

# define AI analysis function
def analyze_stock_with_ai(ticker, fig, model):
    try:
        # save chart as temporary PNG file and read image bytes
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmpfile:
            fig.write_image(tmpfile.name)
            tmpfile_path = tmpfile.name
        with open(tmpfile_path, "rb") as f:
            image_bytes = f.read()
        os.remove(tmpfile_path)

        # encode image as base64
        image_base64 = base64.b64encode(image_bytes).decode('utf-8')

        # analysis prompt
        analysis_prompt = (
            f"You are a Stock Trader specializing in Technical Analysis at a top financial institution. "
            f"Analyze the stock chart for {ticker} based on its candlestick chart and the displayed technical indicators. "
            f"Provide a detailed justification of your analysis, explaining what patterns, signals, and trends you observe. "
            f"Then, based solely on the chart, provide a recommendation from the following options: "
            f"'Strong Buy', 'Buy', 'Weak Buy', 'Hold', 'Weak Sell', 'Sell', or 'Strong Sell'. "
            f"Return your output as a JSON object with two keys: 'action' and 'justification'."
        )

        # prepare content for different models
        content = []
        if "anthropic" in model:
            # Anthropic Claude format
            content = [
                {"type": "text", "text": analysis_prompt},
                {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": image_base64}}
            ]
        elif "openai" in model:
            # OpenAI format
            content = [
                {"type": "text", "text": analysis_prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_base64}"}}
            ]
        elif "gemini" in model:
            # Google Gemini format
            content = [
                {"type": "text", "text": analysis_prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_base64}"}}
            ]
        else:
            # general format
            content = [
                {"type": "text", "text": analysis_prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_base64}"}}
            ]

        # call OpenRouter API    
        response = requests.post(
            url="https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                "HTTP-Referer": SITE_URL,
                "X-Title": SITE_NAME,
                "Content-Type": "application/json"
            },
            json={
                "model": model,
                "messages": [
                    {
                        "role": "user",
                        "content": content
                    }
                ],
                # multi-model compatibility adjustment
                "max_tokens": 1024,  # limit response length
                "temperature": 0.3,  # reduce randomness
                "stream": False
            },
            timeout=120  # increase timeout to 120 seconds
        )
        
        # parse API response   
        response_data = response.json()
        
        if debug_mode:
            st.write("### API Response")
            st.write(response_data)
        
        # fix: correctly handle OpenRouter API response format
        if 'error' in response_data:
            raise Exception(f"API Error: {response_data.get('error', {}).get('message', 'Unknown error')}")
        
        # OpenRouter API puts response in choices[0].message.content
        if 'choices' in response_data and len(response_data['choices']) > 0:
            result_text = response_data['choices'][0]['message']['content']
        else:
            # alternative path: some models may have different response formats
            result_text = response_data.get('response', '')
            if not result_text and debug_mode:
                st.write("无法获取API响应内容。完整响应:")
                st.write(response_data)
        
        # try to parse JSON
        try:
            # find the start and end of the JSON object in the text
            json_start_index = result_text.find('{')
            json_end_index = result_text.rfind('}') + 1  # +1 includes the closing brace
            if json_start_index != -1 and json_end_index > json_start_index:
                json_string = result_text[json_start_index:json_end_index]
                result = json.loads(json_string)
            else:
                # if JSON format is not found, process as plain text
                result = {"action": "Analysis", "justification": result_text}
        except json.JSONDecodeError as e:
            if debug_mode:
                st.error(f"JSON解析错误: {e}")
            # if parsing fails, process as plain text
            result = {"action": "Analysis", "justification": result_text}
        
        return result
    
    except Exception as e:
        error_msg = f"分析过程中发生错误: {str(e)}"
        if debug_mode:
            st.error(error_msg)
            st.error(traceback.format_exc())
        return {"action": "Error", "justification": error_msg}

# fetch data button logic
if fetch_button:
    clear_stock_data_session()
    
    try:
        with st.spinner('Fetching stock data for multiple tickers...'):
            stock_data = {}
            for ticker in tickers:
                # download data
                data = yf.download(ticker, start=start_date, end=end_date, multi_level_index=False)
                
                if data.empty:
                    st.warning(f"No data found for {ticker}.")
                    continue
                
                # clean and prepare data
                cleaned_data = clean_prepare_data(data, ticker)
                if cleaned_data is not None:
                    stock_data[ticker] = cleaned_data
            
            if stock_data:
                st.session_state["stock_data"] = stock_data
                st.success("Stock data loaded successfully for: " + ", ".join(stock_data.keys()))
            else:
                st.error("No valid data found for any of the provided tickers.")
    
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        if debug_mode:
            st.error(traceback.format_exc())
        clear_stock_data_session()

# ensure data is available for analysis
if "stock_data" in st.session_state and st.session_state["stock_data"]:
    
    # create tabs: first tab is overall summary, subsequent tabs are each stock
    tab_names = [labels["overall_summary"]] + list(st.session_state["stock_data"].keys())
    tabs = st.tabs(tab_names)
    
    # store overall results list
    overall_results = []
    
    # process each stock
    for i, ticker in enumerate(st.session_state["stock_data"]):
        data = st.session_state["stock_data"][ticker]
        
        # create chart settings dictionary
        chart_settings = {
            "height": chart_height,
            "theme": chart_theme,
            "show_volume": show_volume,
            "sma_period": sma_period,
            "ema_period": ema_period,
            "bb_period": bb_period,
            "bb_std": bb_std
        }
        
        # create chart and add technical indicators
        fig, data_for_display, calculated_cols = create_chart_with_indicators(
            data, ticker, selected_indicators, chart_settings
        )
        
        # display chart and analysis button in each stock-specific tab
        with tabs[i + 1]:
            st.subheader(f"Analysis for {ticker}")
            
            # display chart
            st.plotly_chart(fig, use_container_width=True)
            
            # display data table
            cols_to_show = ['Open', 'High', 'Low', 'Close', 'Volume'] + calculated_cols
            cols_to_show_existing = [col for col in cols_to_show if col in data_for_display.columns]
            
            if cols_to_show_existing:
                df_display = data_for_display[cols_to_show_existing].copy()
                with st.expander("View Data"):
                    st.dataframe(df_display.astype(float).fillna('').style.format("{:.2f}", na_rep=''))
            
            # AI analysis section
            st.subheader("AI-Powered Analysis")
            
            # check if API key is configured
            if not OPENROUTER_API_KEY or OPENROUTER_API_KEY == "your_openrouter_api_key_here":
                st.error(labels["no_api_key"])
            else:
                analysis_col1, analysis_col2 = st.columns(2)
                
                with analysis_col1:
                    # create unique button key for each tab
                    run_analysis_key = f"run_analysis_{ticker}"
                    run_analysis = st.button(labels["run_analysis"], key=run_analysis_key)
                
                # check if analysis results already exist
                if "analysis_results" not in st.session_state:
                    st.session_state.analysis_results = {}
                
                # if analysis button is clicked or analysis results already exist
                if run_analysis or ticker in st.session_state.analysis_results:
                    
                    # if analysis button is clicked and there is no existing result, perform new analysis
                    if run_analysis and ticker not in st.session_state.analysis_results:
                        with st.spinner(labels["analyzing"].format(ticker=ticker)):
                            result = analyze_stock_with_ai(ticker, fig, selected_model)
                            st.session_state.analysis_results[ticker] = result
                    
                    # get analysis results
                    result = st.session_state.analysis_results.get(ticker, {"action": "N/A", "justification": "No analysis performed yet."})
                    
                    # add to overall results
                    if ticker not in [r["Stock"] for r in overall_results]:
                        overall_results.append({"Stock": ticker, "Recommendation": result.get("action", "N/A")})
                    
                    # display analysis results
                    action = result.get("action", "N/A")
                    justification = result.get("justification", "No justification provided.")
                    
                    action_color = "#4CAF50" if "buy" in action.lower() else "#EF5350" if "sell" in action.lower() else "#2196F3"
                    st.markdown(f"<h3 style='color: {action_color}'>{labels['recommendation']}: {action}</h3>", unsafe_allow_html=True)
                    st.markdown(f"### {labels['justification']}")
                    st.write(justification)
    
    # display table of all results in overall summary tab
    with tabs[0]:
        st.subheader(labels["overall_recommendations"])
        if overall_results:
            df_summary = pd.DataFrame(overall_results)
            st.table(df_summary)
        else:
            st.info(labels["run_for_summary"])

else:
    st.info("Please enter stock tickers and click 'Fetch Data' to display charts.")