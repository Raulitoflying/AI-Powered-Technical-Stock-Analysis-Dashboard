# AI-Powered Technical Stock Analysis Dashboard

## Demo

Try the live demo here: [https://ai-stock-dashboard.streamlit.app/](https://ai-stock-dashboard.streamlit.app/)

## Features

- **Multi-Stock Analysis**: Analyze multiple stocks simultaneously with tabbed interface
- **Stock Group Presets**: Quick selection of common stock groups (FAANG, Tech Giants, etc.)
- **Stock Ticker Suggestions**: Intelligent ticker suggestions as you type
- **Interactive Charts**: Customizable candlestick charts with volume data
- **Customizable Technical Indicators**: Adjustable parameters for SMA, EMA, Bollinger Bands, and VWAP
- **Chart Themes**: Choose between Light, Dark, and Professional themes
- **AI-Powered Analysis**: Technical analysis recommendations using advanced AI models
- **Multi-Language Support**: English and Chinese interfaces
- **Summary View**: Overall recommendations for all analyzed stocks

## Prerequisites

- Python 3.7+
- An OpenRouter API key (sign up at [openrouter.ai](https://openrouter.ai))

## Installation

1. Clone this repository:
```bash
git clone https://github.com/yourusername/AI-Powered-Technical-Stock-Analysis-Dashboard.git
cd AI-Powered-Technical-Stock-Analysis-Dashboard
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up your API key:
   - Create a `.env` file in the project root
   - Add your OpenRouter API key:
   ```
   OPENROUTER_API_KEY=your_api_key_here
   SITE_URL=http://localhost:8501
   SITE_NAME=Stock Analysis Dashboard
   ```

## Usage

1. Run the Streamlit app:
```bash
streamlit run app.py
```

2. In the sidebar:
   - Choose your preferred language (English or Chinese)
   - Select a stock group or enter custom stock tickers
   - Set date range for analysis
   - Customize chart appearance and technical indicators
   - Select AI model for analysis

3. Click "Fetch Data" to load stock information for all selected stocks

4. For each stock tab:
   - View the interactive stock chart with selected indicators
   - Click "Run AI Analysis" to get AI-powered insights and recommendations
   - Examine detailed justification for the AI's recommendation

5. Check the "Overall Summary" tab to see recommendations for all analyzed stocks

## Notes

- The AI analysis requires a valid OpenRouter API key
- For best results with stock ticker suggestions, type at least two characters
- Custom indicators can be configured by expanding the "Customize Indicators" section
