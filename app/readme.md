# 📊 Financial News Sentiment Analyzer

> Analyze recent news sentiment for any public company using a fine-tuned **FinBERT** model with integrated stock price trends and news insights.

---

## 🔍 Overview

This Streamlit web application allows users to:

- Analyze financial news headlines for any **publicly traded company**
- View **sentiment scores** (Positive, Negative, Neutral) using a **fine-tuned FinBERT model**
- Visualize **sentiment distribution** via pie charts
- Track **stock closing price trends** using Alpha Vantage API
- Identify **noteworthy headlines** with strong sentiment
- Compare **multiple tickers** across sentiment metrics
- Run sentiment analysis on **custom user-input headlines**

---

## 🚀 Key Functionalities

### 🔹 Single Ticker Analysis (`Analyze One Ticker`)

- Input a stock ticker symbol (e.g., `AAPL`, `TSLA`)
- Select a date range: `Last 7`, `15`, `30`, or `60 days`
- App fetches recent news using **Finnhub API**
- Headlines filtered for relevance and passed through **FinBERT**
- Outputs:
  - Sentiment-labeled DataFrame with confidence scores
  - **Pie chart** of sentiment distribution
  - **Interactive Plotly chart** of closing stock prices
  - **Expandable list** of notable headlines with high sentiment scores
  - **Auto-generated interpretation** (Bullish, Bearish, Neutral, or Mixed sentiment)

---

### 🔹 Multiple Ticker Comparison (`Compare Multiple Tickers`)

- Input multiple tickers (comma-separated, e.g., `AAPL, MSFT, AMZN`)
- Select a common date range
- For each ticker:
  - News headlines analyzed
  - Sentiment distribution summarized
- **Bar chart** compares sentiment (Positive, Negative, Neutral) across all tickers

---

### 🔹 Custom Headline Analysis (`Enter Custom Headline`)

- Enter any text headline or multiple lines
- Each line is analyzed for sentiment using **FinBERT**
- View:
  - Sentiment label + confidence score
  - Quick insights on how each custom headline might influence market sentiment

---

## ⚙️ Tech Stack

| Component      | Library/API Used                                 |
|----------------|--------------------------------------------------|
| UI Framework   | [Streamlit](https://streamlit.io)                |
| ML Model       | [FinBERT (Fine-tuned)](https://github.com/ProsusAI/finBERT) |
| News API       | [Finnhub](https://finnhub.io)                    |
| Price Data     | [Alpha Vantage](https://www.alphavantage.co)     |
| Visualization  | Plotly, Matplotlib                               |
| NLP Toolkit    | HuggingFace Transformers                         |

---

## 🛡️ Security

- API keys are securely stored using **Streamlit Secrets Management**
- Model is loaded from a local directory (`kkkkkjjjjjj/results`) to reduce dependency on external endpoints

---

