import streamlit as st
import plotly.express as px
import requests
import pandas as pd
import time
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import os
from datetime import datetime, timedelta
from collections import Counter
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline
import torch
from transformers import logging as transformers_logging

# ---------------------- Initial Setup ----------------------
transformers_logging.set_verbosity_error()

# ---------------------- Streamlit Setup ----------------------
st.set_page_config(page_title="Financial News Sentiment Analyzer", layout="centered")

st.markdown(
    """
    <div style="display: flex; align-items: center;">
        <img src="https://cdn-icons-png.flaticon.com/512/1828/1828884.png" width="40" style="margin-right:10px;">
        <h1 style="margin: 0;">Financial News Sentiment Analyzer</h1>
    </div>
    """,
    unsafe_allow_html=True
)
st.markdown("Analyze recent news sentiment for any public company using a fine-tuned FinBERT model.")

# ---------------------- API Keys ----------------------
finnhub_api_key = st.secrets["finnhub_api"]
alpha_api_key = st.secrets["alpha_api"]

# ---------------------- Load FinBERT Model ----------------------
@st.cache_resource(show_spinner=False)
def load_finbert_pipeline():
    try:
        model_name = "kkkkkjjjjjj/results"
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=3,
            ignore_mismatched_sizes=True
        )
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        model = model.to('cpu')
        
        return pipeline(
            "text-classification",
            model=model,
            tokenizer=tokenizer,
            device=-1,
            return_all_scores=False)
    
    except Exception as e:
        st.error(f"Error loading fine-tuned model: {str(e)}")
        st.warning("Falling back to default FinBERT model")
        try:
            return pipeline(
                "text-classification",
                model="ProsusAI/finbert",
                device=-1
            )
        except Exception as fallback_error:
            st.error(f"Failed to load fallback model: {str(fallback_error)}")
            return None


try:
    with st.spinner("Loading sentiment analyzer..."):
        finbert = load_finbert_pipeline()
    if finbert is None:
        st.error("Failed to initialize sentiment analyzer. Please try again later.")
        st.stop()
except Exception as e:
    st.error(f"Critical error initializing model: {str(e)}")
    st.stop()

# ---------------------- Helper Functions ----------------------
def is_valid_ticker(ticker):
    return bool(ticker) and ticker.isalpha() and 1 <= len(ticker) <= 5

# ---------------------- Alpha Vantage Function ----------------------
def fetch_alpha_vantage(symbol, api_key):
    try:
        url = f"https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol={symbol}&apikey={api_key}"
        response = requests.get(url, timeout=10).json()
        prices = response.get("Time Series (Daily)", {})
        if not prices:
            return pd.DataFrame()
            
        df = pd.DataFrame.from_dict(prices, orient="index")
        df.index = pd.to_datetime(df.index)
        df = df.rename(columns={
            '1. open': 'Open',
            '2. high': 'High',
            '3. low': 'Low',
            '4. close': 'Close',
            '5. volume': 'Volume'
        })
        return df
    except Exception as e:
        st.error(f"Error fetching price data: {str(e)}")
        return pd.DataFrame()

# ---------------------- Core Function ----------------------
def get_financial_news_sentiment(symbol: str, date_range_days: int, api_key: str):
    try:
        to_date = datetime.now().strftime('%Y-%m-%d')
        from_date = (datetime.now() - timedelta(days=date_range_days)).strftime('%Y-%m-%d')
        url = f"https://finnhub.io/api/v1/company-news?symbol={symbol}&from={from_date}&to={to_date}&token={api_key}"

        response = requests.get(url, timeout=10)
        if response.status_code != 200:
            st.error(f"API Error {response.status_code}: {response.text}")
            return pd.DataFrame(), {}

        data = response.json()
        if not data:
            return pd.DataFrame(), {}

        articles = [(item.get('headline', ''), item.get('url', '')) for item in data if 'headline' in item]
        filtered_articles = [(h, url) for h, url in articles if h and symbol.lower() in h.lower() and '?' not in h]

        if not filtered_articles:
            return pd.DataFrame(), {}

        headlines_only = [h for h, _ in filtered_articles]
        
        batch_size = 8
        results = []
        for i in range(0, len(headlines_only), batch_size):
            batch = headlines_only[i:i + batch_size]
            try:
                batch_results = finbert(batch)
                results.extend(batch_results)
            except Exception as e:
                st.warning(f"Error processing batch {i//batch_size + 1}: {str(e)}")
                continue

        if not results:
            return pd.DataFrame(), {}

        def map_sentiment_score(label, confidence):
            label = label.lower()
            if label == "positive":
                return round(confidence, 4)
            elif label == "negative":
                return round(-confidence, 4)
            return 0.0

        df = pd.DataFrame({
            "Ticker": symbol.upper(),
            "Headline": headlines_only,
            "Sentiment": [res["label"] for res in results],
            "Sentiment Score": [map_sentiment_score(res["label"], res["score"]) for res in results],
            "URL": [url for _, url in filtered_articles]
        })

        summary = dict(Counter(df["Sentiment"]))
        return df, summary

    except Exception as e:
        st.error(f"Error in sentiment analysis: {str(e)}")
        return pd.DataFrame(), {}

# ---------------------- Tabs Setup ----------------------
tab1, tab2, tab3 = st.tabs([
    "\U0001F50D Analyze One Ticker", 
    "\U0001F4CA Compare Multiple Tickers",
    "\U0001F4DD Enter Custom Headline"  
])

date_map = {
    "Last 7 days": 7,
    "Last 15 days": 15,
    "Last 30 days": 30,
    "Last 60 days": 60
}

# ---------------------- Single Ticker Analysis ----------------------
with tab1:
    col1, col2 = st.columns([2, 1])
    with col1:
        st.markdown("### Enter a company ticker symbol")
        single_ticker_input = st.text_input("", placeholder="e.g., AAPL", key="single_ticker").upper()
    with col2:
        st.markdown("### Date range")
        date_range_single = st.selectbox("Date range",  options=list(date_map.keys()),index=0,key="date_range_single",label_visibility="collapsed")

    analyze_single_btn = st.button("Analyze", type="primary", key="analyze_single")

    if analyze_single_btn and single_ticker_input:
        if not is_valid_ticker(single_ticker_input):
            st.error("Please enter a valid ticker symbol (1-5 letters)")
        elif "," in single_ticker_input:
            st.error("Multiple tickers detected. Please use the **Multiple Ticker** section.")
        else:
            single_ticker = single_ticker_input
            date_range_days = date_map.get(date_range_single, 7)
            with st.spinner("Analyzing sentiment..."):
                df, summary = get_financial_news_sentiment(single_ticker, date_range_days, finnhub_api_key)

            if not df.empty:
                st.subheader("\U0001F4CB Sentiment Analysis Results")
                st.dataframe(df[["Headline", "Sentiment", "Sentiment Score"]], use_container_width=True)

                st.subheader("\U0001F4CA Sentiment Distribution")
                fig, ax = plt.subplots()
                ax.pie(summary.values(), labels=summary.keys(), autopct='%1.1f%%', startangle=140)
                ax.axis("equal")
                st.pyplot(fig)

                st.subheader("\U0001F4C9 Closing Price Trend")
                try:
                    price_df = fetch_alpha_vantage(single_ticker, alpha_api_key)
                    if not price_df.empty:
                        price_df['Close'] = price_df['Close'].astype(float)
                        end_date = datetime.now()
                        start_date = end_date - timedelta(days=date_range_days)
                        price_df = price_df[(price_df.index >= start_date) & (price_df.index <= end_date)]
                        
                        if not price_df.empty:
                            fig = go.Figure()
                            fig.add_trace(go.Scatter(
                                x=price_df.index,
                                y=price_df["Close"],
                                mode='lines+markers',
                                name='Close Price',
                                line=dict(color='royalblue', width=2),
                                marker=dict(size=4)
                            ))
                            fig.update_layout(
                                title=f"{single_ticker} Closing Prices ({date_range_single})",
                                xaxis_title="Date",
                                yaxis_title="Price (USD)",
                                template="plotly_white",
                                hovermode="x unified",
                                legend=dict(x=0, y=1.1, orientation="h"),
                                margin=dict(t=50, b=40)
                            )
                            st.plotly_chart(fig, use_container_width=True)
                        else:
                            st.warning("No price data available for the selected date range.")
                    else:
                        st.warning("\u26A0\uFE0F Could not fetch price data for the selected ticker.")
                except Exception as e:
                    st.error(f"Error generating price chart: {str(e)}")

                with st.expander("\U0001F517 Noteworthy Headlines"):
                    high_threshold = 0.8
                    noteworthy_df = df[df["Sentiment Score"].abs() >= high_threshold]
                    if not noteworthy_df.empty:
                        for _, row in noteworthy_df.iterrows():
                            sentiment_color = "\U0001F7E2" if row["Sentiment"] == "Positive" else "\U0001F534"
                            st.markdown(
                                f"{sentiment_color} **{row['Headline']}**  \n"
                                f"*Sentiment:* `{row['Sentiment']}` | *Score:* `{row['Sentiment Score']}`  \n"
                                f"[Read more →]({row['URL']})"
                            )
                    else:
                        st.info("No particularly strong sentiment headlines found.")

                st.subheader("\U0001F50D Interpretation")
                total = sum(summary.values())
                pos = summary.get("Positive", 0)
                neu = summary.get("Neutral", 0)
                neg = summary.get("Negative", 0)

                st.markdown(f"- Total Headlines: **{total}**")
                st.markdown(f"- Positive: **{pos}** \U0001F7E2")
                st.markdown(f"- Neutral: **{neu}** \U0001F7E1")
                st.markdown(f"- Negative: **{neg}** \U0001F534")

                if pos > neg and pos > neu:
                    st.success(f"**Market Sentiment: Bullish \U0001F7E2**  \nPositive sentiment dominates the recent news around **{single_ticker}**.")
                elif neg > pos and neg > neu:
                    st.error(f"**Market Sentiment: Bearish \U0001F534**  \nNegative sentiment dominates the recent news around **{single_ticker}**.")
                elif neu > pos and neu > neg:
                    st.info(f"**Market Sentiment: Neutral \U0001F7E1**  \nNeutral sentiment dominates the recent news around **{single_ticker}**.")
                else:
                    st.info(f"**Market Sentiment: Mixed ⚖️**  \nNo strong sentiment direction observed for **{single_ticker}**.")
            else:
                st.warning(f"No relevant news found for {single_ticker} in the selected time period.")

# ---------------------- Multiple Ticker Analysis ----------------------
with tab2:
    multi_tickers_input = st.text_input(
        "Enter multiple tickers (comma-separated)",
        placeholder="e.g., AAPL, TSLA, GOOGL",
        key="multi_ticker_input"
    )
    multi_range = st.selectbox("Select date range", options=list(date_map.keys()), index=0, key="multi_date_range")

    compare_btn = st.button("Compare", key="compare_multiple")

    if compare_btn and multi_tickers_input:
        tickers = [t.strip().upper() for t in multi_tickers_input.split(',') if t.strip()]
        valid_tickers = [t for t in tickers if is_valid_ticker(t)]
        
        if not valid_tickers:
            st.error("Please enter at least one valid ticker symbol")
        elif len(valid_tickers) == 1:
            st.error("Single ticker detected. Please enter multiple tickers for comparison.")
        else:
            date_range_days = date_map.get(multi_range, 7)
            all_data = []
            progress_bar = st.progress(0)
            
            with st.spinner("Analyzing sentiment for multiple tickers..."):
                for i, t in enumerate(valid_tickers):
                    progress_bar.progress((i + 1) / len(valid_tickers))
                    df, _ = get_financial_news_sentiment(t, date_range_days, finnhub_api_key)
                    if not df.empty:
                        all_data.append(df)
                    time.sleep(0.5)  # Rate limiting

            progress_bar.empty()
            
            if all_data:
                combined_df = pd.concat(all_data)
                st.subheader("🗞️ News Summary for Multiple Tickers")
                st.dataframe(combined_df[["Ticker", "Headline", "Sentiment"]], use_container_width=True)

                sentiment_counts = combined_df.groupby("Ticker")["Sentiment"].value_counts().unstack().fillna(0)
                sentiment_counts = sentiment_counts.astype(int).reset_index()
                sentiment_counts = pd.melt(sentiment_counts, id_vars="Ticker", var_name="Sentiment", value_name="Count")

                st.subheader("📊 Sentiment Comparison Bar Chart")
                color_map = {"Positive": "green", "Neutral": "blue", "Negative": "red"}
                
                bar_fig = px.bar(
                    sentiment_counts,
                    x="Ticker",
                    y="Count",
                    color="Sentiment",
                    barmode="group",
                    color_discrete_map=color_map
                )
                bar_fig.update_layout(template="plotly_white")
                st.plotly_chart(bar_fig, use_container_width=True)

                # Multi-Ticker Interpretation
                ticker_summary = {}
                for t in valid_tickers:
                    df_t = combined_df[combined_df["Ticker"] == t]
                    if not df_t.empty:
                        counts = dict(Counter(df_t["Sentiment"]))
                        pos = counts.get("Positive", 0)
                        neu = counts.get("Neutral", 0)
                        neg = counts.get("Negative", 0)
                        total = pos + neu + neg
                        score = pos - neg  
                
                        ticker_summary[t] = {
                            "Positive": pos,
                            "Neutral": neu,
                            "Negative": neg,
                            "Total": total,
                            "Score": score
                        }
                
                if ticker_summary:
                    top_bullish = max(ticker_summary.items(), key=lambda x: x[1]["Positive"])[0]
                    top_bearish = max(ticker_summary.items(), key=lambda x: x[1]["Negative"])[0]
                    
                    bullish_count = ticker_summary[top_bullish]["Positive"]
                    bearish_count = ticker_summary[top_bearish]["Negative"]
                    
                    st.subheader("🧠 Strategic Interpretation")
                    st.markdown(f"""
                    - **Top Bullish Sentiment:** **{top_bullish}** with {bullish_count} positive headlines
                    - **Top Bearish Sentiment:** **{top_bearish}** with {bearish_count} negative headlines
                    
                    > 📝 *This analysis is purely for educational purposes and not financial advice.*
                    """)
                    
                    with st.expander("🔍 Detailed Sentiment Breakdown"):
                        st.write("### Sentiment Summary by Ticker")
                        summary_df = pd.DataFrame.from_dict(ticker_summary, orient='index')
                        st.dataframe(summary_df)
                        fig = px.bar(
                            summary_df.reset_index().melt(id_vars='index', value_vars=['Positive', 'Neutral', 'Negative']),
                            x='index',
                            y='value',
                            color='variable',
                            barmode='group',
                            labels={'index': 'Ticker', 'value': 'Count', 'variable': 'Sentiment'},
                            title='Sentiment Comparison Across Tickers'
                        )
                        st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("No valid sentiment data found for the selected tickers.")
                
                st.markdown("*This analysis is purely for educational purposes and not financial advice.*")

# ---------------------- Custom Headline Analysis Tab ----------------------
with tab3:
    st.markdown("## \U0001F4AC Headline Sentiment Tester")
    st.markdown("""
    Enter financial headlines to analyze their sentiment.
    - Headlines should be financial/news related for best results
    - Avoid questions and overly short phrases
    - Minimum 3 words recommended
    """)
    
    user_headlines = st.text_area(
        "Enter headline: ",
        height=150,
        placeholder="Example: Apple reports record profits",
        key="custom_headlines"
    )
    
    if st.button("Analyze", key="analyze_headlines"):
        if user_headlines:
            headlines_list = [h.strip() for h in user_headlines.split('\n') if h.strip() and len(h.split()) >= 3]
            
            if headlines_list:
                with st.spinner("Analyzing..."):
                    try:
                        results = finbert(headlines_list)
                        
                        st.subheader("Results")
                        for headline, result in zip(headlines_list, results):
                            sentiment = result['label']
                            score = result['score']
                            
                            if sentiment == "Positive":
                                icon = "\U0001F7E2"  
                                color = "green"
                            elif sentiment == "Negative":
                                icon = "\U0001F534"  
                                color = "red"
                            else:
                                icon = "\U0001F7E1"  
                                color = "gray"
                            
                            st.markdown(
                                f"{icon} **{headline}**  \n"
                                f"*Sentiment:* `{sentiment}` | *Confidence:* `{score:.2f}`",
                                unsafe_allow_html=True
                            )
                            st.divider()
                    except Exception as e:
                        st.error(f"Error analyzing headlines: {str(e)}")
            else:
                st.warning("Please enter at least one valid headline (minimum 3 words)")
        else:
            st.warning("Please enter some headlines to analyze")

# ---------------------- Footer ----------------------
st.markdown("---")
st.markdown("""
<div style="text-align: center;">
    <p>Financial News Sentiment Analyzer | Powered by FinBERT</p>
    <p><small>Note: This tool is for educational purposes only and not financial advice.</small></p>
</div>
""", unsafe_allow_html=True)
