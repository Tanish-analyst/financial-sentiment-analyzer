# 🚀 Financial News Sentiment Analyzer

An **End-to-End Financial News Sentiment Analyzer** built using a **fine-tuned FinBERT model** for classifying financial headlines into **positive**, **neutral**, or **negative** sentiments.

## 🔍 About the Project

This project leverages the power of **FinBERT** (`yiyanghkust/finbert-tone`) and fine-tunes it on **3,000+ financial headlines** collected via a financial news API. The custom-trained model significantly improves sentiment prediction accuracy from **71%** to **81.91%** on an unseen evaluation dataset.

## 🌐 Interactive Streamlit App

The fine-tuned model is deployed through an intuitive and responsive **Streamlit web application** with the following features:

- 📊 **Single & Multiple Ticker Analysis**  
  Supports sentiment analysis for tickers over **7, 15, 30, and 60-day** ranges.

- ✍️ **Custom Headlines Tab**  
  Real-time sentiment classification for user-inputted headlines.

- 📈 **Data Visualizations & Interpretation**  
  Includes sentiment trend charts, pie charts, and summarizations for better decision-making.

> This tool empowers **traders, analysts, and finance professionals** with **AI-powered sentiment insights** for improved market understanding.

## 🔧 Tech Stack

- Hugging Face Transformers
- PyTorch
- Streamlit
- API Integration (Financial News APIs)

## 📊 Model Performance

**Fine-Tuned Accuracy**: **81.91%**  
*Improved from 71% baseline accuracy on unseen financial news data.*

---
