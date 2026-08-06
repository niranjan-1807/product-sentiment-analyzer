# 📊 Product Review Sentiment Analytics Dashboard

A modern, high-speed **Open Aspect-Based Sentiment Analysis (ABSA)** application powered by the Llama 3 model family on **Groq Cloud API** and built entirely in Python using **Streamlit**.

This application is designed to ingest customer reviews from multiple sources (such as uploaded CSV files or scraped directly from Amazon product URLs via Apify), run them through an LLM inference engine, and provide interactive visualizations summarizing the results.

---

## ✨ Features

- **📈 Datasets Overview**: Real-time summary metrics and ground truth distribution diagrams of local datasets.
- **🧪 Sentiment Playground**: Live playground to test reviews with dynamic product feature/aspect extraction (e.g. comfort, sound quality, battery life) alongside custom evaluation instructions.
- **🔗 Live URL Review Scraper**: Ingests shortened (`amzn.in`, `a.co`) or full Amazon product URLs, scrapes reviews in the cloud, cleans formatting issues, and performs bulk sentiment extraction.
- **⚡ Batch Predictor**: Ingest custom CSV datasets, select the review column, set classification rules, run fast inference, and export the annotated file as a CSV.
- **🎯 Accuracy & Evaluation**: Compares LLM classifications against labeled training data, rendering an interactive Plotly confusion matrix and highlighting incorrect classifications.

---

## 🛠️ Technology Stack

- **Frontend**: Streamlit
- **Data Engineering**: Pandas, Requests, Python-dotenv
- **Visualizations**: Plotly Express, Plotly Heatmap
- **Scraper Pipeline**: Apify Client SDK (using `junglee/amazon-reviews-scraper`)
- **LLM Engine**: Groq SDK (`llama-3.3-70b-versatile` Llama-3 architecture model)
- **Async Execution**: Nest-asyncio

---

## 🚀 Getting Started

### 📋 Prerequisites
Ensure you have **Python 3.10 to 3.13** installed.

### 🔧 Installation

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/yourusername/product-sentiment-analyzer.git
   cd product-sentiment-analyzer
   ```

2. **Create a Virtual Environment**:
   ```bash
   python -m venv venv
   # On Windows:
   .\venv\Scripts\activate
   # On Mac/Linux:
   source venv/bin/activate
   ```

3. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Setup Environment Credentials**:
   Copy the example environment template:
   ```bash
   cp .env.example .env
   ```
   Open the `.env` file and insert your API keys:
   - `GROQ_API_KEY`: Get one from [Groq Console](https://console.groq.com/).
   - `APIFY_API_TOKEN`: Get your Personal API token from [Apify Console](https://console.apify.com/account#/integrations). *(Optional: Falls back to mock simulator if missing).*

5. **Place Sample Data**:
   Ensure `train_data.csv` and `test_data.csv` are in the project root directory.

---

## 🖥️ Running the Application

Start the Streamlit dashboard:
```bash
streamlit run app.py
```
The application will automatically launch in your default web browser (typically at `http://localhost:8501`).

---

## 🛡️ License
Distributed under the MIT License. See `LICENSE` for more information.
