import nest_asyncio
nest_asyncio.apply()

import streamlit as st
import pandas as pd
import os
import html
import time
import plotly.express as px
import plotly.figure_factory as ff
from dotenv import load_dotenv
from cleaner import preprocess_dataset
from analyzer import analyze_review
from scraper import scrape_amazon_reviews

load_dotenv()

# App Page Configuration
st.set_page_config(
    page_title="LLM Sentiment Analyzer",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom styling for premium look (neon glowing sentiment values)
st.markdown("""
    <style>
    .positive-glow {
        color: #2ecc71; text-shadow: 0 0 10px rgba(46, 204, 113, 0.4); font-weight: bold;
    }
    .negative-glow {
        color: #e74c3c; text-shadow: 0 0 10px rgba(231, 76, 60, 0.4); font-weight: bold;
    }
    .neutral-glow {
        color: #f1c40f; text-shadow: 0 0 10px rgba(241, 196, 15, 0.4); font-weight: bold;
    }
    .card {
        padding: 20px; border-radius: 10px; background-color: #1e1e30; border: 1px solid #3b3b5c; margin-bottom: 15px;
    }
    </style>
""", unsafe_allow_html=True)

st.title("📊 Product Review Sentiment Analytics Dashboard")
st.subheader("High-Speed Aspect-Based Sentiment Analysis Powered by Groq")

# Load datasets into cache
@st.cache_data
def load_clean_data():
    train_path = "train_data.csv"
    test_path = "test_data.csv"
    
    train = None
    test = None
    
    if os.path.exists(train_path):
        train = preprocess_dataset(train_path, is_train=True)
    if os.path.exists(test_path):
        test = preprocess_dataset(test_path, is_train=False)
        
    return train, test

train_df, test_df = load_clean_data()

# Check for Groq API key
groq_api_key = os.getenv("GROQ_API_KEY")
if not groq_api_key or groq_api_key == "your_groq_api_key_here":
    st.error("⚠️ GROQ_API_KEY is not set correctly in your `.env` file! Please check it.")
    st.stop()

# Sidebar Navigation
st.sidebar.title("🎛️ Navigation")
menu = st.sidebar.radio("Go to:", [
    "Dashboard Overview", 
    "Interactive Playground",
    "URL Review Analyzer",
    "Batch Predictor (Test Data)", 
    "Accuracy & Evaluation"
])

st.sidebar.markdown("---")
st.sidebar.subheader("⚙️ Model Configuration")
available_models = ["openai/gpt-oss-120b", "qwen/qwen3.8-27b", "openai/gpt-oss-20b"]
configured_model = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")
default_idx = available_models.index(configured_model) if configured_model in available_models else 0
selected_model = st.sidebar.selectbox("Active Groq Model:", available_models, index=default_idx)

# -------------------------------------------------------------
# Tab 1: Dashboard Overview
# -------------------------------------------------------------
if menu == "Dashboard Overview":
    st.header("📈 Datasets Overview")
    if train_df is not None:
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Cleaned Training Rows", len(train_df))
        with col2:
            st.metric("Test Rows to Predict", len(test_df) if test_df is not None else 0)
        with col3:
            st.metric("Total Brands", train_df['brand'].nunique() if 'brand' in train_df.columns else 1)
        
        st.write("---")
        # Distribution Chart
        fig = px.pie(
            train_df, 
            names='sentiment', 
            title='Ground Truth Sentiment Distribution (Training Set)',
            color='sentiment',
            color_discrete_map={'Positive': '#2ecc71', 'Neutral': '#f1c40f', 'Negative': '#e74c3c'}
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Could not load train_data.csv in this directory.")

# -------------------------------------------------------------
# Tab 2: Interactive Playground
# -------------------------------------------------------------
elif menu == "Interactive Playground":
    st.header("🧪 Sentiment Playground")
    st.write("Paste a custom review and watch the LLM automatically extract features/aspects and analyze sentiments in real-time.")
    
    col_input, col_output = st.columns([2, 2])
    
    with col_input:
        st.subheader("Input Fields")
        review_text = st.text_area("Review Text:", "I bought this tablet and the display is gorgeous, but the battery life is horrible and it died in 30 minutes. Extremely disappointed.", height=150)
        custom_rules = st.text_input("Custom Analysis Rules (Optional):", placeholder="e.g. prioritize screen quality sentiment")
        analyze_btn = st.button("Run Sentiment Analysis", type="primary")

    with col_output:
        st.subheader("LLM Analysis Result")
        if analyze_btn:
            with st.spinner(f"Calling Groq LLM ({selected_model})..."):
                result = analyze_review(review_text, custom_rules, model=selected_model)
                
                if "error" in result:
                    st.error(f"⚠️ API Error encountered: {result['error']}")
                
                sentiment = result.get("sentiment", "Neutral")
                conf = result.get("confidence", 0.0) * 100
                reason = result.get("reasoning", "No reasoning supplied.")
                aspects = result.get("aspects", [])
                
                # Safely escape user/LLM generated values to prevent Cross-Site Scripting (XSS)
                escaped_sentiment = html.escape(str(sentiment))
                escaped_reason = html.escape(str(reason))
                
                # Render Glowing Sentiment
                if escaped_sentiment == "Positive":
                    st.markdown(f"### Final Sentiment: <span class='positive-glow'>Positive ({conf:.1f}%)</span>", unsafe_allow_html=True)
                elif escaped_sentiment == "Negative":
                    st.markdown(f"### Final Sentiment: <span class='negative-glow'>Negative ({conf:.1f}%)</span>", unsafe_allow_html=True)
                else:
                    st.markdown(f"### Final Sentiment: <span class='neutral-glow'>Neutral ({conf:.1f}%)</span>", unsafe_allow_html=True)
                
                # Display Reasoning Card
                st.markdown(f"<div class='card'><strong>Reasoning:</strong><br>{escaped_reason}</div>", unsafe_allow_html=True)
                
                # Display Aspect Breakdown
                st.write("**Extracted Aspects & Sentiment:**")
                if aspects:
                    aspect_df = pd.DataFrame(aspects)
                    if not aspect_df.empty and 'aspect' in aspect_df.columns and 'sentiment' in aspect_df.columns:
                        aspect_df.columns = ["Product Feature", "Sentiment"]
                    st.dataframe(aspect_df, hide_index=True, use_container_width=True)
                else:
                    st.info("No specific features or aspects were detected in the review text.")

# -------------------------------------------------------------
# Tab 3: URL Review Analyzer
# -------------------------------------------------------------
elif menu == "URL Review Analyzer":
    st.header("🔗 Live Product Link Review Analyzer")
    st.write("Paste an Amazon product URL to automatically fetch reviews, extract key features, and evaluate sentiments in bulk.")
    
    product_url = st.text_input("Amazon Product URL:", placeholder="e.g. https://www.amazon.com/dp/B08KTZ825X")
    max_reviews = st.slider("Number of reviews to fetch and analyze:", 5, 50, 10)
    custom_rules = st.text_input("Custom Analysis Rules (Optional):", placeholder="e.g. focus on build quality")
    
    if st.button("Scrape & Analyze Reviews", type="primary"):
        if not product_url:
            st.error("Please enter a valid product URL.")
        else:
            with st.status("Scraping product reviews...", expanded=True) as status_tracker:
                status_tracker.write("🔍 Resolving redirect endpoints for input URL...")
                status_tracker.write("🌐 Establishing secure connection to Apify API nodes...")
                reviews, is_simulated = scrape_amazon_reviews(product_url, max_reviews)
                if is_simulated:
                    status_tracker.update(label="⚠️ Scraping completed (Simulation fallback active)", state="complete")
                else:
                    status_tracker.update(label=f"✅ Real reviews successfully scraped: {len(reviews)} items", state="complete")
            
            if is_simulated:
                st.warning("⚠️ Running in Demo/Simulation mode with mock reviews based on your product URL.")
                
            if reviews:
                st.write("---")
                st.subheader("Processing Sentiment & Aspect Analysis...")
                
                results = []
                all_aspects_data = []
                
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                for idx, r_text in enumerate(reviews):
                    status_text.text(f"Analyzing review {idx+1}/{len(reviews)} ({selected_model})...")
                    res = analyze_review(r_text, custom_rules, model=selected_model)
                    time.sleep(0.1)
                    
                    sentiment = res.get("sentiment", "Neutral")
                    results.append({
                        "Review Text": r_text,
                        "Predicted Sentiment": sentiment,
                        "Confidence": res.get("confidence", 0.5),
                        "Reasoning": res.get("reasoning", "")
                    })
                    
                    # Collect aspects
                    for asp_item in res.get("aspects", []):
                        all_aspects_data.append({
                            "Aspect": asp_item.get("aspect", "").strip().capitalize(),
                            "Sentiment": asp_item.get("sentiment", "Neutral")
                        })
                        
                    progress_bar.progress((idx + 1) / len(reviews))
                    
                status_text.text("Analysis completed!")
                
                results_df = pd.DataFrame(results)
                
                # Display overall analytics
                col_c1, col_c2 = st.columns(2)
                
                with col_c1:
                    st.subheader("Overall Sentiment Breakdown")
                    pie_fig = px.pie(
                        results_df, 
                        names='Predicted Sentiment', 
                        color='Predicted Sentiment',
                        color_discrete_map={'Positive': '#2ecc71', 'Neutral': '#f1c40f', 'Negative': '#e74c3c'}
                    )
                    st.plotly_chart(pie_fig, use_container_width=True)
                    
                with col_c2:
                    st.subheader("Extracted Aspect Sentiments")
                    if all_aspects_data:
                        aspects_df = pd.DataFrame(all_aspects_data)
                        # Group by aspect and sentiment to count occurrences
                        grouped_asp = aspects_df.groupby(["Aspect", "Sentiment"]).size().reset_index(name="Count")
                        # Sort by top mentioned aspects
                        top_aspects = aspects_df["Aspect"].value_counts().head(10).index
                        grouped_asp = grouped_asp[grouped_asp["Aspect"].isin(top_aspects)]
                        
                        bar_fig = px.bar(
                            grouped_asp,
                            x="Aspect",
                            y="Count",
                            color="Sentiment",
                            barmode="group",
                            color_discrete_map={'Positive': '#2ecc71', 'Neutral': '#f1c40f', 'Negative': '#e74c3c'},
                            title="Top 10 Mentioned Features & Sentiments"
                        )
                        st.plotly_chart(bar_fig, use_container_width=True)
                    else:
                        st.info("No detailed aspect sentiments were extracted.")
                
                st.subheader("Detailed Reviews Log")
                st.dataframe(results_df, use_container_width=True)
                
                # Option to download predictions
                csv_data = results_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📥 Download Analyzed Reviews CSV",
                    data=csv_data,
                    file_name="scraped_reviews_analysis.csv",
                    mime="text/csv"
                )
            else:
                st.error("No reviews could be retrieved or simulated for this URL.")

# -------------------------------------------------------------
# Tab 4: Batch Predictor
# -------------------------------------------------------------
elif menu == "Batch Predictor (Test Data)":
    st.header("⚡ Batch Sentiment Prediction")
    st.write("Upload any CSV dataset or use the default test dataset to run batch predictions.")
    
    # File Uploader
    uploaded_file = st.file_uploader("Upload your own CSV dataset (optional):", type=["csv"])
    
    df_to_use = None
    file_source = ""
    
    if uploaded_file is not None:
        try:
            df_to_use = pd.read_csv(uploaded_file)
            file_source = uploaded_file.name
            st.success(f"Successfully loaded {file_source} ({len(df_to_use)} rows)")
        except Exception as e:
            st.error(f"Error loading CSV file: {str(e)}")
    elif test_df is not None:
        df_to_use = test_df
        file_source = "test_data.csv"
        st.info(f"Using default test dataset: {file_source} ({len(df_to_use)} rows)")
        
    if df_to_use is not None:
        # Let user choose the column that contains the reviews
        columns = list(df_to_use.columns)
        default_col_idx = 0
        for col in ['reviews.text', 'review.text', 'review', 'text', 'comment', 'content', 'body']:
            if col in columns:
                default_col_idx = columns.index(col)
                break
                
        review_col = st.selectbox("Select the column containing review text:", columns, index=default_col_idx)
        batch_size = st.slider("Select number of reviews to predict:", 5, min(200, len(df_to_use)), min(20, len(df_to_use)))
        custom_rules = st.text_input("Custom Analysis Rules for Batch (Optional):", placeholder="e.g. focus on product stability")
        
        if st.button("Start Batch Prediction", type="primary"):
            # Check if column values are strings/valid to prevent data types crash
            non_null_rows = df_to_use[df_to_use[review_col].notna()]
            if non_null_rows.empty:
                st.error("❌ The selected column contains no valid review strings.")
                st.stop()
                
            subset = df_to_use.head(batch_size).copy()
            
            # Preprocess the review text column to clean it safely
            from cleaner import clean_text
            subset[review_col] = subset[review_col].astype(str).apply(clean_text)
            
            results = []
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            for index, row in subset.iterrows():
                status_text.text(f"Analyzing review {len(results)+1}/{batch_size} ({selected_model})...")
                res = analyze_review(row[review_col], custom_rules, model=selected_model)
                results.append({
                    "predicted_sentiment": res.get("sentiment", "Neutral"),
                    "confidence": res.get("confidence", 0.5),
                    "reasoning": res.get("reasoning", "")
                })
                progress_bar.progress((len(results)) / batch_size)
                time.sleep(0.1)
                
            status_text.text("Batch analysis completed!")
            
            res_df = pd.DataFrame(results)
            output_df = pd.concat([subset.reset_index(drop=True), res_df], axis=1)
            
            st.write("---")
            st.subheader("Predictions Preview")
            
            preview_cols = [review_col, 'predicted_sentiment', 'confidence', 'reasoning']
            preview_df = output_df[[col for col in preview_cols if col in output_df.columns]]
            st.dataframe(preview_df, use_container_width=True)
            
            csv_data = output_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Download Predictions CSV",
                data=csv_data,
                file_name=f"predicted_{file_source}",
                mime="text/csv"
            )
    else:
        st.warning("Please upload a CSV file or ensure test_data.csv is placed in the project directory.")

# -------------------------------------------------------------
# Tab 5: Accuracy & Evaluation
# -------------------------------------------------------------
elif menu == "Accuracy & Evaluation":
    st.header("📈 Model Evaluation")
    st.write("Compare the LLM's predictions against the ground truth labels in your training dataset.")
    
    if train_df is not None:
        eval_size = st.slider("Select evaluation sample size from training data:", 10, 100, 30)
        
        if st.button("Run Evaluation", type="primary"):
            eval_subset = train_df.sample(n=eval_size, random_state=42).copy()
            
            preds = []
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            error_count = 0
            for index, row in eval_subset.iterrows():
                status_text.text(f"Evaluating review {len(preds)+1}/{eval_size} ({selected_model})...")
                res = analyze_review(row['reviews.text'], model=selected_model)
                if "error" in res:
                    error_count += 1
                pred_sentiment = res.get("sentiment", "Neutral")
                preds.append(pred_sentiment)
                progress_bar.progress((len(preds)) / eval_size)
                time.sleep(0.15)
                
            status_text.text("Evaluation completed!")
            eval_subset['predicted'] = preds
            
            if error_count > 0:
                st.warning(f"⚠️ {error_count} reviews encountered API errors or rate limit issues during evaluation.")
            
            # Normalize casing for accurate comparison
            eval_subset['sentiment_norm'] = eval_subset['sentiment'].astype(str).str.strip().str.capitalize()
            eval_subset['predicted_norm'] = eval_subset['predicted'].astype(str).str.strip().str.capitalize()
            
            # Calculate metrics
            correct = (eval_subset['sentiment_norm'] == eval_subset['predicted_norm']).sum()
            accuracy = correct / eval_size
            
            st.write("---")
            st.subheader("Results summary")
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Total Sample Size", eval_size)
            with col2:
                st.metric("Model Accuracy", f"{accuracy * 100:.1f}%")
            
            # Generate confusion matrix
            st.subheader("Confusion Matrix")
            labels = ["Positive", "Neutral", "Negative"]
            matrix = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
            
            for index, row in eval_subset.iterrows():
                try:
                    true_val = row['sentiment_norm']
                    pred_val = row['predicted_norm']
                    if true_val in labels and pred_val in labels:
                        true_idx = labels.index(true_val)
                        pred_idx = labels.index(pred_val)
                        matrix[true_idx][pred_idx] += 1
                except Exception:
                    pass
            
            fig = ff.create_annotated_heatmap(
                z=matrix, 
                x=labels, 
                y=labels, 
                colorscale='Viridis',
                annotation_text=[[str(val) for val in row] for row in matrix]
            )
            fig.update_layout(xaxis_title="Predicted", yaxis_title="Actual")
            st.plotly_chart(fig, use_container_width=True)
            
            st.subheader("Incorrect Predictions Log")
            errors = eval_subset[eval_subset['sentiment_norm'] != eval_subset['predicted_norm']]
            st.dataframe(errors[['reviews.text', 'sentiment', 'predicted']], use_container_width=True)
    else:
        st.warning("Could not find train_data.csv in this directory.")