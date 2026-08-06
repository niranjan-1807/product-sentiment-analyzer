import pandas as pd
import html
import re
import logging
from typing import Optional

# Setup local logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def clean_text(text: Optional[str]) -> str:
    """
    Cleans raw review text by resolving UTF-8 encoding corruption (mojibake)
    and trimming excessive whitespaces.
    
    Args:
        text (str): The raw string input to clean.
        
    Returns:
        str: The sanitized and cleaned string.
    """
    if not isinstance(text, str) or pd.isna(text):
        return ""
        
    # Decode HTML entities (e.g., &amp; -> &)
    text = html.unescape(text)
    
    # Clean up malformed character artifacts from encoding mismatch
    replacements = {
        "‚Äôm": "'m",
        "‚Äô": "'",
        "‚Äì": "-",
        "Äî": "-",
        "Äô": "'",
        "Ä¶": "..."
    }
    for bad_str, good_str in replacements.items():
        text = text.replace(bad_str, good_str)
        
    # Remove redundant whitespaces
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def preprocess_dataset(file_path: str, is_train: bool = True) -> pd.DataFrame:
    """
    Loads a dataset from a CSV file, cleans the text fields, removes null values,
    and normalizes sentiment labels if working with training data.
    
    Args:
        file_path (str): Path to the CSV dataset.
        is_train (bool): If True, normalize and validate training class labels.
        
    Returns:
        pd.DataFrame: Cleaned Pandas DataFrame.
    """
    logger.info(f"Loading dataset: {file_path}")
    try:
        df = pd.read_csv(file_path)
    except Exception as e:
        logger.error(f"Failed to read CSV at {file_path}: {str(e)}")
        raise e
    
    # 1. Drop rows where reviews.text is null natively to optimize processing
    df = df.dropna(subset=['reviews.text'])
    
    # 2. Clean review text and titles
    df['reviews.text'] = df['reviews.text'].astype(str).apply(clean_text)
    
    if 'reviews.title' in df.columns:
        df['reviews.title'] = df['reviews.title'].fillna("").astype(str).apply(clean_text)
    
    # 3. Drop empty reviews after cleaning
    initial_len = len(df)
    df = df[df['reviews.text'] != ""]
    logger.info(f"Cleaned {file_path}. Dropped {initial_len - len(df)} empty rows.")

    # 4. If training data, normalize the sentiment label casing
    if is_train and 'sentiment' in df.columns:
        df['sentiment'] = df['sentiment'].astype(str).str.strip().str.capitalize()
        # Keep only valid classes: Positive, Neutral, Negative
        df = df[df['sentiment'].isin(['Positive', 'Neutral', 'Negative'])]
        
    return df

if __name__ == "__main__":
    # Quick sanity check
    try:
        train_df = preprocess_dataset("train_data.csv", is_train=True)
        test_df = preprocess_dataset("test_data.csv", is_train=False)
        print("\n--- Cleaned Data Stats ---")
        print(f"Train reviews: {len(train_df)}")
        print(f"Test reviews: {len(test_df)}")
        if 'sentiment' in train_df.columns:
            print("Sentiment Distribution:")
            print(train_df['sentiment'].value_counts())
    except FileNotFoundError:
        print("Data files not found in local path. Run from root directory.")