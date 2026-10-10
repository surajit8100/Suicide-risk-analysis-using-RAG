import os
import re
import json
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

RAW_DATASET_PATH = r"C:\Users\Surajit\Downloads\Depression_Severity_Levels_Dataset.csv"
OUTPUT_DIR = r"c:\depression-suicide\data\processed"

# Core stopwords to remove ONLY for TF-IDF baseline (preserving critical sentiment words)
BASELINE_STOPWORDS = {
    "a", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any", "are", 
    "as", "at", "be", "because", "been", "before", "being", "below", "between", "both", "but", 
    "by", "could", "did", "do", "does", "doing", "down", "during", "each", "few", "for", "from", 
    "further", "had", "has", "have", "having", "he", "he'd", "he'll", "he's", "her", "here", 
    "here's", "hers", "herself", "him", "himself", "his", "how", "how's", "i'd", "i'll", "i'm", 
    "i've", "if", "in", "into", "is", "it", "it's", "its", "itself", "let's", "me", "more", 
    "most", "my", "of", "off", "on", "once", "only", "or", "other", "ought", "our", "ours", 
    "ourselves", "out", "over", "own", "same", "she", "she'd", "she'll", "she's", "should", 
    "so", "some", "such", "than", "that", "that's", "the", "their", "theirs", "them", 
    "themselves", "then", "there", "there's", "these", "they", "they'd", "they'll", "they're", 
    "they've", "this", "those", "through", "to", "too", "under", "until", "up", "very", "was", 
    "we", "we'd", "we'll", "we're", "we've", "were", "what", "what's", "when", "when's", "where", 
    "where's", "which", "while", "who", "who's", "whom", "why", "why's", "with", "would", "you", 
    "you'd", "you'll", "you're", "you've", "your", "yours", "yourself", "yourselves"
}
# Sentiment-critical words explicitly kept in both versions: not, no, never, myself, cant, cannot, suicide, kill, self, hurt, pain, die, etc.

def clean_text_light(text: str) -> str:
    """Light text cleaning for transformer embeddings & LLM context."""
    if not isinstance(text, str):
        return ""
    
    # 1. Remove URLs
    text = re.sub(r'https?://\S+|www\.\S+', '[URL]', text)
    # 2. Remove Emails
    text = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', '[EMAIL]', text)
    # 3. Remove Phone numbers
    text = re.sub(r'\b\d{3}[-.\s]?\d{3}[-.\s]?\d{4}\b', '[PHONE]', text)
    # 4. Remove Reddit subreddits and usernames
    text = re.sub(r'r/\w+|u/\w+|@\w+', '[USER]', text)
    # 5. Normalize whitespace and newlines
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def clean_text_stopwords_removed(text: str) -> str:
    """Clean text with stopword removal for TF-IDF baseline."""
    text_clean = clean_text_light(text).lower()
    words = re.findall(r'\b\w+\b', text_clean)
    filtered = [w for w in words if w not in BASELINE_STOPWORDS]
    return " ".join(filtered)

def process_dataset(raw_path: str = RAW_DATASET_PATH, output_dir: str = OUTPUT_DIR, seed: int = 42):
    print(f"[*] Loading raw dataset from: {raw_path}")
    df = pd.read_csv(raw_path)
    print(f"[*] Initial row count: {len(df)}")
    
    # 1. Drop missing text rows
    df = df.dropna(subset=['text']).copy()
    print(f"[*] Row count after dropping nulls: {len(df)}")
    
    # 2. Drop duplicate rows based on text
    df = df.drop_duplicates(subset=['text']).copy()
    print(f"[*] Row count after dropping exact duplicates: {len(df)}")
    
    # 3. Assign unique document IDs
    df['id'] = [f"post_{i}" for i in range(len(df))]
    
    # 4. Clean text versions
    print("[*] Cleaning text (preserving sentiment stopwords for embeddings)...")
    df['text_clean'] = df['text'].apply(clean_text_light)
    df['text_stopwords_removed'] = df['text'].apply(clean_text_stopwords_removed)
    
    # Drop any rows where text_clean is empty after cleaning
    df = df[df['text_clean'].str.len() > 5].copy()
    
    # 5. Stratified train/test split (80/20)
    print("[*] Performing 80/20 stratified train/test split...")
    train_df, test_df = train_test_split(
        df, 
        test_size=0.20, 
        random_state=seed, 
        stratify=df['label']
    )
    
    os.makedirs(output_dir, exist_ok=True)
    train_path = os.path.join(output_dir, "train.csv")
    test_path = os.path.join(output_dir, "test.csv")
    card_path = os.path.join(output_dir, "data_card.json")
    
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)
    print(f"[+] Saved train split ({len(train_df)} rows) to: {train_path}")
    print(f"[+] Saved test split ({len(test_df)} rows) to: {test_path}")
    
    # Data card metadata summary
    data_card = {
        "dataset_name": "Depression Severity Levels Dataset",
        "total_cleaned_rows": len(df),
        "train_rows": len(train_df),
        "test_rows": len(test_df),
        "seed": seed,
        "class_distribution_all": df['label'].value_counts().to_dict(),
        "class_distribution_train": train_df['label'].value_counts().to_dict(),
        "class_distribution_test": test_df['label'].value_counts().to_dict(),
        "avg_text_clean_word_count": float(df['text_clean'].apply(lambda x: len(x.split())).mean())
    }
    
    with open(card_path, "w", encoding="utf-8") as f:
        json.dump(data_card, f, indent=2)
    print(f"[+] Data card saved to: {card_path}")
    return train_df, test_df, data_card

if __name__ == "__main__":
    process_dataset()
