import os
import json
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, f1_score, confusion_matrix

DATA_DIR = r"c:\depression-suicide\data\processed"

def run_tfidf_baseline(data_dir: str = DATA_DIR):
    train_path = os.path.join(data_dir, "train.csv")
    test_path = os.path.join(data_dir, "test.csv")
    
    if not os.path.exists(train_path) or not os.path.exists(test_path):
        raise FileNotFoundError("Train/Test datasets not found. Run preprocessing.py first!")
        
    print("[*] Loading train and test CSVs for TF-IDF baseline...")
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    
    X_train_text = train_df['text_stopwords_removed'].fillna('')
    y_train = train_df['label']
    
    X_test_text = test_df['text_stopwords_removed'].fillna('')
    y_test = test_df['label']
    
    print("[*] Vectorizing text with TF-IDF (max_features=10000, ngrams=(1,2))...")
    vectorizer = TfidfVectorizer(max_features=10000, ngram_range=(1, 2))
    X_train = vectorizer.fit_transform(X_train_text)
    X_test = vectorizer.transform(X_test_text)
    
    print("[*] Training Logistic Regression baseline classifier...")
    clf = LogisticRegression(max_iter=1000, random_state=42, C=1.0)
    clf.fit(X_train, y_train)
    
    y_pred = clf.predict(X_test)
    
    acc = accuracy_score(y_test, y_pred)
    macro_f1 = f1_score(y_test, y_pred, average='macro')
    weighted_f1 = f1_score(y_test, y_pred, average='weighted')
    labels = sorted(y_train.unique().tolist())
    cm = confusion_matrix(y_test, y_pred, labels=labels)
    report = classification_report(y_test, y_pred, output_dict=True)
    
    print("\n" + "="*50)
    print("      TF-IDF + LOGISTIC REGRESSION BASELINE RESULTS      ")
    print("="*50)
    print(f"Accuracy:    {acc:.4f}")
    print(f"Macro-F1:    {macro_f1:.4f}")
    print(f"Weighted F1: {weighted_f1:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    results = {
        "model": "TF-IDF + LogisticRegression",
        "accuracy": float(acc),
        "macro_f1": float(macro_f1),
        "weighted_f1": float(weighted_f1),
        "confusion_matrix": {
            "labels": labels,
            "matrix": cm.tolist()
        },
        "classification_report": report
    }
    
    output_metrics_path = os.path.join(data_dir, "baseline_metrics.json")
    with open(output_metrics_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"[+] Saved baseline metrics to: {output_metrics_path}")
    return results, vectorizer, clf

if __name__ == "__main__":
    run_tfidf_baseline()
