import os
import json
import time
import pandas as pd
import numpy as np
from typing import Dict, Any, List
from sklearn.metrics import classification_report, accuracy_score, f1_score, confusion_matrix
from src.rag_chain import DepressionRAGChain
from src.retrieval import SeverityRetriever

DATA_DIR = r"c:\depression-suicide\data\processed"

def evaluate_rag_pipeline(
    sample_size: int = 20, 
    data_dir: str = DATA_DIR,
    top_k: int = 5
) -> Dict[str, Any]:
    test_path = os.path.join(data_dir, "test.csv")
    if not os.path.exists(test_path):
        raise FileNotFoundError("test.csv not found in data/processed! Run preprocessing.py first.")
        
    print(f"[*] Loading held-out test set from: {test_path}", flush=True)
    test_df = pd.read_csv(test_path)
    
    if len(test_df) > sample_size:
        print(f"[*] Sampling {sample_size} test examples for RAG evaluation...", flush=True)
        eval_df = test_df.groupby('label', group_keys=False).apply(
            lambda x: x.sample(n=min(len(x), max(1, sample_size // 4)), random_state=42)
        ).copy()
    else:
        eval_df = test_df.copy()
        
    print(f"[*] Evaluating RAG chain on {len(eval_df)} test samples...", flush=True)
    rag_chain = DepressionRAGChain()
    
    y_true = []
    y_pred = []
    retrieval_match_counts = []
    risk_flags = []
    latencies = []
    
    for idx, row in eval_df.iterrows():
        true_label = str(row['label']).lower()
        query_text = str(row['text_clean'])
        
        start_t = time.time()
        try:
            result = rag_chain.process_query(query_text, top_k=top_k)
            pred_label = result.get('severity', 'moderate').lower()
            risk_flagged = result.get('risk_language_detected', False)
            
            retrieved = result.get('retrieved_context', [])
            matching_neighbors = sum(1 for doc in retrieved if doc.get('label', '').lower() == true_label)
            retrieval_match_ratio = matching_neighbors / max(1, len(retrieved))
            
        except Exception as e:
            print(f"[!] Error processing eval item {idx}: {e}", flush=True)
            pred_label = "moderate"
            risk_flagged = False
            retrieval_match_ratio = 0.0
            
        elapsed = time.time() - start_t
        
        y_true.append(true_label)
        y_pred.append(pred_label)
        retrieval_match_counts.append(retrieval_match_ratio)
        risk_flags.append(risk_flagged)
        latencies.append(elapsed)
        
        print(f"  [{len(y_true)}/{len(eval_df)}] True: {true_label:<8} | RAG Pred: {pred_label:<8} | Retrieval Precision: {retrieval_match_ratio:.2f} | Risk: {risk_flagged}", flush=True)
        
    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average='macro')
    weighted_f1 = f1_score(y_true, y_pred, average='weighted')
    avg_retrieval_precision = float(np.mean(retrieval_match_counts)) if retrieval_match_counts else 0.0
    avg_latency = float(np.mean(latencies)) if latencies else 0.0
    
    labels = sorted(list(set(y_true + y_pred)))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    report = classification_report(y_true, y_pred, output_dict=True)
    
    baseline_metrics = {}
    baseline_path = os.path.join(data_dir, "baseline_metrics.json")
    if os.path.exists(baseline_path):
        with open(baseline_path, "r", encoding="utf-8") as f:
            baseline_metrics = json.load(f)
            
    eval_results = {
        "num_eval_samples": len(eval_df),
        "rag_accuracy": float(acc),
        "rag_macro_f1": float(macro_f1),
        "rag_weighted_f1": float(weighted_f1),
        "avg_retrieval_label_precision": float(avg_retrieval_precision),
        "avg_latency_seconds": float(avg_latency),
        "confusion_matrix": {
            "labels": labels,
            "matrix": cm.tolist()
        },
        "classification_report": report,
        "comparison_with_baseline": {
            "tfidf_accuracy": baseline_metrics.get("accuracy", None),
            "tfidf_macro_f1": baseline_metrics.get("macro_f1", None)
        }
    }
    
    output_eval_path = os.path.join(data_dir, "rag_evaluation_results.json")
    with open(output_eval_path, "w", encoding="utf-8") as f:
        json.dump(eval_results, f, indent=2)
        
    print("\n" + "="*50, flush=True)
    print("         RAG PIPELINE EVALUATION SUMMARY         ", flush=True)
    print("="*50, flush=True)
    print(f"RAG Accuracy:                   {acc:.4f}", flush=True)
    print(f"RAG Macro-F1:                   {macro_f1:.4f}", flush=True)
    print(f"Avg Retrieval Label Precision:  {avg_retrieval_precision:.4f}", flush=True)
    print(f"Avg Latency per Query:         {avg_latency:.2f}s", flush=True)
    if baseline_metrics:
        print(f"TF-IDF Baseline Accuracy:       {baseline_metrics.get('accuracy', 0):.4f}", flush=True)
        print(f"TF-IDF Baseline Macro-F1:       {baseline_metrics.get('macro_f1', 0):.4f}", flush=True)
    print(f"[+] Detailed report saved to: {output_eval_path}", flush=True)
    
    return eval_results

if __name__ == "__main__":
    evaluate_rag_pipeline(sample_size=12)
