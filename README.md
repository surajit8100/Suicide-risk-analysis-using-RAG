# RAG Pipeline: Depression Severity & Risk-Language Detection

An end-to-end Retrieval-Augmented Generation (RAG) system for evaluating **Depression Severity Levels** (`minimum`, `mild`, `moderate`, `severe`) and detecting self-harm/suicidal risk language from social media (Reddit) text.

Built using **LangChain Groq** (`ChatGroq`), **Chroma DB** (`chromadb`), **Sentence Transformers**, a deterministic **Safety Interceptor Layer**, an **Evaluation Harness**, and a **Streamlit** user interface.

---

## 🌟 Key Features

1. **Preprocessing & Baseline**:
   - Cleans raw Reddit posts by masking URLs, emails, usernames, and `@mentions`.
   - Preserves clinical sentiment stopwords (`not`, `no`, `never`, `myself`).
   - Stratified 80/20 train/test split across 41,873 posts.
   - Evaluates a TF-IDF + Logistic Regression baseline model.

2. **Ingestion & Vector Database**:
   - Converts training posts into LangChain `Document` objects with metadata (`id`, `label`).
   - Embeds text using `sentence-transformers/all-MiniLM-L6-v2`.
   - Persists FAISS/Chroma DB index in `vectorstore/chroma_db`.

3. **Retrieval & Groq LLM Generation**:
   - Top-$k$ similarity retriever with Maximal Marginal Relevance (MMR) support.
   - Powered by `langchain_groq.ChatGroq` (`llama-3.3-70b-versatile`).
   - History-aware conversational retriever supporting multi-turn chat memory.
   - Outputs structured JSON: `severity`, `confidence_note`, `risk_language_detected`, `rationale`, `suggestions`.

4. **Safety & Crisis Interceptor**:
   - Pre- and Post-LLM keyword & regex trigger matcher for explicit self-harm or suicidal intent.
   - Immediately displays 24/7 India Crisis Hotlines (Tele-MANAS `14416`, iCall `9152987821`, Vandrevala `9999666555`, Emergency `112`) whenever risk language is detected.
   - Enforces a non-diagnostic medical disclaimer.

5. **Privacy & Evaluation Harness**:
   - Evaluates test set Accuracy, Macro-F1, Confusion Matrix, and Retrieval Label Precision.
   - Logs session metadata without storing raw user PII text.

---

## 🚀 Getting Started

### 1. Installation

Install Python dependencies:
```bash
pip install -r requirements.txt
```

### 2. Set Up Environment Variables (Optional)
The Groq API key is configured by default, or set your environment variable:
```bash
$env:GROQ_API_KEY="your_groq_api_key_here"
```

---

## ⚙️ Step-by-Step Execution Workflow

### Step 1: Preprocess Data & Split Train/Test
```bash
python -m src.preprocessing
```
*Outputs `data/processed/train.csv`, `data/processed/test.csv`, and `data/processed/data_card.json`.*

### Step 2: Run TF-IDF Baseline Evaluation
```bash
python -m src.baseline
```
*Outputs baseline accuracy, macro-F1, and `data/processed/baseline_metrics.json`.*

### Step 3: Build & Persist Vector Database
```bash
python -m src.ingestion
```
*Indexes ~33,000 training posts into `vectorstore/chroma_db`.*

### Step 4: Run RAG Pipeline Evaluation
```bash
python -m src.evaluation
```
*Evaluates the held-out test set and saves `data/processed/rag_evaluation_results.json`.*

### Step 5: Launch Streamlit Web Application
```bash
streamlit run app.py
```

---

## 📊 Output Schema Example

```json
{
  "severity": "moderate",
  "confidence_note": "High context similarity with historical moderate depression Reddit posts.",
  "risk_language_detected": false,
  "rationale": "The post describes persistent fatigue, feeling overwhelmed by daily tasks, and social isolation over several weeks.",
  "suggestions": [
    "Consider speaking with a qualified therapist or counselor for professional guidance.",
    "Practice sensory grounding exercises like the 5-4-3-2-1 technique.",
    "Maintain small, daily routine milestones and prioritize physical rest."
  ],
  "disclaimer": "IMPORTANT DISCLAIMER: This tool is an automated AI research prototype..."
}
```
