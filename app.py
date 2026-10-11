import os
import json
import time
import datetime
import streamlit as st
from src.safety import CRISIS_RESOURCES_INDIA, STANDARD_DISCLAIMER

# Page Configuration
st.set_page_config(
    page_title="Depression Severity & Risk Detection RAG",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS styling for clinical research look and crisp badges
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E293B;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #64748B;
        margin-bottom: 1.5rem;
    }
    .badge-minimum {
        background-color: #DEF7EC;
        color: #03543F;
        padding: 6px 14px;
        border-radius: 20px;
        font-weight: 700;
        font-size: 1.1rem;
    }
    .badge-mild {
        background-color: #FEF08A;
        color: #713F12;
        padding: 6px 14px;
        border-radius: 20px;
        font-weight: 700;
        font-size: 1.1rem;
    }
    .badge-moderate {
        background-color: #FFEDD5;
        color: #9A3412;
        padding: 6px 14px;
        border-radius: 20px;
        font-weight: 700;
        font-size: 1.1rem;
    }
    .badge-severe {
        background-color: #FEE2E2;
        color: #991B1B;
        padding: 6px 14px;
        border-radius: 20px;
        font-weight: 700;
        font-size: 1.1rem;
    }
     /* Crisis Box: Explicit dark text to guarantee readability in Dark Mode */
    .crisis-box {
      background-color: #FFF1F2 !important;
        border-left: 6px solid #DC2626 !important;
        padding: 1.25rem !important;
        border-radius: 8px !important;
        margin-top: 1rem !important;
        margin-bottom: 1rem !important;
        color: #1E293B !important;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.05);
    }
    .crisis-box h3 {
        color: #DC2626 !important;
        font-size: 1.3rem !important;
        font-weight: 700 !important;
        margin-top: 0 !important;
        margin-bottom: 0.5rem !important;
    }
    .crisis-box p {
        color: #881337 !important;
        font-size: 0.98rem !important;
        margin-bottom: 0.75rem !important;
    }
    .crisis-box ul {
        margin: 0 !important;
        padding-left: 1.4rem !important;
    }
    .crisis-box li {
        color: #334155 !important;
        font-size: 0.96rem !important;
        margin-bottom: 0.5rem !important;
        line-height: 1.5 !important;
    }
    .crisis-box li b {
        color: #0F172A !important;
    }
    .crisis-box .helpline-number {
        color: #DC2626 !important;
        font-size: 1.15rem !important;
        font-weight: 700 !important;
        margin: 0 4px !important;
    }
    .crisis-box .helpline-desc {
        color: #64748B !important;
        font-size: 0.88rem !important;
    }
    .disclaimer-box {
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        padding: 0.8rem;
        border-radius: 6px;
        font-size: 0.85rem;
        color: #64748B;
        margin-top: 1.5rem;
    }

</style>
""", unsafe_allow_html=True)

PRIVACY_LOG_PATH = r"c:\depression-suicide\data\privacy_logs.jsonl"

def log_session_privacy_safe(severity: str, risk_flag: bool, query_len: int, latency: float):
    """Privacy-preserving logger: logs metadata without raw user text."""
    try:
        os.makedirs(os.path.dirname(PRIVACY_LOG_PATH), exist_ok=True)
        log_entry = {
            "timestamp": datetime.datetime.now().isoformat(),
            "query_char_length": query_len,
            "predicted_severity": severity,
            "risk_language_detected": risk_flag,
            "latency_seconds": round(latency, 3)
        }
        with open(PRIVACY_LOG_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(log_entry) + "\n")
    except Exception as e:
        pass

# Initialize RAG Chain in Session State
@st.cache_resource(show_spinner="Initializing LangChain Groq RAG Engine...")
def load_rag_chain():
    from src.rag_chain import DepressionRAGChain
    return DepressionRAGChain()

try:
    rag_chain = load_rag_chain()
    rag_loaded = True
except Exception as e:
    rag_loaded = False
    load_error = str(e)

# Sidebar Configuration
with st.sidebar:
    st.image("https://img.icons8.com/illustrations/100/brain.png", width=70)
    st.title("Settings & Metrics")
    
    st.markdown("---")
    st.subheader("Retriever Configuration")
    top_k = st.slider("Top-K Retrieved Examples", min_value=1, max_value=10, value=5)
    use_mmr = st.checkbox("Enable MMR (Diversity)", value=False)
    
    st.markdown("---")
    st.subheader("System Architecture")
    st.markdown("""
    - **LLM**: `ChatGroq` (`llama-3.3-70b-versatile`)
    - **Vector DB**: Chroma DB (`chromadb`)
    - **Embeddings**: `all-MiniLM-L6-v2`
    - **Dataset**: 41,873 Reddit posts
    """)
    
    st.markdown("---")
    if st.button("Clear Conversation History"):
        st.session_state["messages"] = []
        if rag_loaded:
            rag_chain.history.clear()
        st.rerun()

# Main Header
st.markdown('<div class="main-header">Depression Severity & Risk-Language Detection RAG</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Conversational RAG Workflow for Analyzing Reddit Posts with Safety Interceptors & Crisis Guidance</div>', unsafe_allow_html=True)

if not rag_loaded:
    st.error(f"Failed to load RAG Engine: {load_error}")
    st.info("Make sure Chroma vector store is built (`python -m src.ingestion`) and requirements are installed.")
    st.stop()

# Initialize Chat History
if "messages" not in st.session_state:
    st.session_state["messages"] = []

# Display Past Messages
for msg in st.session_state["messages"]:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"], unsafe_allow_html=True)

# Chat Input Box
user_prompt = st.chat_input("Enter post text or feeling description here...")

if user_prompt:
    # Render User Message
    st.session_state["messages"].append({"role": "user", "content": user_prompt})
    with st.chat_message("user"):
        st.markdown(user_prompt)
        
    # Render AI Assistant Response
    with st.chat_message("assistant"):
        with st.spinner("Analyzing text, retrieving reference posts, and checking safety guardrails..."):
            start_t = time.time()
            output = rag_chain.process_query(user_prompt, top_k=top_k)
            latency = time.time() - start_t
            
            severity = output.get("severity", "moderate").lower()
            risk_detected = output.get("risk_language_detected", False)
            rationale = output.get("rationale", "")
            confidence_note = output.get("confidence_note", "")
            suggestions = output.get("suggestions", [])
            crisis = output.get("crisis_resources") or CRISIS_RESOURCES_INDIA
            disclaimer = output.get("disclaimer", "")
            retrieved = output.get("retrieved_context", [])
            
            # Log privacy-safe metrics
            log_session_privacy_safe(severity, risk_detected, len(user_prompt), latency)
            
            # 1. Crisis Banner (If Risk Detected)
            if risk_detected and crisis:
                # st.markdown(f"""
                # <div class="crisis-box">
                #     <h3 style="color:#DC2626; margin-top:0;">🚨 Immediate Crisis Support Resources ({crisis['country']})</h3>
                #     <p><b>If you or someone you know is in distress, immediate help is available 24/7:</b></p>
                #     <ul>
                #         {"".join(f"<li><b>{h['name']}</b>: <span style='font-size:1.1rem; font-weight:bold;'>{h['number']}</span> ({h['availability']})</li>" for h in crisis['helplines'])}
                #     </ul>
                # </div>
                # """, unsafe_allow_html=True)
                helpline_items = "".join(
                     f"<li><b>{h['name']}</b>: <span style='font-size:1.15rem; font-weight:bold; color:#B91C1C;'>{h['number']}</span> ({h['availability']})</li>"
                     for h in crisis.get("helplines", [])
                     )
                st.markdown(f"""
                 <div class="crisis-box">
                    <h3 style="color:#DC2626; margin-top:0;">🚨 Immediate Crisis Support Resources ({crisis.get('country', 'India')})</h3>
                    <p style="color:#7F1D1D;"><b>If you or someone you know is in distress or having thoughts of self-harm, confidential help is available 24/7:</b></p>
                   <ul>
                   {helpline_items}
                   </ul>
                 </div>
                   """, unsafe_allow_html=True)
                
            # 2. Severity Badge & Risk Status
            badge_class = f"badge-{severity}"
            st.markdown(f"""
            <div style="margin-top: 1rem; margin-bottom: 1rem;">
                <span style="font-size: 1rem; color: #475569; margin-right: 10px;">Estimated Severity Level:</span>
                <span class="{badge_class}">{severity.upper()}</span>
                <span style="margin-left: 20px; font-size: 0.95rem; color: {'#DC2626' if risk_detected else '#16A34A'}; font-weight: 600;">
                    {'⚠️ Risk Language Flagged' if risk_detected else '✅ No Explicit Risk Language'}
                </span>
            </div>
            """, unsafe_allow_html=True)
            
            # 3. Rationale & Confidence Note
            st.markdown(f"**Rationale:** {rationale}")
            if confidence_note:
                st.caption(f"ℹ️ **Confidence Note:** {confidence_note}")
                
            # 4. Non-Diagnostic Suggestions
            if suggestions:
                st.markdown("#### 💡 Supportive & Grounding Strategies")
                for sug in suggestions:
                    st.markdown(f"- {sug}")
                    
            # 5. Retrieved Context Expander
            if retrieved:
                with st.expander(f"📚 View {len(retrieved)} Similar Dataset Examples Retrieved from Chroma DB"):
                    for idx, post in enumerate(retrieved, start=1):
                        st.markdown(f"**Example #{idx}** | **True Dataset Label:** `{post['label'].upper()}`")
                        st.text(post.get('text', post.get('content', '')))
                        if post.get('score') is not None:
                            st.caption(f"Distance Score: {post['score']:.4f}")
                        st.markdown("---")
                        
            # 6. Disclaimer Box
            st.markdown(f'<div class="disclaimer-box">{disclaimer}</div>', unsafe_allow_html=True)
            
            # Save assistant message format for display
            assistant_display = f"""
            Estimated Severity: **{severity.upper()}** | Risk Flag: **{risk_detected}**  
            **Rationale:** {rationale}
            """
            st.session_state["messages"].append({"role": "assistant", "content": assistant_display})
