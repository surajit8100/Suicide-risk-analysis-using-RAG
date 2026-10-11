import os
import json
import re
from typing import Dict, Any, List
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.chat_message_histories import ChatMessageHistory
from src.retrieval import RetrieverConfig, SimilarityRetriever
from src.safety import SafetyInterceptor

GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "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")
AVAILABLE_GROQ_MODELS = [
    "qwen/qwen3.8-27b",
    "openai/gpt-oss-20b",
    "allam-2-7b"
]

SYSTEM_PROMPT = """You are a compassionate, clinical-research assistant evaluating text for Depression Severity Level and Risk-Language Detection.

You are provided with:
1. Historical Reddit posts retrieved from a dataset that have been ground-truth labeled by clinicians into one of 4 depression severity categories:
   - MINIMUM: Normal emotional fluctuations, mild transient stress or minor sadness.
   - MILD: Noticeable low mood, fatigue, self-doubt, but functional in daily activities.
   - MODERATE: Persistent hopelessness, emotional numbness, social withdrawal, interference with daily functioning.
   - SEVERE: Intense debilitating agony, deep hopelessness, severe dysfunction, or implicit distress.

2. Conversation history (if multi-turn).

3. The User's current input text.

YOUR INSTRUCTIONS:
- Analyze the user text carefully, considering emotional tone, duration, impact on daily life, and similarity to the retrieved reference examples.
- Detect whether explicit or implicit self-harm/suicidal risk language is present.
- Output MUST be a single, valid JSON object with EXACTLY the following keys (do not include markdown block markers outside the JSON):

{{
  "severity": "<minimum | mild | moderate | severe>",
  "confidence_note": "<Short explanation of confidence level based on textual evidence and reference similarity>",
  "risk_language_detected": <true or false>,
  "rationale": "<2-3 sentence analysis connecting user symptoms to the estimated severity level>",
  "suggestions": [
    "<Supportive, non-diagnostic grounding or coping suggestion 1>",
    "<Supportive, non-diagnostic grounding or coping suggestion 2>",
    "<Supportive, non-diagnostic grounding or coping suggestion 3>"
  ]
}}

IMPORTANT SAFETY & COMPLIANCE RULES:
- Never provide a formal medical or psychiatric diagnosis.
- Always remain empathetic, objective, and supportive.
- Do NOT output anything outside the JSON object.

Retrieved Context Examples from Labeled Dataset:
{context}
"""

class DepressionRAGChain:
    """End-to-End Conversational RAG Pipeline for Depression Severity & Risk Detection."""
    
    def __init__(self, api_key: str = GROQ_API_KEY, model_names: List[str] = AVAILABLE_GROQ_MODELS):
        self.api_key = api_key
        self.model_names = model_names
        self.retrieverConfig = RetrieverConfig()
        self.retriever = SimilarityRetriever(config=self.retrieverConfig)
        self.safety = SafetyInterceptor()
        self.history = ChatMessageHistory()
        self.active_llm = None
        self._init_groq_llm()
        
    def _init_groq_llm(self):
        for model in self.model_names:
            try:
                print(f"[*] Initializing ChatGroq LLM ({model})...", flush=True)
                llm = ChatGroq(
                    groq_api_key=self.api_key,
                    model_name=model,
                    temperature=0.1,
                    max_tokens=1000
                )
                # Quick ping test
                llm.invoke("Hi")
                self.active_llm = llm
                self.active_model_name = model
                print(f"[+] Successfully connected to ChatGroq model: {model}", flush=True)
                return
            except Exception as e:
                print(f"[!] Model {model} unavailable: {e}. Trying fallback model...", flush=True)
                
        if self.active_llm is None:
            raise RuntimeError("Could not connect to any specified Groq LLM model.")

    def _parse_llm_json(self, raw_text: str) -> Dict[str, Any]:
        """Extracts and parses JSON object from LLM response text."""
        cleaned = raw_text.strip()
        if "```" in cleaned:
            match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', cleaned, re.DOTALL)
            if match:
                cleaned = match.group(1)
            else:
                cleaned = re.sub(r'```[a-z]*', '', cleaned).strip()
                
        try:
            parsed = json.loads(cleaned)
            valid_severities = ["minimum", "mild", "moderate", "severe"]
            if parsed.get("severity", "").lower() not in valid_severities:
                parsed["severity"] = "moderate"
            parsed["severity"] = parsed["severity"].lower()
            return parsed
        except Exception as e:
            print(f"[!] JSON parsing fallback due to error: {e}", flush=True)
            return {
                "severity": "moderate",
                "confidence_note": "Parsed with safety default fallbacks.",
                "risk_language_detected": False,
                "rationale": f"Analysis based on query features: {raw_text[:150]}...",
                "suggestions": [
                    "Consider speaking with a trusted friend, family member, or mental health professional.",
                    "Practice deep breathing or 5-4-3-2-1 sensory grounding techniques.",
                    "Take things one step at a time and prioritize self-care."
                ]
            }

    def process_query(
        self, 
        user_input: str, 
        top_k: int = 5, 
        use_mmr: bool = False
    ) -> Dict[str, Any]:
        """
        Executes full RAG workflow:
        1. Pre-LLM safety & risk check.
        2. Retrieves top-k similar labeled posts from Chroma DB.
        3. Constructs prompt with context & chat history.
        4. Invokes ChatGroq LLM.
        5. Parses JSON & applies post-LLM safety guardrails.
        6. Updates conversation history.
        """
        pre_risk, triggers = self.safety.check_text_for_risk(user_input)
        
        retrieved_posts = []
        context_str = "No vector DB loaded yet."
        try:
            retrieved_posts = self.retriever.similarity_search(user_input, top_k=top_k)
            context_str = self.retriever.format_context_for_prompt(retrieved_posts)
        except Exception as e:
            print(f"[!] Vector retrieval note: {e}", flush=True)
            context_str = "Reference examples unavailable for this query."

        prompt_template = ChatPromptTemplate.from_messages([
            ("system", SYSTEM_PROMPT),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}")
        ])
        
        messages = prompt_template.format_messages(
            context=context_str,
            chat_history=self.history.messages,
            input=user_input
        )
        
        response_msg = self.active_llm.invoke(messages)
        raw_output = response_msg.content
        
        raw_json = self._parse_llm_json(raw_output)
        
        final_output = self.safety.apply_safety_guardrails(user_input, raw_json)
        final_output["retrieved_context"] = retrieved_posts
        
        self.history.add_user_message(user_input)
        self.history.add_ai_message(f"Severity: {final_output['severity']} | Risk: {final_output['risk_language_detected']}")
        
        return final_output

if __name__ == "__main__":
    chain = DepressionRAGChain()
    sample = "I've been feeling extremely down lately. I can't concentrate on work and everything feels pointless."
    print(f"[*] Processing query: '{sample}'", flush=True)
    res = chain.process_query(sample)
    print(json.dumps(res, indent=2))
