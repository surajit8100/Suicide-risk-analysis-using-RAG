import re
from typing import Dict, Any, Tuple, List

# Explicit self-harm & suicide risk trigger keywords/patterns
SELF_HARM_PATTERNS = [
    r'\bkill\s+(myself|me|my\s+life)\b',
    r'\bend\s+(my\s+life|it\s+all|everything)\b',
    r'\bsuici(de|dal)\b',
    r'\bwant\s+to\s+die\b',
    r'\bbetter\s+off\s+dead\b',
    r'\bhang(ing)?\s+(myself|me)\b',
    r'\boverdos(e|ing)\b',
    r'\bslit\s+(my\s+wrists|my\s+throat)\b',
    r'\bjump(ing)?\s+off\b',
    r'\bno\s+reason\s+to\s+live\b',
    r'\bcan\'?t\s+take\s+this\s+anymore\b',
    r'\bgoodbye\s+(world|everyone|cruel\s+world)\b',
    r'\bi\s+don\'?t\s+want\s+to\s+be\s+here\s+anymore\b',
    r'\bself[- ]harm\b',
    r'\bcut(ting)?\s+myself\b'
]

CRISIS_RESOURCES_INDIA = {
    "country": "India",
    "helplines": [
        {
            "name": "Tele-MANAS (Tele Mental Health Assistance and Networking Across States)",
            "number": "14416  or  1800-891-4416",
            "availability": "24/7 Free National Helpline (Toll-Free)"
        },
        {
            "name": "iCall (Tata Institute of Social Sciences)",
            "number": "+91 9152987821",
            "availability": "Mon–Sat, 10:00 AM – 8:00 PM"
        },
        {
            "name": "Vandrevala Foundation Mental Health Helpline",
            "number": "+91 9999666555",
            "availability": "24/7 Free Confidential Support"
        },
        {
            "name": "KIRAN National Helpline",
            "number": "1800-599-0019",
            "availability": "24/7 Multi-lingual Mental Health Support"
        },
        {
            "name": "Emergency Services",
            "number": "112",
            "availability": "National Emergency Number"
        }
    ]
}

STANDARD_DISCLAIMER = (
    "IMPORTANT DISCLAIMER: This tool is an automated AI research prototype intended strictly "
    "for educational evaluation and risk-language detection. It is NOT a diagnostic tool, medical device, "
    "or clinical assessment. If you or someone you know is struggling with mental distress, depression, "
    "or self-harm thoughts, please reach out to qualified healthcare professionals or call a crisis hotline immediately."
)

class SafetyInterceptor:
    """Pre- and Post-LLM Safety Interceptor for self-harm and crisis language detection."""
    
    def __init__(self):
        self.compiled_patterns = [re.compile(pattern, re.IGNORECASE) for pattern in SELF_HARM_PATTERNS]

    def check_text_for_risk(self, text: str) -> Tuple[bool, List[str]]:
        """Scans input text for explicit self-harm or suicidal intent keywords."""
        if not text or not isinstance(text, str):
            return False, []
            
        matched_triggers = []
        for pattern in self.compiled_patterns:
            matches = pattern.findall(text)
            if matches:
                matched_triggers.append(pattern.pattern)
                
        is_risk = len(matched_triggers) > 0
        return is_risk, matched_triggers

    def apply_safety_guardrails(self, user_input: str, llm_response: Dict[str, Any]) -> Dict[str, Any]:
        """
        Enforces post-processing safety rules:
        - Overrides risk_language_detected to True if pre-LLM check or post-LLM check detects self-harm language.
        - Injects crisis resources when risk language is detected.
        - Ensures standard medical disclaimer is always present.
        """
        pre_risk, pre_triggers = self.check_text_for_risk(user_input)
        
        # Check raw LLM rationale text as well
        rationale_text = llm_response.get("rationale", "")
        post_risk, _ = self.check_text_for_risk(rationale_text)
        
        llm_flagged_risk = bool(llm_response.get("risk_language_detected", False))
        
        final_risk_detected = pre_risk or post_risk or llm_flagged_risk
        
        guarded_response = dict(llm_response)
        guarded_response["risk_language_detected"] = final_risk_detected
        guarded_response["disclaimer"] = STANDARD_DISCLAIMER
        
        if final_risk_detected:
            guarded_response["crisis_resources"] = CRISIS_RESOURCES_INDIA
            # Elevate severity suggestion if risk detected
            if guarded_response.get("severity") in ["minimum", "mild"]:
                guarded_response["confidence_note"] = (
                    guarded_response.get("confidence_note", "") + 
                    " [Safety Note: Explicit risk language was detected in the prompt, prompting immediate crisis intervention resources.]"
                )
        else:
            guarded_response["crisis_resources"] = None
            
        return guarded_response

if __name__ == "__main__":
    safety = SafetyInterceptor()
    test_inputs = [
        "I feel sad and tired today.",
        "I want to end my life, I can't take this anymore."
    ]
    for inp in test_inputs:
        is_r, tr = safety.check_text_for_risk(inp)
        print(f"Input: '{inp}' -> Risk: {is_r} | Triggers: {tr}")
