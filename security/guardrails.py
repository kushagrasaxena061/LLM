"""
Comprehensive but honest security guardrails for LLM inputs and UNTRUSTED RAG data.
"""
import re
import base64
import unicodedata
import logging

logger = logging.getLogger(__name__)

class SecurityGuard:
    def __init__(self):
        # LIMITATION DISCLOSURE:
        # These are heuristic and regex-based guardrails. They offer a baseline layer of defense
        # against unsophisticated attacks but are NOT a bulletproof semantic firewall. 
        # They can be bypassed by advanced token smuggling, adversarial suffixes, or zero-shot translation attacks.
        self.injection_patterns = [
            re.compile(r'(?i)ignore (all )?previous instructions'),
            re.compile(r'(?i)disregard (all )?previous instructions'),
            re.compile(r'(?i)you are now (a )?jailbroken'),
            re.compile(r'(?i)(system prompt|initial instructions|developer instructions)'),
            re.compile(r'(?i)forget what you were told'),
            re.compile(r'(?i)reveal your (instructions|prompt)'),
        ]
        self.pii_patterns = {
            "EMAIL": re.compile(r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}'),
            "PHONE": re.compile(r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b'),
        }

    def _check_encoded(self, text: str) -> bool:
        """Basic heuristic to check for Base64 encoded payload attempts."""
        words = text.replace('\n', ' ').split()
        for word in words:
            if len(word) >= 16 and len(word) % 4 == 0:
                try:
                    decoded = base64.b64decode(word).decode('utf-8')
                    if any(p.search(decoded) for p in self.injection_patterns):
                        return True
                except Exception:
                    pass
        return False

    def _normalize_and_check(self, text: str) -> bool:
        """Normalize unicode to catch homoglyphs or zero-width character obfuscation."""
        normalized = unicodedata.normalize('NFKC', text)
        cleaned = re.sub(r'[\u200B-\u200D\uFEFF]', '', normalized)
        return any(p.search(cleaned) for p in self.injection_patterns)

    def sanitize_pii(self, text: str) -> str:
        return self.validate_input(text)["sanitized_prompt"]

    def validate_input(self, prompt: str, is_retrieved_context: bool = False) -> dict:
        # 8. Oversized input protection
        if len(prompt) > 10000:
            return {"is_safe": False, "injection_detected": True, "matched_pattern": "oversized", "sanitized_prompt": "", "reason": "Input exceeds safe length limits."}

        # 1, 3, 5, 6, 7. Injection and Obfuscation checks
        if self._check_encoded(prompt):
            return {"is_safe": False, "injection_detected": True, "matched_pattern": "encoded_payload", "sanitized_prompt": "", "reason": "Encoded prompt injection detected."}

        if self._normalize_and_check(prompt):
            return {"is_safe": False, "injection_detected": True, "matched_pattern": "regex_heuristic", "sanitized_prompt": "", "reason": "Direct prompt injection or extraction detected."}

        # 9. PII protection
        sanitized = prompt
        for label, pattern in self.pii_patterns.items():
            sanitized = pattern.sub(f'[REDACTED_{label}]', sanitized)

        # 4, 10. RAG Document Security - Explicit Escape
        if is_retrieved_context:
            # Strip injection tokens that might escape context blocks
            sanitized = sanitized.replace("<|im_start|>", "").replace("<|im_end|>", "").replace("<|endoftext|>", "")
            
        return {
            "is_safe": True, "injection_detected": False,
            "matched_pattern": None,
            "sanitized_prompt": sanitized,
            "reason": "Passed heuristic checks. LIMITATION: Not immune to semantic bypass."
        }
        
    def isolate_untrusted_context(self, text: str) -> str:
        """
        Wraps retrieved documents in safe isolation tags and strips container escape sequences.
        Ensures UNTRUSTED DATA never becomes system instructions.
        """
        safe_text = self.validate_input(text, is_retrieved_context=True)
        if not safe_text["is_safe"]:
            return "<untrusted_document>\n[MALICIOUS CONTENT REDACTED]\n</untrusted_document>"
        return f"<untrusted_document>\n{safe_text['sanitized_prompt']}\n</untrusted_document>"

# Backward compatibility functions
_guard = SecurityGuard()
def detect_prompt_injection(prompt: str) -> bool:
    return not _guard.validate_input(prompt)["is_safe"]

def sanitize_pii(text: str) -> str:
    return _guard.validate_input(text)["sanitized_prompt"]
