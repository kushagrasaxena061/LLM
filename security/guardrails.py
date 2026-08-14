# security/guardrails.py
"""Security layer providing prompt injection defense and PII sanitization."""

import re
from typing import Dict, Tuple
from utils.logger import get_logger

logger = get_logger(__name__)

class SecurityGuardrail:
    def __init__(self):
        # Heuristic detection patterns for common prompt injections and jailbreaks
        self.injection_patterns = [
            r"ignore (all )?previous instructions",
            r"disregard the above",
            r"you are now in DAN mode",
            r"system prompt:",
            r"reveal your instructions",
            r"bypass safety protocols",
            r"override rules?"
        ]
        
        # Regex patterns for Personally Identifiable Information (PII)
        self.pii_patterns = {
            "email": r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}",
            "phone": r"\b\d{3}[-.]?\d{3}[-.]?\d{4}\b",
            "ssn": r"\b\d{3}-\d{2}-\d{4}\b",
            "credit_card": r"\b(?:\d[ -]*?){13,16}\b"
        }

    def check_prompt_injection(self, text: str) -> Tuple[bool, str]:
        """Checks if input text contains prompt injection or jailbreak attempts."""
        for pattern in self.injection_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                logger.warning("Prompt injection attempt detected", pattern=pattern)
                return True, pattern
        return False, ""

    def sanitize_pii(self, text: str) -> str:
        """Redacts sensitive PII information from text."""
        sanitized = text
        for pii_type, pattern in self.pii_patterns.items():
            sanitized = re.sub(pattern, f"[REDACTED_{pii_type.upper()}]", sanitized)
        return sanitized

    def validate_input(self, prompt: str) -> Dict:
        """Validates and sanitizes incoming user prompts."""
        is_injection, pattern = self.check_prompt_injection(prompt)
        clean_prompt = self.sanitize_pii(prompt)
        
        return {
            "is_safe": not is_injection,
            "injection_detected": is_injection,
            "matched_pattern": pattern,
            "sanitized_prompt": clean_prompt
        }

# Alias both class names so imports for either SecurityGuard or SecurityGuardrail succeed
SecurityGuard = SecurityGuardrail
