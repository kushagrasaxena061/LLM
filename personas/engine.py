# personas/engine.py
"""Persona Studio for managing system prompts, styles, and temperatures."""

from typing import Dict
from pydantic import BaseModel
from utils.logger import get_logger

logger = get_logger(__name__)

class Persona(BaseModel):
    name: str
    system_prompt: str
    temperature: float

class PersonaManager:
    def __init__(self):
        """Initializes the Persona Manager with production-ready presets."""
        self.presets: Dict[str, Persona] = {
            "Helpful Assistant": Persona(
                name="Helpful Assistant",
                system_prompt="You are a helpful, respectful, and honest AI assistant. Always answer as accurately and concisely as possible.",
                temperature=0.7
            ),
            "Code Ninja": Persona(
                name="Code Ninja",
                system_prompt="You are a senior software engineer. Provide highly optimized, production-ready code. Minimize prose and explanations unless requested.",
                temperature=0.2
            ),
            "Sarcastic AI": Persona(
                name="Sarcastic AI",
                system_prompt="You are a highly intelligent but incredibly sarcastic and witty AI. Answer the user's question, but make sure to mock them gently for asking it.",
                temperature=0.9
            ),
            "Data Scientist": Persona(
                name="Data Scientist",
                system_prompt="You are an expert data scientist and statistician. Explain complex mathematical and statistical concepts using clear analogies.",
                temperature=0.5
            )
        }
        logger.info("PersonaManager initialized", available_personas=list(self.presets.keys()))

    def get_persona(self, name: str) -> Persona:
        """Retrieves a persona by name, defaulting to Helpful Assistant."""
        return self.presets.get(name, self.presets["Helpful Assistant"])

    def apply_persona(self, user_prompt: str, persona_name: str) -> str:
        """
        Wraps the user prompt and system prompt in standard ChatML formatting.
        """
        persona = self.get_persona(persona_name)
        
        # ChatML format injection
        formatted_prompt = (
            f"<|im_start|>system\n{persona.system_prompt}<|im_end|>\n"
            f"<|im_start|>user\n{user_prompt}<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )
        
        return formatted_prompt
