# inference/chat.py
"""Stateful multi-turn conversational session manager with ChatML and sliding context."""

from typing import List, Dict
from personas.engine import PersonaManager
from inference.generate import generate_text

class ChatSessionManager:
    """Manages multi-turn conversation history, persona formatting, and context truncation."""
    def __init__(self, model, tokenizer, device: str = "cpu", max_context_chars: int = 2000):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.max_context_chars = max_context_chars
        self.persona_manager = PersonaManager()
        self.history: List[Dict[str, str]] = []

    def add_message(self, role: str, content: str):
        """Appends a turn to conversation history."""
        self.history.append({"role": role, "content": content})

    def clear_history(self):
        """Resets conversation history."""
        self.history.clear()

    def build_chatml_prompt(self, persona_name: str = "General Assistant") -> str:
        """Constructs an end-to-end ChatML prompt from conversation history."""
        persona = self.persona_manager.get_persona(persona_name)
        prompt_parts = [f"<|im_start|>system\n{persona.system_prompt}<|im_end|>"]
        
        for msg in self.history:
            role = msg["role"]
            content = msg["content"]
            prompt_parts.append(f"<|im_start|>{role}\n{content}<|im_end|>")
            
        prompt_parts.append("<|im_start|>assistant\n")
        full_prompt = "\n".join(prompt_parts)
        
        # Sliding context window truncation if prompt exceeds character limit
        if len(full_prompt) > self.max_context_chars:
            full_prompt = full_prompt[-self.max_context_chars:]
            
        return full_prompt

    def respond(self, user_message: str, persona_name: str = "General Assistant", max_new_tokens: int = 25) -> str:
        """Appends user message, generates autoregressive response, and stores in history."""
        self.add_message("user", user_message)
        prompt = self.build_chatml_prompt(persona_name)
        persona = self.persona_manager.get_persona(persona_name)
        
        response = generate_text(
            model=self.model,
            tokenizer=self.tokenizer,
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            device=self.device,
            temperature=persona.temperature
        )
        
        self.add_message("assistant", response)
        return response
