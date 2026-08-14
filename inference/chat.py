# inference/chat.py
"""Multi-turn Chat Session Manager with ChatML formatting and sliding context."""

from typing import List, Dict
from personas.engine import PersonaManager
from inference.generate import generate_text

class ChatSessionManager:
    def __init__(self, model, tokenizer, device: str = "cpu", max_context_chars: int = 4000):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.max_context_chars = max_context_chars
        self.persona_manager = PersonaManager()
        self.history: List[Dict[str, str]] = []

    def add_message(self, role: str, content: str):
        self.history.append({"role": role, "content": content})

    def clear_history(self):
        self.history.clear()

    def clean_output(self, text: str) -> str:
        tags_to_remove = ["<|im_start|>", "<|im_end|>", "<|endoftext|>", "system\n", "user\n", "assistant\n"]
        for tag in tags_to_remove:
            text = text.replace(tag, "")
        text = text.replace("", "")
        return text.strip()

    def build_chatml_prompt(self, persona_name: str = "General Assistant") -> str:
        persona = self.persona_manager.get_persona(persona_name)
        prompt_parts = [f"<|im_start|>system\n{persona.system_prompt}<|im_end|>"]
        for msg in self.history:
            role = msg["role"]
            content = msg["content"]
            prompt_parts.append(f"<|im_start|>{role}\n{content}<|im_end|>")
        prompt_parts.append("<|im_start|>assistant\n")
        
        full_prompt = "\n".join(prompt_parts)
        if len(full_prompt) > self.max_context_chars:
            full_prompt = full_prompt[-self.max_context_chars:]
        return full_prompt

    def respond(self, user_message: str, persona_name: str = "General Assistant", max_new_tokens: int = 35) -> str:
        self.add_message("user", user_message)
        prompt = self.build_chatml_prompt(persona_name)
        persona = self.persona_manager.get_persona(persona_name)
        
        full_out = generate_text(
            model=self.model,
            tokenizer=self.tokenizer,
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            device=self.device,
            temperature=persona.temperature,
            stop_tokens=["<|im_end|>", "<|endoftext|>"]
        )
        
        if "<|im_start|>assistant\n" in full_out:
            assistant_response = full_out.split("<|im_start|>assistant\n")[-1]
        else:
            assistant_response = full_out[len(prompt):] if full_out.startswith(prompt) else full_out
            
        assistant_response = self.clean_output(assistant_response)
        self.add_message("assistant", assistant_response)
        return assistant_response
