# rag/citation.py
"""Citation grounding engine that tags generated text with source references."""

from typing import List, Dict, Tuple

class CitationGrounder:
    """Maps retrieved document chunks into structured citations."""
    @staticmethod
    def format_grounded_context(chunks: List[Tuple[str, float]]) -> Tuple[str, Dict[int, str]]:
        """
        Formats retrieved chunks with explicit citation identifiers [Source 1], [Source 2].
        """
        grounded_lines = []
        citation_map = {}
        
        for idx, (chunk, score) in enumerate(chunks, 1):
            tag = f"[Source {idx}]"
            grounded_lines.append(f"{tag} (Relevance: {score:.3f}): {chunk}")
            citation_map[idx] = chunk
            
        return "\n".join(grounded_lines), citation_map

    @staticmethod
    def ground_response(response: str, citation_map: Dict[int, str]) -> str:
        """Appends a citation footer linking sources to generated output."""
        footer_lines = ["\n\n### Grounded Sources:"]
        for idx, chunk in citation_map.items():
            snippet = chunk[:80] + "..." if len(chunk) > 80 else chunk
            footer_lines.append(f"- **[Source {idx}]**: {snippet}")
        return response + "\n".join(footer_lines)
