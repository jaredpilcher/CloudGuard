"""Deterministic sentence segmenter with no external dependencies."""

import re
from typing import List

class SentenceSegmenter:
    """
    Deterministic rule-based sentence segmenter.
    Splits on sentence boundaries without requiring external libraries.
    """
    
    def __init__(self, min_length: int = 5):
        """
        Initialize segmenter.
        
        Args:
            min_length: Minimum character length for a valid segment
        """
        self.min_length = min_length
        
    def segment(self, text: str) -> List[str]:
        """
        Segment text into sentences using deterministic rules.
        
        Args:
            text: Input text to segment
            
        Returns:
            List[str]: List of sentence segments
        """
        if not text.strip():
            return []
            
        # Split on sentence-ending punctuation followed by whitespace
        sentences = re.split(r'(?<=[.?!])\s+', text.strip())
        
        # Filter out very short segments and clean up
        result = []
        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) >= self.min_length:
                result.append(sentence)
                
        return result