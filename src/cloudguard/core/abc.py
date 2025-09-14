"""Protocol interfaces for dependency injection from the host graph."""

from typing import Protocol, Sequence, List, Optional, Mapping, Any
import numpy as np

class Embeddings(Protocol):
    """Graph-injected embeddings provider. Implement with SentenceTransformers, OpenAI, Cohere, etc."""
    
    def embed(self, texts: Sequence[str]) -> np.ndarray:
        """
        Embed a sequence of texts into vectors.
        
        Args:
            texts: Sequence of text strings to embed
            
        Returns:
            np.ndarray: Shape (N, D). Should be L2-normalized by the provider or caller.
        """
        ...

class Segmenter(Protocol):
    """Graph-injected text segmenter (optional). If None, use deterministic sentence splitter."""
    
    def segment(self, text: str) -> List[str]:
        """
        Segment text into meaningful units (sentences, requests, etc.).
        
        Args:
            text: Input text to segment
            
        Returns:
            List[str]: List of text segments
        """
        ...

class Logger(Protocol):
    """Optional structured logging interface."""
    
    def info(self, msg: str, **kv: Any) -> None:
        """Log info level message with optional key-value context."""
        ...
        
    def warn(self, msg: str, **kv: Any) -> None:
        """Log warning level message with optional key-value context."""
        ...
        
    def error(self, msg: str, **kv: Any) -> None:
        """Log error level message with optional key-value context."""
        ...

class Meter(Protocol):
    """Optional metrics collection interface."""
    
    def inc(self, name: str, amount: int = 1, **tags: str) -> None:
        """Increment a counter metric with optional tags."""
        ...
        
    def observe(self, name: str, value: float, **tags: str) -> None:
        """Record an observation metric with optional tags."""
        ...