"""
OpenAI Embeddings Provider for CloudGuard

This module provides real OpenAI embedding functionality following CloudGuard's
dependency injection pattern. It implements the Embeddings protocol and can be
used as a drop-in replacement for MockEmbedder.
"""

import os
import numpy as np
from typing import List, Union, Optional

# Optional import for OpenAI - gracefully handle if not available
try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    OpenAI = None

class OpenAIEmbedder:
    """
    Real OpenAI embeddings provider for CloudGuard.
    
    Follows CloudGuard's dependency injection pattern - embeddings provider
    is injected by the host application rather than being a hard dependency.
    """
    
    def __init__(self, 
                 model: str = "text-embedding-3-small", 
                 api_key: Optional[str] = None,
                 max_retries: int = 3):
        """
        Initialize OpenAI embedder.
        
        Args:
            model: OpenAI embedding model to use
            api_key: OpenAI API key (if None, uses OPENAI_API_KEY environment variable)
            max_retries: Maximum number of retry attempts for API calls
        """
        if not OPENAI_AVAILABLE:
            raise ImportError(
                "OpenAI package is not installed. Install it with: pip install openai"
            )
            
        self.model = model
        self.max_retries = max_retries
        
        # Get API key from parameter or environment
        if api_key is None:
            api_key = os.environ.get("OPENAI_API_KEY")
            
        if not api_key:
            raise ValueError(
                "OpenAI API key is required. Provide via api_key parameter "
                "or set OPENAI_API_KEY environment variable."
            )
            
        # Initialize OpenAI client
        # the newest OpenAI model is "gpt-5" which was released August 7, 2025.
        # do not change this unless explicitly requested by the user
        self.client = OpenAI(api_key=str(api_key))
        
        # Cache embedding dimension for consistency
        self._dimension = None
    
    def embed(self, texts: Union[str, List[str]]) -> np.ndarray:
        """
        Generate embeddings for text(s).
        
        Args:
            texts: Single text string or list of text strings
            
        Returns:
            numpy array of embeddings with shape (n_texts, embedding_dim)
        """
        # Normalize input to list
        if isinstance(texts, str):
            texts = [texts]
            
        if not texts:
            return np.array([])
            
        try:
            # Call OpenAI embeddings API
            response = self.client.embeddings.create(
                model=self.model,
                input=texts
            )
            
            # Extract embeddings from response
            embeddings = []
            for embedding_obj in response.data:
                embeddings.append(embedding_obj.embedding)
                
            # Convert to numpy array and ensure float32 for consistency
            embeddings_array = np.array(embeddings, dtype=np.float32)
            
            # Cache dimension from first embedding
            if self._dimension is None:
                self._dimension = embeddings_array.shape[1]
                
            return embeddings_array
            
        except Exception as e:
            raise RuntimeError(f"Failed to generate OpenAI embeddings: {e}")
    
    @property
    def dimension(self) -> int:
        """Get embedding dimension (requires at least one embedding call)."""
        if self._dimension is None:
            # Generate a test embedding to determine dimension
            test_embedding = self.embed(["test"])
            return test_embedding.shape[1]
        return self._dimension
    
    def __repr__(self) -> str:
        return f"OpenAIEmbedder(model='{self.model}', dimension={self._dimension})"


# Convenience factory functions for common use cases
def create_openai_embedder(model: str = "text-embedding-3-small") -> Optional[OpenAIEmbedder]:
    """
    Create OpenAI embedder with default settings.
    
    Args:
        model: OpenAI embedding model to use
        
    Returns:
        Configured OpenAIEmbedder instance, or None if OpenAI is not available
    """
    try:
        return OpenAIEmbedder(model=model)
    except (ImportError, ValueError):
        return None


def create_fast_embedder() -> Optional[OpenAIEmbedder]:
    """Create fast, cost-effective OpenAI embedder."""
    try:
        return OpenAIEmbedder(model="text-embedding-3-small")
    except (ImportError, ValueError):
        return None


def create_high_quality_embedder() -> Optional[OpenAIEmbedder]:
    """Create high-quality OpenAI embedder."""
    try:
        return OpenAIEmbedder(model="text-embedding-3-large")
    except (ImportError, ValueError):
        return None


def is_openai_available() -> bool:
    """Check if OpenAI package and API key are available."""
    return OPENAI_AVAILABLE and os.environ.get("OPENAI_API_KEY") is not None