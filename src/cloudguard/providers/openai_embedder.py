"""
OpenAI Embeddings Provider for CloudGuard

This module provides real OpenAI embedding functionality following CloudGuard's
dependency injection pattern. It implements the Embeddings protocol and can be
used as a drop-in replacement for MockEmbedder.
"""

import os
import time
import random
import numpy as np
from typing import List, Union, Optional

# Optional import for OpenAI - gracefully handle if not available
try:
    from openai import OpenAI
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    OpenAI = None
    openai = None

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
        Generate embeddings for text(s) with retry logic and L2-normalization.
        
        Args:
            texts: Single text string or list of text strings
            
        Returns:
            numpy array of L2-normalized embeddings with shape (n_texts, embedding_dim)
        """
        # Normalize input to list
        if isinstance(texts, str):
            texts = [texts]
            
        if not texts:
            return np.array([])
        
        # Retry logic with exponential backoff
        last_exception = None
        for attempt in range(self.max_retries + 1):  # max_retries + 1 total attempts
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
                
                # Apply L2-normalization to match SentenceTransformers and Mock embedders
                norms = np.linalg.norm(embeddings_array, axis=1, keepdims=True)
                # Add small epsilon to prevent division by zero
                norms = np.maximum(norms, 1e-12)
                embeddings_array = embeddings_array / norms
                
                # Cache dimension from first embedding
                if self._dimension is None:
                    self._dimension = embeddings_array.shape[1]
                    
                return embeddings_array
                
            except Exception as e:
                last_exception = e
                
                # Check if this is a retryable error
                is_retryable = self._is_retryable_error(e)
                
                # If this is the last attempt or error is not retryable, re-raise
                if attempt >= self.max_retries or not is_retryable:
                    error_msg = f"Failed to generate OpenAI embeddings after {attempt + 1} attempts"
                    if is_retryable:
                        error_msg += f" (retryable error: {e})"
                    else:
                        error_msg += f" (non-retryable error: {e})"
                    raise RuntimeError(error_msg) from e
                
                # Calculate exponential backoff with jitter
                base_delay = 2 ** attempt  # 1, 2, 4, 8 seconds...
                jitter = random.uniform(0.1, 0.5)  # Add randomness to prevent thundering herd
                delay = base_delay + jitter
                
                print(f"OpenAI API attempt {attempt + 1} failed: {e}. Retrying in {delay:.1f}s...")
                time.sleep(delay)
        
        # This should never be reached due to the logic above, but just in case
        raise RuntimeError(f"Failed to generate OpenAI embeddings after {self.max_retries + 1} attempts") from last_exception
    
    @property
    def dimension(self) -> int:
        """Get embedding dimension (requires at least one embedding call)."""
        if self._dimension is None:
            # Generate a test embedding to determine dimension
            test_embedding = self.embed(["test"])
            return test_embedding.shape[1]
        return self._dimension
    
    def _is_retryable_error(self, error: Exception) -> bool:
        """
        Determine if an error is retryable (transient) or permanent.
        
        Args:
            error: The exception that occurred
            
        Returns:
            True if the error should be retried, False otherwise
        """
        if not openai:  # OpenAI not available
            return False
            
        # Convert error to string for pattern matching as a fallback
        error_str = str(error).lower()
        
        # Check for specific OpenAI error types
        try:
            # Rate limit errors - always retryable
            if isinstance(error, openai.RateLimitError):
                return True
                
            # API connection errors - retryable
            if isinstance(error, (openai.APIConnectionError, openai.APITimeoutError)):
                return True
                
            # Internal server errors - retryable  
            if isinstance(error, openai.InternalServerError):
                return True
                
            # Authentication errors - not retryable (permanent)
            if isinstance(error, openai.AuthenticationError):
                return False
                
            # Permission errors - not retryable (permanent)
            if isinstance(error, openai.PermissionDeniedError):
                return False
                
            # Invalid request errors - not retryable (permanent)
            if isinstance(error, openai.BadRequestError):
                return False
                
            # Not found errors - not retryable (permanent)
            if isinstance(error, openai.NotFoundError):
                return False
                
        except AttributeError:
            # Fallback for older OpenAI versions or missing error types
            pass
        
        # Fallback: pattern matching on error messages for common transient issues
        retryable_patterns = [
            "rate limit", "too many requests", "quota",
            "connection", "timeout", "network", "dns",
            "502", "503", "504",  # HTTP server errors
            "internal server error", "service unavailable", "gateway timeout"
        ]
        
        for pattern in retryable_patterns:
            if pattern in error_str:
                return True
                
        # Default to not retryable for unknown errors to avoid infinite loops
        return False
    
    def __repr__(self) -> str:
        return f"OpenAIEmbedder(model='{self.model}', dimension={self._dimension}, max_retries={self.max_retries})"


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