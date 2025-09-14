"""Mock embedder for testing and offline use."""

import numpy as np
import hashlib
from typing import Sequence


class MockEmbedder:
    """A simple deterministic mock embedder for testing and offline use."""
    
    def __init__(self, dimension: int = 128):
        """Initialize mock embedder.
        
        Args:
            dimension: Embedding dimension (default 128 for faster computation)
        """
        self.dimension = dimension
    
    def embed(self, texts: Sequence[str]) -> np.ndarray:
        """Create deterministic embeddings from text hashes.
        
        This creates reasonably realistic embeddings where semantically similar
        texts will have similar embeddings (though not as sophisticated as real
        embeddings).
        
        Args:
            texts: Sequence of texts to embed
            
        Returns:
            np.ndarray: L2-normalized embeddings of shape (N, dimension)
        """
        if not texts:
            # Return empty array with correct shape for empty input
            return np.empty((0, self.dimension))
        
        embeddings = []
        
        for text in texts:
            # Create deterministic embedding from text content
            embedding = self._text_to_embedding(text)
            embeddings.append(embedding)
        
        embeddings = np.array(embeddings)
        
        # L2-normalize
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-12)  # Avoid division by zero
        embeddings = embeddings / norms
        
        return embeddings
    
    def _text_to_embedding(self, text: str) -> np.ndarray:
        """Convert text to deterministic embedding vector."""
        # Clean and normalize text
        text = text.lower().strip()
        
        # Use multiple hash functions to create different components
        hash_funcs = [
            lambda t: hashlib.md5(t.encode()).hexdigest(),
            lambda t: hashlib.sha1(t.encode()).hexdigest(),
            lambda t: hashlib.sha256(t.encode()).hexdigest(),
        ]
        
        # Create embedding components
        embedding = np.zeros(self.dimension)
        
        for i, hash_func in enumerate(hash_funcs):
            hash_str = hash_func(text)
            
            # Convert hex string to numbers
            for j, char in enumerate(hash_str[:self.dimension//len(hash_funcs)]):
                idx = (i * self.dimension // len(hash_funcs)) + j
                if idx < self.dimension:
                    # Convert hex char to number and normalize to [-1, 1]
                    embedding[idx] = (int(char, 16) / 15.0) * 2 - 1
        
        # Add some simple semantic features
        # Words with similar meanings get similar base patterns
        semantic_words = {
            'billing': [0.8, -0.2, 0.3],
            'payment': [0.7, -0.3, 0.4], 
            'invoice': [0.6, -0.2, 0.5],
            'technical': [-0.2, 0.8, 0.1],
            'support': [-0.3, 0.7, 0.2],
            'help': [-0.1, 0.6, 0.3],
            'error': [-0.4, 0.5, -0.2],
            'account': [0.2, 0.1, 0.7],
            'profile': [0.3, 0.0, 0.6],
            'login': [0.1, 0.2, 0.8],
        }
        
        # Apply semantic adjustments
        for word, pattern in semantic_words.items():
            if word in text:
                for i, val in enumerate(pattern):
                    if i < self.dimension:
                        embedding[i] += val * 0.3  # Small semantic boost
        
        return embedding


def create_mock_embedder(dimension: int = 128) -> MockEmbedder:
    """Create a mock embedder instance."""
    return MockEmbedder(dimension=dimension)