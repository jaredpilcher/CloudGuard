"""Vector distance and normalization utilities."""

import numpy as np

def l2_normalize(M: np.ndarray) -> np.ndarray:
    """
    L2 normalize vectors along the last dimension.
    
    Args:
        M: Array of shape (..., D) where D is the vector dimension
        
    Returns:
        np.ndarray: L2-normalized vectors of same shape
    """
    norm = np.linalg.norm(M, axis=-1, keepdims=True) + 1e-12
    return M / norm

def cosine(u: np.ndarray, v: np.ndarray) -> float:
    """
    Compute cosine similarity between two L2-normalized vectors.
    
    Args:
        u: First vector (must be L2-normalized)
        v: Second vector (must be L2-normalized)
        
    Returns:
        float: Cosine similarity (dot product for normalized vectors)
    """
    return float(np.dot(u, v))

def cosine_batch(queries: np.ndarray, targets: np.ndarray) -> np.ndarray:
    """
    Compute cosine similarities between queries and targets.
    
    Args:
        queries: Shape (N, D) - query vectors (L2-normalized)
        targets: Shape (M, D) - target vectors (L2-normalized)
        
    Returns:
        np.ndarray: Shape (N, M) - similarity matrix
    """
    return np.dot(queries, targets.T)