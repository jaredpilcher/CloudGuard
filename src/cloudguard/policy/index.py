"""Region index builder using injected embeddings provider."""

from dataclasses import dataclass
from typing import List, Optional
import numpy as np
from ..core.abc import Embeddings
from ..core.distance import l2_normalize
from .schema import CloudPolicy

@dataclass
class RegionProto:
    """Prototype representing a semantic region with embedding centroid."""
    region_id: str
    routes_to: Optional[str]
    centroid: np.ndarray         # shape (D,) - L2-normalized centroid vector
    label: str                   # human-readable label
    seed_count: int             # number of seeds used to build this centroid

@dataclass
class RegionIndex:
    """Index of semantic regions with their embedding centroids."""
    protos: List[RegionProto]
    dim: int                    # embedding dimension
    policy_version: int         # for cache invalidation
    
    def find_best_match(self, query_vector: np.ndarray) -> tuple[Optional[RegionProto], float]:
        """
        Find the best matching region for a query vector.
        
        Args:
            query_vector: L2-normalized query vector of shape (D,)
            
        Returns:
            tuple: (best_region_proto_or_none, similarity_score)
        """
        if not self.protos:
            return None, 0.0
            
        best_proto = None
        best_score = -1.0
        
        for proto in self.protos:
            # Cosine similarity (dot product for normalized vectors)
            score = float(np.dot(query_vector, proto.centroid))
            if score > best_score:
                best_score = score
                best_proto = proto
                
        return best_proto, best_score
    
    def get_region_by_id(self, region_id: str) -> Optional[RegionProto]:
        """Get region prototype by ID."""
        for proto in self.protos:
            if proto.region_id == region_id:
                return proto
        return None

def build_region_index(policy: CloudPolicy, embedder: Embeddings) -> RegionIndex:
    """
    Build a region index from policy using the injected embeddings provider.
    
    Args:
        policy: Validated CloudGuard policy
        embedder: Embeddings provider (implements Protocol)
        
    Returns:
        RegionIndex: Built index with region centroids
        
    Raises:
        ValueError: If embeddings fail or have inconsistent dimensions
    """
    if not policy.regions:
        return RegionIndex(protos=[], dim=0, policy_version=policy.version)
    
    protos: List[RegionProto] = []
    embedding_dim = None
    
    for region in policy.regions:
        if not region.seeds:
            # Skip regions with no seeds (validation should catch this)
            continue
            
        try:
            # Get embeddings for all seeds in this region
            seed_vecs = embedder.embed(region.seeds)  # Shape: (N, D)
            
            if seed_vecs.size == 0:
                raise ValueError(f"Embedder returned empty array for region '{region.id}'")
                
            # Ensure 2D array
            if seed_vecs.ndim == 1:
                seed_vecs = seed_vecs.reshape(1, -1)
            elif seed_vecs.ndim != 2:
                raise ValueError(f"Expected 2D embeddings, got shape {seed_vecs.shape}")
                
            # Check dimension consistency
            curr_dim = seed_vecs.shape[1]
            if embedding_dim is None:
                embedding_dim = curr_dim
            elif embedding_dim != curr_dim:
                raise ValueError(f"Inconsistent embedding dimensions: {embedding_dim} vs {curr_dim}")
                
            # L2-normalize seed vectors
            seed_vecs = l2_normalize(seed_vecs)
            
            # Compute centroid and normalize it
            centroid = seed_vecs.mean(axis=0)
            centroid_norm = np.linalg.norm(centroid)
            if centroid_norm < 1e-12:
                raise ValueError(f"Degenerate centroid for region '{region.id}' (norm ~0)")
            centroid = centroid / centroid_norm
            
            # Create region prototype
            proto = RegionProto(
                region_id=region.id,
                routes_to=region.routes_to,
                centroid=centroid,
                label=region.label,
                seed_count=len(region.seeds)
            )
            protos.append(proto)
            
        except Exception as e:
            raise ValueError(f"Failed to build centroid for region '{region.id}': {e}")
    
    final_dim = embedding_dim if embedding_dim is not None else 0
    return RegionIndex(protos=protos, dim=final_dim, policy_version=policy.version)