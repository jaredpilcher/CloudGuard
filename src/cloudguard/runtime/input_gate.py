"""Input gate for routing decisions based on embedding similarity."""

import numpy as np
from typing import Optional
from ..core.types import RouteDecision
from ..core.distance import l2_normalize, cosine
from ..policy.index import RegionIndex
from ..policy.schema import CloudPolicy
from ..core.abc import Embeddings, Logger, Meter

class InputCloudGate:
    """
    Input gate that routes queries based on semantic similarity to regions.
    Uses cosine similarity with configurable thresholds and abstain margins.
    """
    
    def __init__(self, *, policy: CloudPolicy, index: RegionIndex, embedder: Embeddings,
                 logger: Optional[Logger] = None, meter: Optional[Meter] = None):
        """
        Initialize input gate with policy and dependencies.
        
        Args:
            policy: Validated CloudGuard policy
            index: Built region index with centroids
            embedder: Embeddings provider for query embedding
            logger: Optional structured logger
            meter: Optional metrics collector
        """
        self.policy = policy
        self.index = index
        self.embedder = embedder
        self.log = logger
        self.meter = meter

    def route(self, text: str) -> RouteDecision:
        """
        Route a text query to the best matching region or abstain.
        
        Args:
            text: Input text to route
            
        Returns:
            RouteDecision: Routing decision with score and target
        """
        # Embed the query text
        try:
            query_embeddings = self.embedder.embed([text])  # Shape: (1, D)
            if query_embeddings.size == 0:
                if self.log:
                    self.log.error("Empty embedding returned", text_length=len(text))
                return self._abstain_decision(None, 0.0, "embedding_failed")
                    
            # Ensure 2D and normalize
            if query_embeddings.ndim == 1:
                query_embeddings = query_embeddings.reshape(1, -1)
            query_vec = l2_normalize(query_embeddings)[0]  # Shape: (D,)
            
        except Exception as e:
            if self.log:
                self.log.error("Embedding failed", error=str(e), text_length=len(text))
            return self._abstain_decision(None, 0.0, "embedding_error")
        
        # Find best matching region
        best_proto, best_score = self.index.find_best_match(query_vec)
        
        # Apply thresholds
        in_cloud_threshold = self.policy.thresholds.in_cloud
        margin = self.policy.thresholds.margin
        
        # Check if clearly in a region
        if best_proto and best_score >= in_cloud_threshold:
            if self.meter:
                self.meter.inc("cloudguard.in_cloud_hit", region=best_proto.region_id)
            if self.log:
                self.log.info("in_cloud_route", 
                            region=best_proto.region_id, 
                            score=best_score,
                            target=best_proto.routes_to)
            
            return RouteDecision(
                decision="route",
                score=best_score,
                region_id=best_proto.region_id,
                target=best_proto.routes_to,
                reason="in_cloud"
            )
        
        # Determine abstain reason
        if best_proto and (in_cloud_threshold - margin) <= best_score < in_cloud_threshold:
            reason = "near_boundary"
            if self.meter:
                self.meter.inc("cloudguard.boundary_abstain", region=best_proto.region_id)
        else:
            reason = "out_of_cloud"
            if self.meter:
                self.meter.inc("cloudguard.out_of_cloud")
        
        region_id = best_proto.region_id if best_proto else None
        if self.log:
            self.log.warn("abstain_decision",
                         region=region_id,
                         score=best_score,
                         reason=reason)
        
        return self._abstain_decision(best_proto, best_score, reason)
    
    def _abstain_decision(self, best_proto, score: float, reason: str) -> RouteDecision:
        """Create an abstain decision with appropriate target."""
        return RouteDecision(
            decision="abstain",
            score=score,
            region_id=best_proto.region_id if best_proto else None,
            target=self.policy.routing.default_target,
            reason=reason
        )