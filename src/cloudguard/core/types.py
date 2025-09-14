"""Data types and result structures for CloudGuard operations."""

from dataclasses import dataclass
from typing import List, Optional, Dict, Any

@dataclass
class RouteDecision:
    """Result of input routing decision."""
    decision: str                # "route" | "abstain"
    score: float                # Similarity score to best matching region
    region_id: Optional[str] = None
    target: Optional[str] = None     # Where to route (from policy)
    reason: Optional[str] = None     # Why abstained (if applicable)

@dataclass
class ValidationResult:
    """Result of output validation and filtering."""
    ok: bool                        # Overall validation passed
    kept_text: str                 # Text after filtering (if any)
    coverage: List[bool]           # Per input segment coverage
    dropped_segments: int          # Number of segments dropped as off-topic
    scores_summary: Dict[str, float]  # Optional score statistics (avg/min/max)
    
    @property
    def coverage_ratio(self) -> float:
        """Fraction of input segments covered by output."""
        if not self.coverage:
            return 0.0
        return sum(self.coverage) / len(self.coverage)