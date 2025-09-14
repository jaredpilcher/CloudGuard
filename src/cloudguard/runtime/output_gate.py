"""Output gate for validation and filtering of LLM responses."""

from typing import List, Optional
import numpy as np
from ..core.types import ValidationResult
from ..core.distance import l2_normalize, cosine
from ..policy.schema import CloudPolicy
from ..core.abc import Embeddings, Segmenter, Logger, Meter

class OutputCloudGate:
    """
    Output gate that validates LLM responses against user input.
    Checks coverage and optionally filters off-topic segments.
    """
    
    def __init__(self, *, policy: CloudPolicy, embedder: Embeddings,
                 segmenter: Optional[Segmenter] = None,
                 require_coverage: bool = True,
                 drop_offtopic: bool = True,
                 logger: Optional[Logger] = None, 
                 meter: Optional[Meter] = None):
        """
        Initialize output gate with policy and dependencies.
        
        Args:
            policy: Validated CloudGuard policy
            embedder: Embeddings provider
            segmenter: Optional text segmenter (fallback to deterministic)
            require_coverage: Whether all input segments must be covered
            drop_offtopic: Whether to filter out off-topic output segments
            logger: Optional structured logger
            meter: Optional metrics collector
        """
        self.policy = policy
        self.embedder = embedder
        self.segmenter = segmenter
        self.require_coverage = require_coverage
        self.drop_offtopic = drop_offtopic
        self.log = logger
        self.meter = meter

    def _segment_text(self, text: str) -> List[str]:
        """Segment text using injected segmenter or fallback."""
        if self.segmenter:
            return self.segmenter.segment(text)
        
        # Fallback deterministic sentence splitter
        import re
        sentences = re.split(r'(?<=[.?!])\s+', text.strip())
        return [s.strip() for s in sentences if s.strip()]

    def validate(self, user_text: str, llm_text: str) -> ValidationResult:
        """
        Validate LLM output against user input for coverage and relevance.
        
        Args:
            user_text: Original user input text
            llm_text: LLM generated response text
            
        Returns:
            ValidationResult: Validation outcome with filtered text
        """
        try:
            # Segment both texts
            user_segments = self._segment_text(user_text)
            llm_segments = self._segment_text(llm_text)
            
            if not user_segments:
                if self.log:
                    self.log.warn("Empty user segments", user_text_length=len(user_text))
                return ValidationResult(ok=True, kept_text=llm_text, coverage=[], 
                                      dropped_segments=0, scores_summary={})
            
            if not llm_segments:
                if self.log:
                    self.log.warn("Empty LLM segments", llm_text_length=len(llm_text))
                return ValidationResult(ok=False, kept_text="", coverage=[False] * len(user_segments),
                                      dropped_segments=0, scores_summary={})
            
            # Embed all segments
            user_vecs = self.embedder.embed(user_segments)
            llm_vecs = self.embedder.embed(llm_segments)
            
            # Normalize
            user_vecs = l2_normalize(user_vecs)
            llm_vecs = l2_normalize(llm_vecs)
            
            threshold = self.policy.thresholds.in_cloud
            
            # Check coverage: each user segment must have ≥1 aligned LLM segment
            coverage = []
            for i, user_vec in enumerate(user_vecs):
                max_sim = max((cosine(user_vec, llm_vec) for llm_vec in llm_vecs), default=0.0)
                is_covered = max_sim >= threshold
                coverage.append(is_covered)
            
            # Filter LLM segments for relevance (if enabled)
            kept_segments = []
            dropped_count = 0
            
            for i, llm_vec in enumerate(llm_vecs):
                max_sim = max((cosine(llm_vec, user_vec) for user_vec in user_vecs), default=0.0)
                is_relevant = max_sim >= threshold
                
                if is_relevant or not self.drop_offtopic:
                    kept_segments.append(llm_segments[i])
                else:
                    dropped_count += 1
            
            # Overall validation
            validation_ok = all(coverage) if self.require_coverage else True
            kept_text = " ".join(kept_segments)
            
            # Metrics
            if self.meter:
                coverage_ratio = sum(coverage) / len(coverage) if coverage else 0.0
                self.meter.observe("cloudguard.coverage_ratio", coverage_ratio)
                self.meter.inc("cloudguard.dropped_segments", dropped_count)
                
            if self.log:
                self.log.info("output_validation",
                            coverage_ratio=sum(coverage) / len(coverage) if coverage else 0.0,
                            dropped_segments=dropped_count,
                            validation_ok=validation_ok)
            
            return ValidationResult(
                ok=validation_ok,
                kept_text=kept_text,
                coverage=coverage,
                dropped_segments=dropped_count,
                scores_summary={
                    "coverage_ratio": sum(coverage) / len(coverage) if coverage else 0.0
                }
            )
            
        except Exception as e:
            if self.log:
                self.log.error("Validation failed", error=str(e))
            return ValidationResult(
                ok=False,
                kept_text=llm_text,  # Keep original on error
                coverage=[False] * len(self._segment_text(user_text)),
                dropped_segments=0,
                scores_summary={}
            )