"""Pydantic schemas for YAML policy validation."""

from pydantic import BaseModel, Field
from typing import List, Optional

class Thresholds(BaseModel):
    """Similarity thresholds for routing and validation decisions."""
    in_cloud: float = Field(default=0.80, ge=0.0, le=1.0, 
                           description="Cosine threshold to be considered 'inside' a region")
    margin: float = Field(default=0.05, ge=0.0, le=0.5,
                         description="Abstain band near boundaries (creates uncertainty zone)")

class RoutingCfg(BaseModel):
    """Routing configuration for handling abstain decisions."""
    abstain_action: str = Field(default="fallback", 
                               description="Action when abstaining: fallback|human|block|warn")
    default_target: Optional[str] = Field(default=None,
                                        description="Default routing target for abstain cases")

class Region(BaseModel):
    """Definition of a semantic region with routing configuration."""
    id: str = Field(description="Unique identifier for this region")
    label: str = Field(description="Human-readable label")
    seeds: List[str] = Field(description="Seed texts that define this region's semantic space")
    routes_to: Optional[str] = Field(default=None, description="Target to route to when in this region")
    
    class Config:
        extra = "forbid"  # Strict validation

class CloudPolicy(BaseModel):
    """Complete policy configuration for CloudGuard."""
    version: int = Field(default=1, description="Policy schema version")
    thresholds: Thresholds = Field(description="Similarity thresholds")
    routing: RoutingCfg = Field(description="Routing configuration")
    regions: List[Region] = Field(description="Semantic regions definition")
    
    class Config:
        extra = "forbid"  # Strict validation
        
    def validate_regions(self) -> List[str]:
        """Validate region configuration and return any issues."""
        issues = []
        
        # Check for duplicate region IDs
        ids = [r.id for r in self.regions]
        duplicates = set([x for x in ids if ids.count(x) > 1])
        if duplicates:
            issues.append(f"Duplicate region IDs: {duplicates}")
            
        # Check that each region has at least one seed
        empty_regions = [r.id for r in self.regions if not r.seeds]
        if empty_regions:
            issues.append(f"Regions with no seeds: {empty_regions}")
            
        # Validate routing targets exist or are None
        invalid_routes = []
        for region in self.regions:
            if region.routes_to and not region.routes_to.strip():
                invalid_routes.append(f"Region '{region.id}' has empty routes_to")
        if invalid_routes:
            issues.extend(invalid_routes)
            
        return issues