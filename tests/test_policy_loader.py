"""Test policy loading and validation."""

import pytest
import tempfile
from pathlib import Path

from cloudguard.policy.loader import load_policy, load_policy_from_string, PolicyLoadError
from cloudguard.policy.schema import CloudPolicy, Thresholds, RoutingCfg, Region


class TestPolicyLoading:
    """Test policy loading from YAML files and strings."""
    
    def test_load_valid_policy_from_string(self, sample_policy_yaml):
        """Test loading valid policy from YAML string."""
        policy = load_policy_from_string(sample_policy_yaml)
        
        assert isinstance(policy, CloudPolicy)
        assert policy.version == 1
        assert policy.thresholds.in_cloud == 0.75
        assert policy.thresholds.margin == 0.1
        assert policy.routing.abstain_action == "fallback"
        assert len(policy.regions) == 2
        
        # Check first region
        billing_region = policy.regions[0]
        assert billing_region.id == "billing"
        assert billing_region.label == "Billing Issues"
        assert len(billing_region.seeds) == 2
        assert billing_region.routes_to == "billing_agent"
    
    def test_load_valid_policy_from_file(self, temp_policy_file):
        """Test loading valid policy from file."""
        policy = load_policy(temp_policy_file)
        
        assert isinstance(policy, CloudPolicy)
        assert policy.version == 1
        assert len(policy.regions) == 2
    
    def test_load_invalid_yaml(self):
        """Test loading invalid YAML."""
        invalid_yaml = """
        invalid: yaml: content:
          - missing: bracket
        """
        
        with pytest.raises(PolicyLoadError, match="Invalid YAML"):
            load_policy_from_string(invalid_yaml)
    
    def test_load_missing_required_fields(self):
        """Test loading policy with missing required fields."""
        incomplete_yaml = """
        version: 1
        # Missing thresholds, routing, and regions
        """
        
        with pytest.raises(PolicyLoadError, match="validation failed"):
            load_policy_from_string(incomplete_yaml)
    
    def test_load_invalid_threshold_values(self):
        """Test loading policy with invalid threshold values."""
        invalid_yaml = """
        version: 1
        thresholds:
          in_cloud: 1.5  # Invalid - should be <= 1.0
          margin: -0.1   # Invalid - should be >= 0.0
        routing:
          abstain_action: fallback
        regions: []
        """
        
        with pytest.raises(PolicyLoadError, match="validation failed"):
            load_policy_from_string(invalid_yaml)
    
    def test_load_duplicate_region_ids(self):
        """Test loading policy with duplicate region IDs."""
        duplicate_yaml = """
        version: 1
        thresholds:
          in_cloud: 0.8
          margin: 0.1
        routing:
          abstain_action: fallback
        regions:
          - id: billing
            label: Billing
            seeds: ["invoice"]
          - id: billing  # Duplicate ID
            label: Another Billing
            seeds: ["payment"]
        """
        
        with pytest.raises(PolicyLoadError, match="Duplicate region IDs"):
            load_policy_from_string(duplicate_yaml)
    
    def test_load_empty_region_seeds(self):
        """Test loading policy with empty region seeds."""
        empty_seeds_yaml = """
        version: 1
        thresholds:
          in_cloud: 0.8
          margin: 0.1
        routing:
          abstain_action: fallback
        regions:
          - id: billing
            label: Billing
            seeds: []  # Empty seeds
        """
        
        with pytest.raises(PolicyLoadError, match="no seeds"):
            load_policy_from_string(empty_seeds_yaml)
    
    def test_load_nonexistent_file(self):
        """Test loading from non-existent file."""
        nonexistent_path = Path("/does/not/exist.yaml")
        
        with pytest.raises(PolicyLoadError, match="not found"):
            load_policy(nonexistent_path)
    
    def test_load_extra_forbidden_fields(self):
        """Test that extra fields are forbidden."""
        extra_fields_yaml = """
        version: 1
        thresholds:
          in_cloud: 0.8
          margin: 0.1
        routing:
          abstain_action: fallback
        regions:
          - id: billing
            label: Billing
            seeds: ["invoice"]
        extra_field: not_allowed  # This should be forbidden
        """
        
        with pytest.raises(PolicyLoadError, match="validation failed"):
            load_policy_from_string(extra_fields_yaml)


class TestPolicySchema:
    """Test the policy schema validation."""
    
    def test_thresholds_validation(self):
        """Test threshold validation."""
        # Valid thresholds
        thresholds = Thresholds(in_cloud=0.8, margin=0.1)
        assert thresholds.in_cloud == 0.8
        assert thresholds.margin == 0.1
        
        # Test defaults
        default_thresholds = Thresholds()
        assert default_thresholds.in_cloud == 0.80
        assert default_thresholds.margin == 0.05
    
    def test_routing_config_validation(self):
        """Test routing configuration validation."""
        routing = RoutingCfg(abstain_action="human", default_target="human_agent")
        assert routing.abstain_action == "human"
        assert routing.default_target == "human_agent"
        
        # Test defaults
        default_routing = RoutingCfg()
        assert default_routing.abstain_action == "fallback"
        assert default_routing.default_target is None
    
    def test_region_validation(self):
        """Test region validation."""
        region = Region(
            id="test_region",
            label="Test Region",
            seeds=["test seed 1", "test seed 2"],
            routes_to="test_handler"
        )
        
        assert region.id == "test_region"
        assert region.label == "Test Region"
        assert len(region.seeds) == 2
        assert region.routes_to == "test_handler"
    
    def test_policy_coverage_ratio(self):
        """Test policy validation methods."""
        # This would test the validate_regions method
        # For now, we test basic policy structure
        policy_dict = {
            "version": 1,
            "thresholds": {"in_cloud": 0.8, "margin": 0.1},
            "routing": {"abstain_action": "fallback"},
            "regions": [
                {
                    "id": "test",
                    "label": "Test",
                    "seeds": ["test seed"]
                }
            ]
        }
        
        policy = CloudPolicy.model_validate(policy_dict)
        issues = policy.validate_regions()
        assert isinstance(issues, list)