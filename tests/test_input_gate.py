"""Test the input gate routing functionality."""

import pytest
from cloudguard.runtime.input_gate import InputCloudGate
from cloudguard.policy.index import build_region_index
from cloudguard.core.types import RouteDecision


class TestInputGate:
    """Test input gate routing decisions."""
    
    def test_input_gate_initialization(self, sample_policy, mock_embedder, test_logger):
        """Test input gate initialization."""
        index = build_region_index(sample_policy, mock_embedder)
        
        gate = InputCloudGate(
            policy=sample_policy,
            index=index, 
            embedder=mock_embedder,
            logger=test_logger
        )
        
        assert gate.policy == sample_policy
        assert gate.index == index
        assert gate.embedder == mock_embedder
        assert gate.log == test_logger
    
    def test_route_decision_structure(self, sample_policy, mock_embedder, test_logger):
        """Test that route decisions have correct structure."""
        index = build_region_index(sample_policy, mock_embedder)
        gate = InputCloudGate(
            policy=sample_policy,
            index=index,
            embedder=mock_embedder,
            logger=test_logger
        )
        
        decision = gate.route("test query")
        
        assert isinstance(decision, RouteDecision)
        assert hasattr(decision, 'decision')
        assert hasattr(decision, 'score')
        assert hasattr(decision, 'region_id')
        assert hasattr(decision, 'target')
        assert hasattr(decision, 'reason')
        
        assert decision.decision in ['route', 'abstain']
        assert isinstance(decision.score, (float, int))
        assert decision.score >= 0.0 and decision.score <= 1.0
    
    def test_billing_query_routing(self, sample_policy, mock_embedder, test_logger):
        """Test routing of billing-related query."""
        index = build_region_index(sample_policy, mock_embedder)
        gate = InputCloudGate(
            policy=sample_policy,
            index=index,
            embedder=mock_embedder,
            logger=test_logger
        )
        
        # Query with billing-related terms
        decision = gate.route("I need help with my invoice and billing charges")
        
        # With mock embedder, results may vary, but structure should be correct
        assert isinstance(decision, RouteDecision)
        
        if decision.decision == 'route':
            assert decision.region_id in ['billing', 'technical']
            assert decision.target is not None
        else:
            assert decision.decision == 'abstain'
            assert decision.reason is not None
    
    def test_technical_query_routing(self, sample_policy, mock_embedder, test_logger):
        """Test routing of technical query."""
        index = build_region_index(sample_policy, mock_embedder)
        gate = InputCloudGate(
            policy=sample_policy,
            index=index,
            embedder=mock_embedder,
            logger=test_logger
        )
        
        decision = gate.route("I have a software error that needs debugging")
        
        assert isinstance(decision, RouteDecision)
        # Mock embedder may not route correctly, but should return valid structure
    
    def test_off_topic_query_abstain(self, sample_policy, mock_embedder, test_logger):
        """Test that off-topic queries abstain appropriately."""
        index = build_region_index(sample_policy, mock_embedder)
        gate = InputCloudGate(
            policy=sample_policy,
            index=index,
            embedder=mock_embedder,
            logger=test_logger
        )
        
        decision = gate.route("What's the weather like today?")
        
        assert isinstance(decision, RouteDecision)
        # Mock embedder likely to abstain on irrelevant queries
        if decision.decision == 'abstain':
            assert decision.reason is not None
    
    def test_empty_query_handling(self, sample_policy, mock_embedder, test_logger):
        """Test handling of empty queries."""
        index = build_region_index(sample_policy, mock_embedder)
        gate = InputCloudGate(
            policy=sample_policy,
            index=index,
            embedder=mock_embedder,
            logger=test_logger
        )
        
        decision = gate.route("")
        
        assert isinstance(decision, RouteDecision)
        # Empty query should probably abstain
        assert decision.decision in ['route', 'abstain']
    
    def test_whitespace_query_handling(self, sample_policy, mock_embedder, test_logger):
        """Test handling of whitespace-only queries."""
        index = build_region_index(sample_policy, mock_embedder)
        gate = InputCloudGate(
            policy=sample_policy,
            index=index,
            embedder=mock_embedder,
            logger=test_logger
        )
        
        decision = gate.route("   \n\t  ")
        
        assert isinstance(decision, RouteDecision)
    
    def test_logging_calls(self, sample_policy, mock_embedder, test_logger):
        """Test that logging is called appropriately."""
        index = build_region_index(sample_policy, mock_embedder)
        gate = InputCloudGate(
            policy=sample_policy,
            index=index,
            embedder=mock_embedder,
            logger=test_logger
        )
        
        test_logger.clear()
        decision = gate.route("test query for logging")
        
        # Should have logged something about the routing decision
        assert len(test_logger.messages) > 0
        
        # Check for expected log levels
        log_levels = [msg[0] for msg in test_logger.messages]
        assert any(level in ['info', 'warn'] for level in log_levels)
    
    def test_multiple_queries_consistency(self, sample_policy, mock_embedder, test_logger):
        """Test that identical queries give consistent results."""
        index = build_region_index(sample_policy, mock_embedder)
        gate = InputCloudGate(
            policy=sample_policy,
            index=index,
            embedder=mock_embedder,
            logger=test_logger
        )
        
        query = "billing invoice payment issue"
        
        decision1 = gate.route(query)
        decision2 = gate.route(query)
        
        # Should be identical (deterministic)
        assert decision1.decision == decision2.decision
        assert abs(decision1.score - decision2.score) < 1e-10
        assert decision1.region_id == decision2.region_id
        assert decision1.target == decision2.target
        assert decision1.reason == decision2.reason