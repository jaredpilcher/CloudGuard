"""Test the output gate validation functionality."""

import pytest
from cloudguard.runtime.output_gate import OutputCloudGate
from cloudguard.core.types import ValidationResult


class TestOutputGate:
    """Test output gate validation functionality."""
    
    def test_output_gate_initialization(self, sample_policy, mock_embedder, test_logger):
        """Test output gate initialization."""
        gate = OutputCloudGate(
            policy=sample_policy,
            embedder=mock_embedder,
            logger=test_logger,
            require_coverage=True,
            drop_offtopic=True
        )
        
        assert gate.policy == sample_policy
        assert gate.embedder == mock_embedder
        assert gate.log == test_logger
        assert gate.require_coverage is True
        assert gate.drop_offtopic is True
    
    def test_validation_result_structure(self, sample_policy, mock_embedder, test_logger):
        """Test that validation results have correct structure."""
        gate = OutputCloudGate(
            policy=sample_policy,
            embedder=mock_embedder,
            logger=test_logger
        )
        
        result = gate.validate("test user input", "test llm output")
        
        assert isinstance(result, ValidationResult)
        assert hasattr(result, 'ok')
        assert hasattr(result, 'kept_text')
        assert hasattr(result, 'coverage')
        assert hasattr(result, 'dropped_segments')
        assert hasattr(result, 'scores_summary')
        
        assert isinstance(result.ok, bool)
        assert isinstance(result.kept_text, str)
        assert isinstance(result.coverage, list)
        assert isinstance(result.dropped_segments, int)
        assert result.dropped_segments >= 0
    
    def test_coverage_ratio_property(self, sample_policy, mock_embedder):
        """Test the coverage_ratio property calculation."""
        gate = OutputCloudGate(policy=sample_policy, embedder=mock_embedder)
        
        result = gate.validate(
            "Please help with billing",
            "I can help you with your billing questions and payment issues."
        )
        
        assert hasattr(result, 'coverage_ratio')
        coverage_ratio = result.coverage_ratio
        
        assert isinstance(coverage_ratio, float)
        assert 0.0 <= coverage_ratio <= 1.0
        
        if result.coverage:
            expected_ratio = sum(result.coverage) / len(result.coverage)
            assert abs(coverage_ratio - expected_ratio) < 1e-10
    
    def test_relevant_output_validation(self, sample_policy, mock_embedder, test_logger):
        """Test validation of relevant output."""
        gate = OutputCloudGate(
            policy=sample_policy,
            embedder=mock_embedder,
            logger=test_logger,
            require_coverage=False  # Don't require perfect coverage for this test
        )
        
        user_input = "I need help with my invoice"
        llm_output = "I can help you with billing and invoice questions"
        
        result = gate.validate(user_input, llm_output)
        
        assert isinstance(result, ValidationResult)
        assert len(result.coverage) >= 1  # Should have at least one input segment
        assert isinstance(result.ok, bool)
    
    def test_off_topic_output_filtering(self, sample_policy, mock_embedder, test_logger):
        """Test filtering of off-topic output segments."""
        gate = OutputCloudGate(
            policy=sample_policy,
            embedder=mock_embedder,
            logger=test_logger,
            drop_offtopic=True,
            require_coverage=False
        )
        
        user_input = "Help with billing"
        llm_output = "I can help with billing. Also, the weather is nice today."
        
        result = gate.validate(user_input, llm_output)
        
        assert isinstance(result, ValidationResult)
        # With drop_offtopic=True, may filter some content
        assert isinstance(result.kept_text, str)
    
    def test_empty_input_handling(self, sample_policy, mock_embedder, test_logger):
        """Test handling of empty user input."""
        gate = OutputCloudGate(
            policy=sample_policy,
            embedder=mock_embedder,
            logger=test_logger
        )
        
        result = gate.validate("", "Some LLM output")
        
        assert isinstance(result, ValidationResult)
        assert result.ok is True  # Empty input should be allowed
        assert result.kept_text == "Some LLM output"
    
    def test_empty_output_handling(self, sample_policy, mock_embedder, test_logger):
        """Test handling of empty LLM output."""
        gate = OutputCloudGate(
            policy=sample_policy,
            embedder=mock_embedder,
            logger=test_logger
        )
        
        result = gate.validate("Some user input", "")
        
        assert isinstance(result, ValidationResult)
        assert result.ok is False  # Empty output should fail validation
        assert result.kept_text == ""
        assert len(result.coverage) >= 1  # Should have coverage info for user input
        assert all(not covered for covered in result.coverage)  # No coverage
    
    def test_require_coverage_true(self, sample_policy, mock_embedder, test_logger):
        """Test validation with require_coverage=True."""
        gate = OutputCloudGate(
            policy=sample_policy,
            embedder=mock_embedder,
            logger=test_logger,
            require_coverage=True
        )
        
        result = gate.validate("user input", "llm output")
        
        assert isinstance(result, ValidationResult)
        # With mock embedder, coverage may not be perfect
        # But validation should complete successfully
    
    def test_require_coverage_false(self, sample_policy, mock_embedder, test_logger):
        """Test validation with require_coverage=False."""
        gate = OutputCloudGate(
            policy=sample_policy,
            embedder=mock_embedder,
            logger=test_logger,
            require_coverage=False
        )
        
        result = gate.validate("user input", "llm output")
        
        assert isinstance(result, ValidationResult)
        # With require_coverage=False, should be more lenient
    
    def test_drop_offtopic_false(self, sample_policy, mock_embedder, test_logger):
        """Test validation with drop_offtopic=False."""
        gate = OutputCloudGate(
            policy=sample_policy,
            embedder=mock_embedder,
            logger=test_logger,
            drop_offtopic=False
        )
        
        user_input = "billing help"
        llm_output = "I can help with billing. The weather is nice."
        
        result = gate.validate(user_input, llm_output)
        
        assert isinstance(result, ValidationResult)
        # With drop_offtopic=False, should keep all output
        assert result.kept_text == llm_output
        assert result.dropped_segments == 0
    
    def test_segmentation_functionality(self, sample_policy, mock_embedder, test_logger):
        """Test text segmentation works correctly."""
        gate = OutputCloudGate(
            policy=sample_policy,
            embedder=mock_embedder,
            logger=test_logger
        )
        
        # Multi-sentence input and output
        user_input = "I need help with billing. Can you assist with invoices?"
        llm_output = "Yes, I can help with billing. I also handle invoice questions. Let me know what you need."
        
        result = gate.validate(user_input, llm_output)
        
        assert isinstance(result, ValidationResult)
        assert len(result.coverage) >= 2  # Should detect multiple user segments
    
    def test_logging_during_validation(self, sample_policy, mock_embedder, test_logger):
        """Test that validation logs appropriately."""
        gate = OutputCloudGate(
            policy=sample_policy,
            embedder=mock_embedder,
            logger=test_logger
        )
        
        test_logger.clear()
        gate.validate("test input", "test output")
        
        # Should have logged validation info
        assert len(test_logger.messages) > 0
        log_levels = [msg[0] for msg in test_logger.messages]
        assert 'info' in log_levels