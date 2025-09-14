"""Test configuration and fixtures."""

import pytest
from pathlib import Path
import tempfile
import yaml

from cloudguard.providers.mock_embedder import create_mock_embedder
from cloudguard.policy.loader import load_policy_from_string


@pytest.fixture
def mock_embedder():
    """Provide a mock embedder for testing."""
    return create_mock_embedder(dimension=64)  # Smaller dimension for faster tests


@pytest.fixture
def sample_policy_yaml():
    """Provide a sample policy YAML for testing."""
    return """
version: 1
thresholds:
  in_cloud: 0.75
  margin: 0.1
routing:
  abstain_action: fallback
  default_target: general_agent
regions:
  - id: billing
    label: Billing Issues
    seeds:
      - "invoice billing payment charges"
      - "refunds credit card processing"
    routes_to: billing_agent
  - id: technical
    label: Technical Support
    seeds:
      - "error install troubleshoot debug"
      - "software bug configuration issue"
    routes_to: tech_agent
"""


@pytest.fixture
def sample_policy(sample_policy_yaml):
    """Provide a loaded policy object for testing."""
    return load_policy_from_string(sample_policy_yaml)


@pytest.fixture
def temp_policy_file(sample_policy_yaml):
    """Provide a temporary policy file for testing."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        f.write(sample_policy_yaml)
        temp_path = Path(f.name)
    
    yield temp_path
    
    # Cleanup
    if temp_path.exists():
        temp_path.unlink()


class SimpleTestLogger:
    """Simple logger for testing that captures messages."""
    
    def __init__(self):
        self.messages = []
    
    def info(self, msg: str, **kv):
        self.messages.append(('info', msg, kv))
    
    def warn(self, msg: str, **kv):
        self.messages.append(('warn', msg, kv))
    
    def error(self, msg: str, **kv):
        self.messages.append(('error', msg, kv))
    
    def clear(self):
        """Clear captured messages."""
        self.messages.clear()


@pytest.fixture
def test_logger():
    """Provide a test logger that captures messages."""
    return SimpleTestLogger()