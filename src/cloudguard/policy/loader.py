"""YAML policy loading and validation."""

import yaml
from pathlib import Path
from typing import Union
from .schema import CloudPolicy

class PolicyLoadError(Exception):
    """Exception raised when policy loading or validation fails."""
    pass

def load_policy(path: Union[str, Path]) -> CloudPolicy:
    """
    Load and validate a CloudGuard policy from YAML file.
    
    Args:
        path: Path to YAML policy file
        
    Returns:
        CloudPolicy: Validated policy object
        
    Raises:
        PolicyLoadError: If file cannot be read or policy is invalid
    """
    path = Path(path)
    
    if not path.exists():
        raise PolicyLoadError(f"Policy file not found: {path}")
        
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
    except yaml.YAMLError as e:
        raise PolicyLoadError(f"Invalid YAML in {path}: {e}")
    except Exception as e:
        raise PolicyLoadError(f"Cannot read policy file {path}: {e}")
        
    if not isinstance(data, dict):
        raise PolicyLoadError(f"Policy file must contain a YAML mapping, got {type(data)}")
        
    try:
        policy = CloudPolicy.model_validate(data)
    except Exception as e:
        raise PolicyLoadError(f"Policy validation failed: {e}")
        
    # Run additional validation
    issues = policy.validate_regions()
    if issues:
        raise PolicyLoadError(f"Policy validation issues: {'; '.join(issues)}")
        
    return policy

def load_policy_from_string(yaml_content: str) -> CloudPolicy:
    """
    Load and validate a CloudGuard policy from YAML string.
    
    Args:
        yaml_content: YAML content as string
        
    Returns:
        CloudPolicy: Validated policy object
        
    Raises:
        PolicyLoadError: If YAML is invalid or policy validation fails
    """
    try:
        data = yaml.safe_load(yaml_content)
    except yaml.YAMLError as e:
        raise PolicyLoadError(f"Invalid YAML content: {e}")
        
    if not isinstance(data, dict):
        raise PolicyLoadError(f"Policy content must contain a YAML mapping, got {type(data)}")
        
    try:
        policy = CloudPolicy.model_validate(data)
    except Exception as e:
        raise PolicyLoadError(f"Policy validation failed: {e}")
        
    # Run additional validation
    issues = policy.validate_regions()
    if issues:
        raise PolicyLoadError(f"Policy validation issues: {'; '.join(issues)}")
        
    return policy