"""Small utility functions."""

import hashlib
import json
from typing import Any, Dict

def hash_text(text: str) -> str:
    """Create a stable hash of text content."""
    return hashlib.sha256(text.encode('utf-8')).hexdigest()[:16]

def safe_json(obj: Any) -> str:
    """Safely serialize object to JSON, handling numpy types."""
    def serialize_item(item):
        if hasattr(item, 'item'):  # numpy scalar
            return item.item()
        elif hasattr(item, 'tolist'):  # numpy array
            return item.tolist()
        elif hasattr(item, '__dict__'):  # dataclass or object
            return {k: serialize_item(v) for k, v in item.__dict__.items()}
        elif isinstance(item, (list, tuple)):
            return [serialize_item(x) for x in item]
        elif isinstance(item, dict):
            return {k: serialize_item(v) for k, v in item.items()}
        else:
            return item
    
    try:
        return json.dumps(serialize_item(obj), indent=2)
    except Exception as e:
        return f"<serialization error: {e}>"