# CloudGuard

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-passing-brightgreen.svg)](tests/)

**CloudGuard** is a pure plugin library for LangGraph that provides embedding-based guardrails and semantic routing capabilities. It operates as a dependency injection framework where the host application provides embeddings, segmenters, and other external services at runtime.

## Features

✨ **Semantic Input Routing**: Route user queries to appropriate handlers based on similarity to predefined regions  
🛡️ **Output Validation**: Ensure LLM responses stay on-topic and provide adequate coverage of user input  
🔌 **Plugin Architecture**: Pure dependency injection - works with any embedding provider  
📋 **Policy-Driven**: YAML-based configuration for semantic regions and routing rules  
🏗️ **LangGraph Integration**: Drop-in nodes for your LangGraph workflows  
🧪 **Testing Support**: Built-in mock embedder for offline development and testing  

## Quick Start

### Installation

```bash
pip install langgraph-cloudguard
```

### Basic Usage

```python
from cloudguard.policy.loader import load_policy
from cloudguard.policy.index import build_region_index
from cloudguard.runtime.input_gate import InputCloudGate
from cloudguard.providers.mock_embedder import create_mock_embedder

# Load policy from YAML
policy = load_policy("policy.yaml")

# Create embedder (use real embedder in production)
embedder = create_mock_embedder()

# Build semantic index
index = build_region_index(policy, embedder)

# Create input gate
input_gate = InputCloudGate(
    policy=policy,
    index=index, 
    embedder=embedder
)

# Route a query
decision = input_gate.route("I need help with my billing")
print(f"Route to: {decision.target}")  # -> billing_agent
```

### Policy Configuration

Create a `policy.yaml` file defining your semantic regions:

```yaml
version: 1

thresholds:
  in_cloud: 0.80      # Similarity threshold for routing 
  margin: 0.05        # Abstain margin for uncertain cases

routing:
  abstain_action: fallback
  default_target: general_agent

regions:
  - id: billing
    label: Billing & Invoices
    seeds:
      - "invoice billing payment charges refunds"
      - "subscription fees account balance"
    routes_to: billing_agent

  - id: technical
    label: Technical Support  
    seeds:
      - "error troubleshooting debug installation"
      - "software bugs configuration issues"
    routes_to: tech_agent
```

## Architecture

CloudGuard follows a **pure plugin model** where it assumes nothing about your AI infrastructure:

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Host Graph    │    │   CloudGuard    │    │   Policy YAML   │
│                 │    │                 │    │                 │
│ ┌─────────────┐ │    │ ┌─────────────┐ │    │ regions:        │
│ │ Embeddings  │─┼────┼─│ Input Gate  │ │    │ - billing       │
│ │ Provider    │ │    │ │             │ │    │ - technical     │
│ └─────────────┘ │    │ └─────────────┘ │    │ - account       │
│                 │    │                 │    │                 │
│ ┌─────────────┐ │    │ ┌─────────────┐ │    │ thresholds:     │
│ │ Segmenter   │─┼────┼─│ Output Gate │ │    │   in_cloud: 0.8 │
│ │ (Optional)  │ │    │ │             │ │    │   margin: 0.1   │
│ └─────────────┘ │    │ └─────────────┘ │    │                 │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

### Core Components

- **Input Gate**: Routes queries to handlers based on semantic similarity
- **Output Gate**: Validates LLM responses for coverage and relevance  
- **Policy Engine**: YAML-driven configuration with Pydantic validation
- **Embedding Abstraction**: Protocol-based interface for any embedding provider

## Usage Patterns

### 1. Input Routing

```python
from cloudguard.runtime.input_gate import InputCloudGate
from cloudguard.adapters.langgraph.nodes import make_input_gate_node

# Direct usage
decision = input_gate.route(user_query)
if decision.decision == 'route':
    # Send to specific handler
    handler = get_handler(decision.target)
else:
    # Handle abstain case
    handler = get_fallback_handler()

# LangGraph integration  
input_node = make_input_gate_node(input_gate, text_key="user_input")
```

### 2. Output Validation

```python
from cloudguard.runtime.output_gate import OutputCloudGate

# Create output gate
output_gate = OutputCloudGate(
    policy=policy,
    embedder=embedder,
    require_coverage=True,
    drop_offtopic=True
)

# Validate LLM response
result = output_gate.validate(user_input, llm_response)
if result.ok:
    return result.kept_text  # Filtered response
else:
    return generate_fallback_response()
```

### 3. Embedding Providers

CloudGuard supports any embedding provider through its protocol interface:

```python
# OpenAI embeddings (requires openai package)
from cloudguard.providers.openai_embedder import create_openai_embedder
embedder = create_openai_embedder()

# SentenceTransformers (requires sentence-transformers package)  
from sentence_transformers import SentenceTransformer
import numpy as np

class SbertEmbedder:
    def __init__(self):
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
    
    def embed(self, texts):
        embeddings = self.model.encode(texts)
        # L2 normalize
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-12
        return embeddings / norms

# Mock embedder for testing
from cloudguard.providers.mock_embedder import create_mock_embedder
embedder = create_mock_embedder()
```

## LangGraph Integration

### Node Integration

```python
from cloudguard.adapters.langgraph.nodes import make_input_gate_node, make_output_gate_node
from cloudguard.adapters.langgraph.state_keys import USER_INPUT, LLM_OUTPUT

# Create LangGraph nodes
input_node = make_input_gate_node(input_gate, text_key=USER_INPUT)
output_node = make_output_gate_node(output_gate, 
                                   input_key=USER_INPUT, 
                                   output_key=LLM_OUTPUT)

# Use in your graph
from langgraph import Graph

graph = Graph()
graph.add_node("route_input", input_node)  
graph.add_node("validate_output", output_node)
```

### State Management

CloudGuard provides standard state keys for consistent integration:

```python
from cloudguard.adapters.langgraph.state_keys import (
    USER_INPUT,           # "user_input"
    LLM_OUTPUT,          # "llm_output" 
    CLOUDGUARD_ROUTE,    # "cloudguard_route"
    CLOUDGUARD_OUTPUT    # "cloudguard_output"
)
```

## Examples

The repository includes comprehensive examples:

- **[Router Minimal](examples/01_router_minimal/)**: Basic input routing setup
- **[Output Guard](examples/02_output_guard/)**: Output validation and filtering  
- **[Cloud Discovery](examples/03_cloud_discovery/)**: Automatic semantic region discovery
- **[Sales Style Analysis](examples/04_sales_style/)**: Advanced policy generation and analysis

Run examples:
```bash
cd examples/01_router_minimal
python app.py
```

## Testing

CloudGuard includes a comprehensive test suite and mock embedder for development:

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=cloudguard

# Use mock embedder in your tests
from cloudguard.providers.mock_embedder import create_mock_embedder
embedder = create_mock_embedder(dimension=64)  # Smaller for faster tests
```

## Configuration

### Policy Schema

```yaml
version: 1                    # Schema version (required)

thresholds:
  in_cloud: 0.75             # Cosine similarity threshold (0.0-1.0)
  margin: 0.1                # Abstain margin for uncertainty (0.0-0.5)

routing:
  abstain_action: fallback   # fallback|human|block|warn
  default_target: general    # Where to route abstain cases

regions:
  - id: region_name         # Unique identifier  
    label: Display Name     # Human-readable name
    seeds:                  # Seed texts defining semantic space
      - "example text 1"
      - "example text 2"  
    routes_to: handler_id   # Target handler (optional)
```

### Environment Variables

```bash
# For OpenAI embeddings
export OPENAI_API_KEY=your_key

# For Azure OpenAI
export AZURE_OPENAI_KEY=your_key
export AZURE_OPENAI_ENDPOINT=your_endpoint
```

## Advanced Usage

### Custom Segmenters

Implement the `Segmenter` protocol for custom text segmentation:

```python
from cloudguard.core.abc import Segmenter

class CustomSegmenter:
    def segment(self, text: str) -> List[str]:
        # Your custom segmentation logic
        return text.split('. ')

output_gate = OutputCloudGate(
    policy=policy,
    embedder=embedder, 
    segmenter=CustomSegmenter()
)
```

### Logging and Metrics

CloudGuard supports structured logging and metrics collection:

```python
class MyLogger:
    def info(self, msg: str, **kv): 
        print(f"INFO: {msg}", kv)
    def warn(self, msg: str, **kv): 
        print(f"WARN: {msg}", kv)

class MyMeter:
    def inc(self, name: str, amount: int = 1, **tags): 
        # Increment counter
        pass
    def observe(self, name: str, value: float, **tags):
        # Record observation  
        pass

gate = InputCloudGate(
    policy=policy,
    index=index,
    embedder=embedder,
    logger=MyLogger(),
    meter=MyMeter()
)
```

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Run tests (`pytest`)
4. Commit changes (`git commit -m 'Add amazing feature'`)
5. Push to branch (`git push origin feature/amazing-feature`)
6. Open a Pull Request

### Development Setup

```bash
git clone https://github.com/your-org/cloudguard.git
cd cloudguard
pip install -e ".[test,examples]"
pytest
```

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Built for the LangGraph ecosystem
- Inspired by semantic routing and guardrail patterns
- Thanks to the open-source AI community

---

**Need Help?** 
- 📚 Check out the [examples](examples/) directory
- 🐛 [Report issues](https://github.com/your-org/cloudguard/issues)
- 💬 [Start a discussion](https://github.com/your-org/cloudguard/discussions)