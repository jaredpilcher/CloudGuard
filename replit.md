# Overview

CloudGuard is a pure plugin library for LangGraph that provides embedding-based guardrails and semantic routing capabilities. It operates as a dependency injection framework where the host application provides embeddings, segmenters, and other external services at runtime. The system enables input routing based on semantic similarity to predefined regions and output validation to ensure LLM responses stay on-topic and provide adequate coverage of user input.

# User Preferences

Preferred communication style: Simple, everyday language.

# System Architecture

## Core Design Principles

**Dependency Injection Architecture**: CloudGuard follows a pure plugin model where it assumes nothing about embeddings or LLM providers. All external dependencies (embeddings, segmenters, loggers) are injected at runtime through protocol interfaces, allowing flexibility in choosing local models (SentenceTransformers) or API services (OpenAI, Cohere).

**Policy-Driven Configuration**: The system uses YAML-based policies that define semantic regions through seed texts, routing targets, and similarity thresholds. Policies are validated using Pydantic schemas and compiled into embedding-based indices for efficient runtime lookups.

**Dual Gate System**: 
- **Input Gate**: Routes incoming queries to appropriate handlers based on semantic similarity to predefined regions
- **Output Gate**: Validates LLM responses for topic coverage and filters off-topic segments

## Data Layer

**Policy Storage**: YAML files define semantic regions with seed texts, routing configuration, and similarity thresholds. No database required - policies are loaded and compiled into in-memory indices.

**Vector Operations**: Uses numpy for efficient cosine similarity calculations between L2-normalized embeddings. Region centroids are computed from seed text embeddings during index building.

## Integration Layer

**LangGraph Adapter**: Provides node factories that wrap CloudGuard gates into LangGraph-compatible runnables. Uses standardized state keys for input/output coordination.

**Protocol-Based Interfaces**: Defines abstract protocols for Embeddings, Segmenter, and Logger allowing clean integration with various providers without tight coupling.

## Runtime Components

**Region Indexing**: Builds semantic region prototypes with embedding centroids from policy seed texts. Supports efficient best-match queries using cosine similarity.

**Routing Logic**: Input gate embeds queries and finds best matching regions, with configurable abstain margins for uncertain cases. Supports fallback routing for abstain decisions.

**Validation Engine**: Output gate segments both user input and LLM responses, checks coverage requirements, and optionally filters off-topic segments based on embedding similarity.

# External Dependencies

## Core Dependencies
- **Pydantic**: Schema validation for YAML policies and data structures
- **NumPy**: Vector operations and cosine similarity calculations
- **LangChain Core**: Integration with LangGraph execution framework
- **PyYAML**: Policy file parsing and configuration loading

## Optional Runtime Dependencies
- **SentenceTransformers**: Local embedding models (injected by host)
- **OpenAI API**: Cloud-based embeddings and LLM services (injected by host)
- **Custom Segmenters**: Text segmentation providers (optional, fallback included)

## Development Dependencies
- **Pytest**: Testing framework with coverage reporting
- **Example Applications**: Demonstration scripts showing various embedding provider integrations

The architecture ensures CloudGuard remains provider-agnostic while supporting both local and cloud-based AI services through its dependency injection system.