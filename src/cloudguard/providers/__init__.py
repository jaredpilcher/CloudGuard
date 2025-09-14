"""
CloudGuard Providers Package

This package contains embedding providers and external service integrations
for CloudGuard. All providers follow CloudGuard's dependency injection pattern.
"""

from .openai_embedder import OpenAIEmbedder, create_openai_embedder

__all__ = ['OpenAIEmbedder', 'create_openai_embedder']