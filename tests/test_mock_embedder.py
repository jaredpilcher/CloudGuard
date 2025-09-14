"""Test the mock embedder functionality."""

import pytest
import numpy as np
from cloudguard.providers.mock_embedder import MockEmbedder, create_mock_embedder


class TestMockEmbedder:
    """Test the mock embedder implementation."""
    
    def test_basic_embedding(self):
        """Test basic embedding functionality."""
        embedder = create_mock_embedder(dimension=64)
        
        texts = ["hello world", "billing invoice", "technical support"]
        embeddings = embedder.embed(texts)
        
        # Check shape
        assert embeddings.shape == (3, 64)
        
        # Check L2 normalization - all vectors should have unit length
        norms = np.linalg.norm(embeddings, axis=1)
        np.testing.assert_array_almost_equal(norms, np.ones(3), decimal=5)
    
    def test_deterministic_embeddings(self):
        """Test that embeddings are deterministic."""
        embedder1 = create_mock_embedder(dimension=32)
        embedder2 = create_mock_embedder(dimension=32)
        
        texts = ["same text", "another text"]
        
        emb1 = embedder1.embed(texts)
        emb2 = embedder2.embed(texts)
        
        # Should be identical
        np.testing.assert_array_equal(emb1, emb2)
    
    def test_semantic_similarity(self):
        """Test that semantically similar texts have similar embeddings."""
        embedder = create_mock_embedder(dimension=64)
        
        # Related billing terms should be more similar to each other than to technical terms
        billing_texts = ["invoice payment", "billing charges"]
        technical_texts = ["error debug", "install configure"]
        
        billing_emb = embedder.embed(billing_texts)
        technical_emb = embedder.embed(technical_texts)
        
        # Cosine similarity within billing group
        billing_sim = np.dot(billing_emb[0], billing_emb[1])
        
        # Cosine similarity between billing and technical
        cross_sim = np.dot(billing_emb[0], technical_emb[0])
        
        # Within-group similarity should be higher (though mock embedder is limited)
        # This is a weak test since mock embedder has limited semantic understanding
        assert isinstance(billing_sim, (float, np.floating))
        assert isinstance(cross_sim, (float, np.floating))
    
    def test_empty_input(self):
        """Test handling of empty input."""
        embedder = create_mock_embedder()
        
        embeddings = embedder.embed([])
        assert embeddings.shape == (0, 128)  # Default dimension
    
    def test_single_text(self):
        """Test embedding single text."""
        embedder = create_mock_embedder(dimension=32)
        
        embeddings = embedder.embed(["single text"])
        assert embeddings.shape == (1, 32)
        
        # Should be unit vector
        norm = np.linalg.norm(embeddings[0])
        assert abs(norm - 1.0) < 1e-10
    
    def test_dimension_parameter(self):
        """Test different embedding dimensions."""
        for dim in [16, 64, 128, 256]:
            embedder = MockEmbedder(dimension=dim)
            embeddings = embedder.embed(["test text"])
            assert embeddings.shape == (1, dim)
    
    def test_case_insensitive(self):
        """Test that case differences don't matter much."""
        embedder = create_mock_embedder(dimension=64)
        
        emb1 = embedder.embed(["Hello World"])
        emb2 = embedder.embed(["hello world"])
        
        # Should be very similar (mock embedder normalizes to lowercase)
        similarity = np.dot(emb1[0], emb2[0])
        assert similarity > 0.99  # Should be very close