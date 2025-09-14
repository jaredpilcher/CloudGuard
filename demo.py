#!/usr/bin/env python3
"""
CloudGuard Demo - Shows input routing with real embeddings.
Demonstrates CloudGuard's dual-purpose capabilities: routing and validation.
"""

import sys
import numpy as np
from pathlib import Path

# Add src to path so we can import cloudguard
sys.path.insert(0, str(Path(__file__).parent / "src"))

from cloudguard.policy.loader import load_policy
from cloudguard.policy.index import build_region_index
from cloudguard.runtime.input_gate import InputCloudGate

# SentenceTransformers embeddings adapter
class SbertEmb:
    """Local SentenceTransformers embeddings adapter."""
    
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        try:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(model_name)
            print(f"✅ Loaded SentenceTransformers model: {model_name}")
        except ImportError:
            raise ImportError("Install sentence-transformers: pip install sentence-transformers")
    
    def embed(self, texts):
        embeddings = self.model.encode(texts)  # (N, D)
        # L2-normalize
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-12
        return embeddings / norms

class SimpleLogger:
    """Simple console logger for demo."""
    
    def info(self, msg: str, **kv):
        details = " ".join(f"{k}={v}" for k, v in kv.items())
        print(f"INFO: {msg} {details}")
        
    def warn(self, msg: str, **kv):
        details = " ".join(f"{k}={v}" for k, v in kv.items())
        print(f"WARN: {msg} {details}")
        
    def error(self, msg: str, **kv):
        details = " ".join(f"{k}={v}" for k, v in kv.items())
        print(f"ERROR: {msg} {details}")

def create_embedder():
    """Create the best available embedder with proper fallbacks."""
    
    # Try different embedding providers in priority order:
    # 1. OpenAI (best quality, requires API key and working API)
    # 2. SentenceTransformers (local, good quality)
    embedder = None
    
    # Try OpenAI first
    try:
        from cloudguard.providers.openai_embedder import create_openai_embedder, is_openai_available
        
        if is_openai_available():
            print("🔍 Testing OpenAI API connection...")
            test_embedder = create_openai_embedder()
            # Test with a small embedding to verify API is working
            try:
                test_result = test_embedder.embed(["test connection"])
                if test_result.shape[0] > 0 and test_result.shape[1] > 0:
                    embedder = test_embedder
                    print("✅ Using OpenAI embeddings for best quality semantic understanding")
                    return embedder
                else:
                    raise RuntimeError("OpenAI API returned empty or invalid results")
            except Exception as api_error:
                print(f"⚠️  OpenAI API test failed: {api_error}")
                print("🔄 Falling back to local embeddings...")
        else:
            print("⚠️  OpenAI not available (missing package or API key)")
    except Exception as import_error:
        print(f"⚠️  OpenAI setup failed: {import_error}")
    
    # Try SentenceTransformers if OpenAI failed
    if embedder is None:
        try:
            print("🔍 Initializing SentenceTransformers...")
            embedder = SbertEmb()  # Local option
            print("✅ Using SentenceTransformers embeddings for local processing")
            return embedder
        except ImportError as sbert_error:
            print(f"⚠️  SentenceTransformers not available: {sbert_error}")
        except Exception as sbert_error:
            print(f"⚠️  SentenceTransformers initialization failed: {sbert_error}")
    
    # Fall back to mock embedder for demo purposes
    if embedder is None:
        try:
            from cloudguard.providers.mock_embedder import create_mock_embedder
            print("🔍 Falling back to mock embedder for demo...")
            embedder = create_mock_embedder()
            print("✅ Using mock embedder for demo (limited accuracy)")
            return embedder
        except Exception as mock_error:
            print(f"⚠️  Mock embedder failed: {mock_error}")
    
    # Exit with error if no embeddings available at all
    print("❌ No embedding provider available.")
    print("   Install sentence-transformers for local embeddings:")
    print("   pip install sentence-transformers")
    print("   Or configure OpenAI API key: export OPENAI_API_KEY=your_key")
    raise RuntimeError("No embedding provider available")

def run_demo():
    """Run the CloudGuard demo."""
    print("🛡️  CloudGuard Demo - Semantic Routing")
    print("=" * 50)
    
    try:
        # Load policy
        print("📋 Loading policy from demo_policy.yaml...")
        policy = load_policy("demo_policy.yaml")
        print(f"✅ Loaded policy with {len(policy.regions)} regions")
        
        # Create embedder with real embeddings
        print("🧠 Creating embedder and building region index...")
        embedder = create_embedder()
        index = build_region_index(policy, embedder)
        print(f"✅ Built index with {len(index.protos)} region prototypes")
        
        # Create input gate
        logger = SimpleLogger()
        gate = InputCloudGate(policy=policy, index=index, embedder=embedder, logger=logger)
        
        # Test queries
        test_queries = [
            "I need help with my invoice and payment issues",
            "The software won't install properly on my computer", 
            "How do I reset my password and update my profile?",
            "What's the weather like today?",  # Should abstain
            "I love pizza and ice cream"      # Should abstain
        ]
        
        print("\n🧪 Testing routing decisions:")
        print("-" * 50)
        
        for i, query in enumerate(test_queries, 1):
            print(f"\n{i}. Query: \"{query}\"")
            decision = gate.route(query)
            
            if decision.decision == "route":
                print(f"   ✅ ROUTE -> {decision.target} (region: {decision.region_id})")
                print(f"   📊 Score: {decision.score:.3f}")
            else:
                print(f"   ⚠️  ABSTAIN -> {decision.target or 'none'}")
                print(f"   📊 Score: {decision.score:.3f}, Reason: {decision.reason}")
        
        print("\n🎉 Demo completed successfully!")
        
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(run_demo())