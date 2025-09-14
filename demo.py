#!/usr/bin/env python3
"""
CloudGuard Demo - Shows input routing without external dependencies.
Uses a simple mock embedder for demonstration purposes.
"""

import sys
import numpy as np
from pathlib import Path

# Add src to path so we can import cloudguard
sys.path.insert(0, str(Path(__file__).parent / "src"))

from cloudguard.policy.loader import load_policy
from cloudguard.policy.index import build_region_index
from cloudguard.runtime.input_gate import InputCloudGate

class MockEmbedder:
    """Simple mock embedder for demo purposes - generates consistent, realistic vectors."""
    
    def __init__(self, dim: int = 384):
        self.dim = dim
        # Enhanced word-based features for better demo results
        self.word_features = {
            # Billing cluster
            "billing": np.array([1.0, 0.0, 0.0]),
            "invoice": np.array([0.95, 0.05, 0.0]), 
            "payment": np.array([0.9, 0.1, 0.0]),
            "refund": np.array([0.85, 0.15, 0.0]),
            "charge": np.array([0.8, 0.2, 0.0]),
            "balance": np.array([0.75, 0.25, 0.0]),
            
            # Tech support cluster  
            "software": np.array([0.0, 1.0, 0.0]),
            "install": np.array([0.05, 0.95, 0.0]),
            "tech": np.array([0.1, 0.9, 0.0]),
            "support": np.array([0.15, 0.85, 0.0]),
            "error": np.array([0.2, 0.8, 0.0]),
            "computer": np.array([0.1, 0.85, 0.05]),
            "troubleshoot": np.array([0.05, 0.9, 0.05]),
            
            # Account cluster
            "account": np.array([0.0, 0.0, 1.0]),
            "profile": np.array([0.05, 0.05, 0.9]),
            "password": np.array([0.1, 0.1, 0.8]),
            "login": np.array([0.15, 0.15, 0.7]),
            "reset": np.array([0.1, 0.2, 0.7]),
            "update": np.array([0.05, 0.15, 0.8])
        }
        
    def embed(self, texts):
        """Create embeddings with strong word-based features for realistic demo."""
        embeddings = []
        for text in texts:
            text_lower = text.lower()
            
            # Start with small random base to add some noise
            vec = np.random.normal(0, 0.05, self.dim)
            
            # Add strong word features if found
            feature_strength = 0.0
            main_feature = np.zeros(3)
            
            for word, feature in self.word_features.items():
                if word in text_lower:
                    main_feature += feature * 2.0  # Strong feature signal
                    feature_strength += 1.0
            
            # If we found relevant words, make the signal dominant
            if feature_strength > 0:
                vec[:3] = main_feature / max(feature_strength, 1.0)
                # Add some coherent signal to rest of vector
                vec[3:] *= 0.1  # Reduce noise in other dimensions
            
            # Normalize to unit length
            norm = np.linalg.norm(vec)
            if norm > 1e-12:
                vec = vec / norm
            embeddings.append(vec)
            
        return np.array(embeddings)

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

def run_demo():
    """Run the CloudGuard demo."""
    print("🛡️  CloudGuard Demo - Semantic Routing")
    print("=" * 50)
    
    try:
        # Load policy
        print("📋 Loading policy from demo_policy.yaml...")
        policy = load_policy("demo_policy.yaml")
        print(f"✅ Loaded policy with {len(policy.regions)} regions")
        
        # Create embedder and build index
        print("🧠 Creating embedder and building region index...")
        embedder = MockEmbedder()
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