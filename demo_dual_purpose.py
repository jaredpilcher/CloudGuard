#!/usr/bin/env python3
"""
CloudGuard Dual Purpose Demo - Shows input gates for both routing AND cloud validation.

Demonstrates:
1. ROUTING: Route specific queries to targeted agents
2. VALIDATION: Check if any input is "within the cloud" (valid domain) vs out-of-scope
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
    """Enhanced mock embedder for realistic routing demo."""
    
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
        print(f"    [LOG] INFO: {msg} {details}")
        
    def warn(self, msg: str, **kv):
        details = " ".join(f"{k}={v}" for k, v in kv.items())
        print(f"    [LOG] WARN: {msg} {details}")
        
    def error(self, msg: str, **kv):
        details = " ".join(f"{k}={v}" for k, v in kv.items())
        print(f"    [LOG] ERROR: {msg} {details}")

def demo_routing_use_case(gate: InputCloudGate):
    """Demonstrate using input gate for ROUTING to specific agents."""
    print("\n🔀 USE CASE 1: ROUTING - Direct queries to specific agents")
    print("=" * 60)
    
    routing_queries = [
        "I need help with my invoice and payment issues",
        "The software won't install properly on my computer", 
        "How do I reset my password and update my profile?"
    ]
    
    for i, query in enumerate(routing_queries, 1):
        print(f"\n{i}. Query: \"{query}\"")
        decision = gate.route(query)
        
        if decision.decision == "route":
            print(f"   ✅ ROUTE → {decision.target}")
            print(f"   🎯 Region: {decision.region_id}")
            print(f"   📊 Confidence: {decision.score:.1%}")
            print(f"   💡 Action: Forward to specialized {decision.target}")
        else:
            print(f"   ⚠️  ABSTAIN → {decision.target or 'fallback'}")
            print(f"   📊 Score: {decision.score:.3f}")

def demo_cloud_validation_use_case(gate: InputCloudGate):
    """Demonstrate using input gate for CLOUD VALIDATION (in-scope vs out-of-scope)."""
    print("\n🛡️  USE CASE 2: CLOUD VALIDATION - Check if input is within business scope")
    print("=" * 70)
    
    mixed_queries = [
        # In-scope queries (should be in cloud)
        "I have a billing question",
        "Need technical support", 
        "Account settings help",
        # Out-of-scope queries (should be out of cloud)
        "What's the weather like today?",
        "Tell me a joke about cats",
        "What's the capital of France?",
        "I love pizza and ice cream"
    ]
    
    in_cloud_count = 0
    out_of_cloud_count = 0
    
    for i, query in enumerate(mixed_queries, 1):
        print(f"\n{i}. Query: \"{query}\"")
        decision = gate.route(query)
        
        # Use the decision for cloud validation
        is_in_cloud = decision.decision == "route" or decision.score >= gate.policy.thresholds.in_cloud
        
        if is_in_cloud:
            in_cloud_count += 1
            print(f"   ✅ IN CLOUD - Valid business query (score: {decision.score:.3f})")
            print(f"   💡 Action: Process normally, route to {decision.target or 'appropriate handler'}")
        else:
            out_of_cloud_count += 1
            print(f"   ❌ OUT OF CLOUD - Out of scope (score: {decision.score:.3f})")
            print(f"   💡 Action: Handle with fallback/disclaimer or redirect")
    
    print(f"\n📊 SUMMARY:")
    print(f"   • In-scope queries: {in_cloud_count}")
    print(f"   • Out-of-scope queries: {out_of_cloud_count}")
    print(f"   • Cloud coverage: {in_cloud_count}/{len(mixed_queries)} ({in_cloud_count/len(mixed_queries):.1%})")

def demo_combined_workflow(gate: InputCloudGate):
    """Demonstrate a combined workflow using both routing and validation."""
    print("\n🔄 USE CASE 3: COMBINED WORKFLOW - Routing + Validation in production")
    print("=" * 65)
    
    def process_user_input(query: str) -> str:
        """Simulate a production workflow using CloudGuard for both purposes."""
        print(f"\n📝 Processing: \"{query}\"")
        
        decision = gate.route(query)
        
        # First: Cloud validation (is this a valid business query?)
        is_valid_domain = decision.score >= (gate.policy.thresholds.in_cloud - gate.policy.thresholds.margin)
        
        if not is_valid_domain:
            return f"🚫 OUT OF SCOPE: Sorry, I can only help with billing, technical support, and account issues. (Score: {decision.score:.3f})"
        
        # Second: Routing decision (which specific agent?)
        if decision.decision == "route":
            return f"✅ ROUTED to {decision.target}: I'll connect you with our {decision.region_id} specialist. (Score: {decision.score:.3f})"
        else:
            return f"🤔 IN SCOPE but uncertain routing: Let me connect you with our general support team who can help determine the best specialist. (Score: {decision.score:.3f})"
    
    test_inputs = [
        "My credit card was charged twice",
        "Software keeps crashing",
        "Forgot my login password", 
        "What's your company address?",  # Edge case - might be business related
        "How to make pancakes?"          # Clearly out of scope
    ]
    
    for query in test_inputs:
        response = process_user_input(query)
        print(f"   Response: {response}")

def run_demo():
    """Run the dual-purpose CloudGuard demo."""
    print("🛡️  CloudGuard Dual Purpose Demo")
    print("Routing + Cloud Validation")
    print("=" * 50)
    
    try:
        # Load policy and create gate
        print("📋 Loading policy...")
        policy = load_policy("demo_policy.yaml")
        
        print("🧠 Building region index...")
        embedder = MockEmbedder()
        index = build_region_index(policy, embedder)
        
        print("🔧 Creating input gate...")
        logger = SimpleLogger()
        gate = InputCloudGate(policy=policy, index=index, embedder=embedder, logger=logger)
        
        print(f"✅ Setup complete! Threshold: {policy.thresholds.in_cloud}, Margin: {policy.thresholds.margin}")
        
        # Run the three demo scenarios
        demo_routing_use_case(gate)
        demo_cloud_validation_use_case(gate)
        demo_combined_workflow(gate)
        
        print("\n🎉 Dual-purpose demo completed successfully!")
        print("\n💡 Key Takeaway: CloudGuard input gates serve dual purposes:")
        print("   1. ROUTING: Direct specific queries to appropriate agents")
        print("   2. VALIDATION: Determine if queries are within business scope")
        
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(run_demo())