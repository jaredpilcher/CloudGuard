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
from cloudguard.providers.openai_embedder import create_openai_embedder, is_openai_available


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
        embedder = None
        
        # Try OpenAI first with runtime testing
        if is_openai_available():
            print("🔍 Testing OpenAI API connection...")
            try:
                test_embedder = create_openai_embedder()
                # Test with a small embedding to verify API is working
                test_result = test_embedder.embed(["test connection"])
                if test_result.shape[0] > 0 and test_result.shape[1] > 0:
                    embedder = test_embedder
                    print("✅ Using OpenAI embeddings for real semantic understanding")
                else:
                    raise RuntimeError("OpenAI API returned empty results")
            except Exception as api_error:
                print(f"⚠️  OpenAI API test failed: {api_error}")
        else:
            print("⚠️  OpenAI embeddings unavailable (missing package or API key)")
        
        # No fallback - require real embeddings
        if embedder is None:
            print("❌ No real embedding provider available.")
            print("   Install sentence-transformers for local embeddings:")
            print("   pip install sentence-transformers")
            print("   Or configure OpenAI API key: export OPENAI_API_KEY=your_key")
            return 1
        
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