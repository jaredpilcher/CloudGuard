"""
Example 1: Router (Input Gate) with injected embeddings

Shows how to inject SentenceTransformers or OpenAI embeddings
into CloudGuard for semantic routing decisions.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from cloudguard.policy.loader import load_policy
from cloudguard.policy.index import build_region_index
from cloudguard.runtime.input_gate import InputCloudGate
from cloudguard.adapters.langgraph.nodes import make_input_gate_node
from cloudguard.adapters.langgraph.state_keys import USER_INPUT
import numpy as np

# Example A: SentenceTransformers (local)
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

# Note: OpenAI embeddings are now handled by the CloudGuard provider system
# See src/cloudguard/providers/openai_embedder.py for the improved implementation

# Simple console logger
class ConsoleLogger:
    def info(self, msg: str, **kv): print(f"INFO: {msg} {kv}")
    def warn(self, msg: str, **kv): print(f"WARN: {msg} {kv}")
    def error(self, msg: str, **kv): print(f"ERROR: {msg} {kv}")

def main():
    print("🔀 CloudGuard Router Example")
    print("=" * 40)
    
    # Try different embedding providers in priority order with runtime error handling:
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
        except ImportError as sbert_error:
            print(f"⚠️  SentenceTransformers not available: {sbert_error}")
        except Exception as sbert_error:
            print(f"⚠️  SentenceTransformers initialization failed: {sbert_error}")
    
    # Exit with error if no real embeddings available
    if embedder is None:
        print("❌ No real embedding provider available.")
        print("   Install sentence-transformers for local embeddings:")
        print("   pip install sentence-transformers")
        print("   Or configure OpenAI API key: export OPENAI_API_KEY=your_key")
        return
    
    # Load policy and build index
    policy_path = Path(__file__).parent / "clouds.yaml"
    policy = load_policy(policy_path)
    index = build_region_index(policy, embedder)
    
    # Create input gate
    logger = ConsoleLogger()
    input_gate = InputCloudGate(policy=policy, index=index, embedder=embedder, logger=logger)
    
    # Create LangGraph node (example usage)
    input_gate_node = make_input_gate_node(input_gate, text_key=USER_INPUT)
    
    # Test routing
    test_cases = [
        "I need help with my billing statement",
        "The app crashes when I try to login", 
        "How do I update my account settings?",
        "What's the capital of France?"  # Should abstain
    ]
    
    print("\n🧪 Testing routing decisions:")
    for query in test_cases:
        print(f"\n📝 Query: '{query}'")
        
        # Direct gate usage
        decision = input_gate.route(query)
        
        if decision.decision == "route":
            print(f"✅ ROUTE → {decision.target} (region: {decision.region_id}, score: {decision.score:.3f})")
        else:
            print(f"⚠️  ABSTAIN → {decision.target} (reason: {decision.reason}, score: {decision.score:.3f})")
        
        # LangGraph node usage example
        state = {USER_INPUT: query}
        node_result = input_gate_node.invoke(state)
        cloudguard_result = node_result["cloudguard_input"]
        print(f"🎯 Node result: {cloudguard_result['decision']} → {cloudguard_result.get('target', 'none')}")

if __name__ == "__main__":
    main()