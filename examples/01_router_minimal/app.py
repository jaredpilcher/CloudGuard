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

# Example B: OpenAI embeddings (API)
class OpenAIEmb:
    """OpenAI API embeddings adapter."""
    
    def __init__(self, model: str = "text-embedding-3-small"):
        try:
            import openai
            self.client = openai.OpenAI()  # ensure env OPENAI_API_KEY is set
            self.model = model
            print(f"✅ Initialized OpenAI embeddings: {model}")
        except ImportError:
            raise ImportError("Install openai: pip install openai")
    
    def embed(self, texts):
        response = self.client.embeddings.create(model=self.model, input=texts)
        embeddings = np.array([d.embedding for d in response.data], dtype=np.float32)
        # L2-normalize
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-12
        return embeddings / norms

# Simple console logger
class ConsoleLogger:
    def info(self, msg: str, **kv): print(f"INFO: {msg} {kv}")
    def warn(self, msg: str, **kv): print(f"WARN: {msg} {kv}")
    def error(self, msg: str, **kv): print(f"ERROR: {msg} {kv}")

def main():
    print("🔀 CloudGuard Router Example")
    print("=" * 40)
    
    # Choose embedder (comment out one)
    try:
        # For demo purposes, use a simple mock embedder if sentence-transformers not available
        try:
            embedder = SbertEmb()  # Local option
        except ImportError:
            print("📝 sentence-transformers not available, using mock embedder for demo")
            
            # Simple mock embedder for demo purposes
            class MockEmbedder:
                def __init__(self, dim=384):
                    self.dim = dim
                    self.word_features = {
                        "billing": np.array([1.0, 0.0, 0.0]), "invoice": np.array([0.95, 0.05, 0.0]), 
                        "payment": np.array([0.9, 0.1, 0.0]), "software": np.array([0.0, 1.0, 0.0]),
                        "install": np.array([0.05, 0.95, 0.0]), "tech": np.array([0.1, 0.9, 0.0]),
                        "support": np.array([0.15, 0.85, 0.0]), "error": np.array([0.2, 0.8, 0.0]),
                        "computer": np.array([0.1, 0.85, 0.05])
                    }
                def embed(self, texts):
                    embeddings = []
                    for text in texts:
                        text_lower = text.lower()
                        vec = np.random.normal(0, 0.05, self.dim)
                        main_feature = np.zeros(3)
                        feature_strength = 0.0
                        for word, feature in self.word_features.items():
                            if word in text_lower:
                                main_feature += feature * 2.0
                                feature_strength += 1.0
                        if feature_strength > 0:
                            vec[:3] = main_feature / max(feature_strength, 1.0)
                            vec[3:] *= 0.1
                        norm = np.linalg.norm(vec)
                        if norm > 1e-12:
                            vec = vec / norm
                        embeddings.append(vec)
                    return np.array(embeddings)
                    
            embedder = MockEmbedder()
        # embedder = OpenAIEmb()  # API option (requires OPENAI_API_KEY)
    except Exception as e:
        print(f"❌ {e}")
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