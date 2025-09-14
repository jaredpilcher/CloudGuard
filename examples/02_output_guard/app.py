"""
Example 2: Output Guard with multiple segmenter options

Shows how to inject embeddings and segmenters into CloudGuard
for LLM output validation and filtering.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from cloudguard.policy.loader import load_policy
from cloudguard.runtime.output_gate import OutputCloudGate
from cloudguard.adapters.langgraph.nodes import make_output_gate_node
from cloudguard.adapters.langgraph.state_keys import USER_INPUT, LLM_OUTPUT
from cloudguard.segmenters.sentence import SentenceSegmenter
import numpy as np

# Embeddings (same as Example 1)
class SbertEmb:
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        try:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(model_name)
            print(f"✅ Loaded SentenceTransformers model: {model_name}")
        except ImportError:
            raise ImportError("Install sentence-transformers: pip install sentence-transformers")
    
    def embed(self, texts):
        embeddings = self.model.encode(texts)
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-12
        return embeddings / norms

# Segmenter option A: deterministic (no deps)
class SimpleSentenceSegmenter:
    def segment(self, text: str):
        import re
        sentences = re.split(r'(?<=[.?!])\s+', text.strip())
        return [s.strip() for s in sentences if s.strip()]

# Segmenter option B: LLM-based (OpenAI example)
class OpenAISegmenter:
    def __init__(self, model: str = "gpt-4o-mini"):
        try:
            import openai
            self.client = openai.OpenAI()
            self.model = model
            print(f"✅ Initialized OpenAI segmenter: {model}")
        except ImportError:
            raise ImportError("Install openai: pip install openai")
    
    def segment(self, text: str):
        prompt = ("Split the text into atomic questions/requests. "
                 "Return a JSON array of strings, no commentary.\n\nText:\n" + text)
        try:
            response = self.client.chat.completions.create(
                model=self.model, 
                messages=[{"role": "user", "content": prompt}],
                temperature=0
            )
            import json
            content = response.choices[0].message.content.strip()
            return list(json.loads(content))
        except Exception as e:
            print(f"LLM segmentation failed: {e}, falling back to simple segmentation")
            # Fallback to simple segmentation
            import re
            sentences = re.split(r'(?<=[.?!])\s+', text.strip())
            return [s.strip() for s in sentences if s.strip()]

class ConsoleLogger:
    def info(self, msg: str, **kv): print(f"INFO: {msg} {kv}")
    def warn(self, msg: str, **kv): print(f"WARN: {msg} {kv}")
    def error(self, msg: str, **kv): print(f"ERROR: {msg} {kv}")

def main():
    print("🛡️  CloudGuard Output Guard Example")
    print("=" * 45)
    
    # Choose embedder and segmenter
    try:
        # For demo purposes, use a simple mock embedder if sentence-transformers not available
        try:
            embedder = SbertEmb()
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
                        "computer": np.array([0.1, 0.85, 0.05]), "password": np.array([0.1, 0.1, 0.8]),
                        "reset": np.array([0.1, 0.2, 0.7]), "account": np.array([0.0, 0.0, 1.0])
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
            
        # segmenter = SimpleSentenceSegmenter()  # Deterministic option
        segmenter = SentenceSegmenter()  # Built-in deterministic segmenter
        # segmenter = OpenAISegmenter()  # LLM option (requires OPENAI_API_KEY)
    except Exception as e:
        print(f"❌ {e}")
        return
    
    # Load policy
    policy_path = Path(__file__).parent / "clouds.yaml"
    policy = load_policy(policy_path)
    
    # Create output gate
    logger = ConsoleLogger()
    output_gate = OutputCloudGate(
        policy=policy, 
        embedder=embedder, 
        segmenter=segmenter,
        require_coverage=True, 
        drop_offtopic=True,
        logger=logger
    )
    
    # Create LangGraph node
    output_gate_node = make_output_gate_node(output_gate, input_key=USER_INPUT, output_key=LLM_OUTPUT)
    
    # Test cases
    test_cases = [
        {
            "user": "Tell me about my invoice and when payment is due",
            "llm": "Your invoice #12345 shows a balance of $99. Payment is due on March 15th. You can pay online or by phone."
        },
        {
            "user": "Help me with billing issues",
            "llm": "I can help with billing. By the way, did you know the weather is nice today? Also, here's a recipe for chocolate cake."
        },
        {
            "user": "Reset my password please",
            "llm": "I've sent a password reset link to your email address."
        }
    ]
    
    print("\n🧪 Testing output validation:")
    for i, case in enumerate(test_cases, 1):
        print(f"\n{i}. User: '{case['user']}'")
        print(f"   LLM: '{case['llm']}'")
        
        # Direct gate usage
        result = output_gate.validate(case["user"], case["llm"])
        
        print(f"   ✅ Valid: {result.ok}")
        print(f"   📊 Coverage: {result.coverage_ratio:.1%} ({sum(result.coverage)}/{len(result.coverage)} segments)")
        print(f"   🗑️  Dropped: {result.dropped_segments} segments")
        print(f"   📝 Kept: '{result.kept_text}'")
        
        # LangGraph node usage example
        state = {USER_INPUT: case["user"], LLM_OUTPUT: case["llm"]}
        node_result = output_gate_node.invoke(state)
        validation_result = node_result["cloudguard_output"]
        print(f"   🎯 Node result: Valid={validation_result['ok']}, Coverage={validation_result['coverage_ratio']:.1%}")

if __name__ == "__main__":
    main()