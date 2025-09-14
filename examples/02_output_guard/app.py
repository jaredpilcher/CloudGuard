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

# Note: OpenAI-based segmentation can be implemented using the CloudGuard 
# OpenAI provider system. For now, using deterministic segmentation.

class ConsoleLogger:
    def info(self, msg: str, **kv): print(f"INFO: {msg} {kv}")
    def warn(self, msg: str, **kv): print(f"WARN: {msg} {kv}")
    def error(self, msg: str, **kv): print(f"ERROR: {msg} {kv}")

def main():
    print("🛡️  CloudGuard Output Guard Example")
    print("=" * 45)
    
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
            embedder = SbertEmb()
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
            
    # segmenter = SimpleSentenceSegmenter()  # Deterministic option
    segmenter = SentenceSegmenter()  # Built-in deterministic segmenter
    
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