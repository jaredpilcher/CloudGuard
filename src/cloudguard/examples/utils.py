"""Utility functions for examples."""

def create_embedder_with_fallback():
    """Create the best available embedder with comprehensive fallbacks."""
    
    # Try different embedding providers in priority order:
    # 1. OpenAI (best quality, requires API key and working API)
    # 2. SentenceTransformers (local, good quality)
    # 3. Mock embedder (for testing/demo, limited accuracy)
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
            from sentence_transformers import SentenceTransformer
            import numpy as np
            
            print("🔍 Initializing SentenceTransformers...")
            
            class SbertEmb:
                def __init__(self, model_name="sentence-transformers/all-MiniLM-L6-v2"):
                    self.model = SentenceTransformer(model_name)
                    print(f"✅ Loaded SentenceTransformers model: {model_name}")
                
                def embed(self, texts):
                    embeddings = self.model.encode(texts)
                    norms = np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-12
                    return embeddings / norms
            
            embedder = SbertEmb()
            print("✅ Using SentenceTransformers embeddings for local processing")
            return embedder
            
        except ImportError as sbert_error:
            print(f"⚠️  SentenceTransformers not available: {sbert_error}")
        except Exception as sbert_error:
            print(f"⚠️  SentenceTransformers initialization failed: {sbert_error}")
    
    # Fall back to mock embedder
    if embedder is None:
        try:
            from cloudguard.providers.mock_embedder import create_mock_embedder
            print("🔍 Falling back to mock embedder...")
            embedder = create_mock_embedder()
            print("✅ Using mock embedder (limited accuracy, good for testing)")
            print("   💡 For production use, install: pip install sentence-transformers")
            return embedder
        except Exception as mock_error:
            print(f"⚠️  Mock embedder failed: {mock_error}")
    
    # If we get here, nothing worked
    print("❌ No embedding provider available.")
    print("   Install sentence-transformers for local embeddings:")
    print("   pip install sentence-transformers")
    print("   Or configure OpenAI API key: export OPENAI_API_KEY=your_key")
    raise RuntimeError("No embedding provider available")


class SimpleConsoleLogger:
    """Simple console logger for examples."""
    
    def info(self, msg: str, **kv):
        details = " ".join(f"{k}={v}" for k, v in kv.items())
        print(f"INFO: {msg} {details}" if details else f"INFO: {msg}")
        
    def warn(self, msg: str, **kv):
        details = " ".join(f"{k}={v}" for k, v in kv.items())
        print(f"WARN: {msg} {details}" if details else f"WARN: {msg}")
        
    def error(self, msg: str, **kv):
        details = " ".join(f"{k}={v}" for k, v in kv.items())
        print(f"ERROR: {msg} {details}" if details else f"ERROR: {msg}")