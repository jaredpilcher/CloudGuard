#!/usr/bin/env python3
"""
CloudGuard Sales Style Routing - Test semantic region overlap in sales response styles.

This example demonstrates:
1. Separating regions by response format rather than topic category
2. Testing semantic overlap between "forceful closing" vs "non-assertive" styles
3. Error handling when regions overlap too much
4. Proper routing and gating when regions are distinct
"""

import sys
import csv
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Set
from dataclasses import dataclass
import json

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from cloudguard.policy.schema import CloudPolicy, Region, Thresholds, RoutingCfg
from cloudguard.policy.loader import load_policy_from_string
from cloudguard.policy.index import build_region_index
from cloudguard.runtime.input_gate import InputCloudGate
from cloudguard.runtime.output_gate import OutputCloudGate
from cloudguard.core.distance import l2_normalize, cosine
import yaml

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
        else:
            print("⚠️  OpenAI not available (missing package or API key)")
    except ImportError as e:
        print(f"⚠️  OpenAI import failed: {e}")
    
    # Fallback to SentenceTransformers
    try:
        print("🔍 Initializing SentenceTransformers...")
        embedder = SbertEmb()
        print("✅ Using local SentenceTransformers for embedding generation")
        return embedder
    except ImportError:
        print("⚠️  SentenceTransformers not available: Install sentence-transformers: pip install sentence-transformers")
    except Exception as e:
        print(f"⚠️  SentenceTransformers initialization failed: {e}")
    
    # No embedder available
    print("❌ No real embedding provider available.")
    print("   Install sentence-transformers for local embeddings:")
    print("   pip install sentence-transformers")
    print("   Or configure OpenAI API key: export OPENAI_API_KEY=your_key")
    raise RuntimeError("No real embedding provider available")

@dataclass
class EmbeddingExample:
    """Single labeled embedding example."""
    text: str
    embedding: np.ndarray
    cloud_label: str  # e.g., "forceful", "not_assertive", "out_of_cloud"
    confidence: float = 1.0

@dataclass
class DiscoveredCloud:
    """A discovered semantic cloud with its characteristics."""
    cloud_id: str
    centroid: np.ndarray
    examples: List[EmbeddingExample]
    radius: float  # Average distance from centroid
    threshold: float  # Recommended similarity threshold

class SalesStyleAnalyzer:
    """Analyzes semantic overlap in sales response styles."""
    
    def __init__(self, min_cloud_size: int = 3, similarity_threshold: float = 0.7):
        self.min_cloud_size = min_cloud_size
        self.similarity_threshold = similarity_threshold
        self.discovered_clouds: List[DiscoveredCloud] = []
        
    def load_csv_data(self, csv_path: Path) -> List[EmbeddingExample]:
        """Load labeled embedding data from CSV file."""
        examples = []
        
        print(f"📄 Loading sales data from {csv_path}")
        
        try:
            with open(csv_path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                required_columns = ['text', 'cloud_label']
                
                if reader.fieldnames is None:
                    raise ValueError("CSV file has no headers")
                    
                if not all(col in reader.fieldnames for col in required_columns):
                    raise ValueError(f"CSV must contain columns: {required_columns}")
                
                # Check if embeddings are provided or need to be generated
                has_embeddings = any(col.startswith('emb_') for col in reader.fieldnames)
                
                # If no embeddings provided, we need a real embedder
                embedder = None
                if not has_embeddings:
                    embedder = create_embedder()  # This will raise an error if no real embedder available
                    print("✅ Using real embeddings for sales style analysis")
                
                for row in reader:
                    text = str(row['text'])
                    cloud_label = str(row['cloud_label'])
                    confidence = float(row.get('confidence', 1.0))
                    
                    if has_embeddings and reader.fieldnames is not None:
                        # Extract embedding from emb_0, emb_1, emb_2, ... columns
                        emb_cols = [col for col in reader.fieldnames if col.startswith('emb_')]
                        emb_cols.sort(key=lambda x: int(x.split('_')[1]))
                        embedding = np.array([float(row[col]) for col in emb_cols], dtype=np.float32)
                    else:
                        # Generate real embedding using OpenAI or SentenceTransformers
                        embedding = embedder.embed([text])[0]
                    
                    # Normalize embedding
                    embedding = l2_normalize(embedding.reshape(1, -1))[0]
                    
                    examples.append(EmbeddingExample(text, embedding, cloud_label, confidence))
            
            print(f"✅ Loaded {len(examples)} sales examples")
            return examples
            
        except Exception as e:
            raise ValueError(f"Failed to load CSV data: {e}")
    
    def analyze_region_overlap(self, examples: List[EmbeddingExample]) -> Dict[str, float]:
        """Analyze semantic overlap between different sales style regions."""
        
        # Group examples by cloud label
        cloud_groups = {}
        for example in examples:
            if example.cloud_label == "out_of_cloud":
                continue  # Skip negative examples for overlap analysis
                
            if example.cloud_label not in cloud_groups:
                cloud_groups[example.cloud_label] = []
            cloud_groups[example.cloud_label].append(example)
        
        print(f"🔍 Analyzing overlap between {len(cloud_groups)} sales style regions")
        
        # Calculate centroids for each region
        centroids = {}
        for cloud_id, cloud_examples in cloud_groups.items():
            if len(cloud_examples) >= self.min_cloud_size:
                embeddings = np.array([ex.embedding for ex in cloud_examples])
                centroid = embeddings.mean(axis=0)
                centroid = centroid / (np.linalg.norm(centroid) + 1e-12)
                centroids[cloud_id] = centroid
        
        # Calculate pairwise similarities between centroids
        similarities = {}
        region_pairs = list(centroids.keys())
        
        for i, region1 in enumerate(region_pairs):
            for j, region2 in enumerate(region_pairs):
                if i < j:  # Only calculate each pair once
                    sim = cosine(centroids[region1], centroids[region2])
                    pair_key = f"{region1} vs {region2}"
                    similarities[pair_key] = sim
                    
                    print(f"📊 Semantic similarity: {pair_key} = {sim:.3f}")
                    
                    # Check for problematic overlap
                    if sim > 0.8:
                        print(f"⚠️  HIGH OVERLAP WARNING: {pair_key} similarity = {sim:.3f}")
                        print(f"   Regions may be too similar for reliable routing!")
                    elif sim > 0.6:
                        print(f"🔶 MODERATE OVERLAP: {pair_key} similarity = {sim:.3f}")
                        print(f"   Consider adjusting training data or thresholds")
                    else:
                        print(f"✅ GOOD SEPARATION: {pair_key} similarity = {sim:.3f}")
        
        return similarities
    
    def discover_clouds(self, examples: List[EmbeddingExample]) -> List[DiscoveredCloud]:
        """Discover clouds from labeled examples."""
        
        # Group examples by cloud label
        cloud_groups = {}
        for example in examples:
            if example.cloud_label == "out_of_cloud":
                continue  # Skip negative examples for cloud discovery
                
            if example.cloud_label not in cloud_groups:
                cloud_groups[example.cloud_label] = []
            cloud_groups[example.cloud_label].append(example)
        
        print(f"🔍 Discovering sales style clouds from {len(cloud_groups)} labeled groups")
        
        discovered = []
        for cloud_id, cloud_examples in cloud_groups.items():
            if len(cloud_examples) < self.min_cloud_size:
                print(f"⚠️  Skipping {cloud_id}: only {len(cloud_examples)} examples (min: {self.min_cloud_size})")
                continue
                
            # Compute centroid
            embeddings = np.array([ex.embedding for ex in cloud_examples])
            centroid = embeddings.mean(axis=0)
            centroid = centroid / (np.linalg.norm(centroid) + 1e-12)
            
            # Compute radius (average distance from centroid)
            distances = [cosine(emb, centroid) for emb in embeddings]
            radius = np.mean(distances)
            
            # Recommend threshold based on radius with safety margin
            threshold = max(0.3, radius - 0.1)  # Conservative threshold
            
            cloud = DiscoveredCloud(cloud_id, centroid, cloud_examples, radius, threshold)
            discovered.append(cloud)
            
            print(f"✅ Discovered '{cloud_id}' cloud: {len(cloud_examples)} examples, radius={radius:.3f}, threshold={threshold:.3f}")
        
        self.discovered_clouds = discovered
        return discovered
    
    def generate_policy(self, clouds: List[DiscoveredCloud]) -> str:
        """Generate CloudGuard policy YAML from discovered clouds."""
        
        # Create regions from discovered clouds
        regions = []
        for cloud in clouds:
            # Use cloud examples as seed texts
            seed_texts = [ex.text for ex in cloud.examples[:5]]  # Use first 5 as seeds
            
            region = {
                'id': cloud.cloud_id,
                'label': f"{cloud.cloud_id.replace('_', ' ').title()} Sales Style",
                'seeds': seed_texts,  # 'seeds' not 'seed_texts'
                'routes_to': f"{cloud.cloud_id}_handler"  # 'routes_to' not 'routing.target'
            }
            regions.append(region)
        
        # Create proper CloudGuard policy structure
        policy_dict = {
            'version': 1,
            'thresholds': {
                'in_cloud': 0.8,  # Standard threshold
                'margin': 0.1     # Abstain margin
            },
            'routing': {
                'abstain_action': 'fallback',
                'default_target': 'default_sales_handler'
            },
            'regions': regions
        }
        
        return yaml.dump(policy_dict, default_flow_style=False)

class SimpleLogger:
    """Simple console logger."""
    
    def info(self, msg: str, **kv):
        details = " ".join(f"{k}={v}" for k, v in kv.items())
        print(f"INFO: {msg} {details}")
        
    def warn(self, msg: str, **kv):
        details = " ".join(f"{k}={v}" for k, v in kv.items())
        print(f"WARN: {msg} {details}")
        
    def error(self, msg: str, **kv):
        details = " ".join(f"{k}={v}" for k, v in kv.items())
        print(f"ERROR: {msg} {details}")

class SimpleTestEmbedder:
    """Simple embedder for testing with pre-computed data."""
    def embed(self, texts):
        # Simple embedding simulation for testing
        embeddings = []
        for text in texts:
            if "decide today" in text.lower() or "expires" in text.lower() or "now" in text.lower():
                # Forceful pattern
                emb = np.array([0.9, 0.2, -0.1, 0.3, 0.1, 0.05, -0.15, 0.25, 0.08, -0.12])
            elif "maybe" in text.lower() or "think about" in text.lower() or "comfortable" in text.lower():
                # Not assertive pattern  
                emb = np.array([-0.1, 0.85, 0.3, -0.05, 0.2, -0.12, 0.15, 0.08, 0.25, -0.18])
            else:
                # Neutral/out of cloud
                emb = np.array([0.05, 0.1, 0.02, -0.8, 0.45, 0.25, -0.35, 0.55, -0.42, 0.38])
            
            # Normalize
            emb = emb / (np.linalg.norm(emb) + 1e-12)
            embeddings.append(emb)
        return np.array(embeddings)

def test_routing_and_gating(policy_yaml: str, embedder, logger):
    """Test both input routing and output gating with the generated policy."""
    
    print("\n🧪 Testing Input Routing")
    print("-" * 50)
    
    # Load policy and build index
    policy = load_policy_from_string(policy_yaml)
    index = build_region_index(policy, embedder)
    
    # Create input gate
    input_gate = InputCloudGate(policy=policy, index=index, embedder=embedder, logger=logger)
    
    # Test input routing
    test_inputs = [
        "You absolutely must decide today - this offer expires at midnight!",
        "Maybe you'd like to consider our options when you're ready",
        "Everyone is buying this product - don't miss out on the trend",
        "I don't want to pressure you, take your time to think about it",
        "Sign now or you'll lose this deal forever",
        "What's the weather forecast for tomorrow?",  # Should abstain
    ]
    
    for i, query in enumerate(test_inputs, 1):
        print(f"\n{i}. Input: \"{query}\"")
        decision = input_gate.route(query)
        
        if decision.decision == "route":
            print(f"   ✅ ROUTE -> {decision.target} (region: {decision.region_id})")
            print(f"   📊 Score: {decision.score:.3f}")
        else:
            print(f"   ⚠️  ABSTAIN -> {decision.target or 'none'}")
            print(f"   📊 Score: {decision.score:.3f}, Reason: {decision.reason}")
    
    print("\n🛡️  Testing Output Gating")
    print("-" * 50)
    
    # Create output gate
    output_gate = OutputCloudGate(policy=policy, embedder=embedder, logger=logger)
    
    # Test output validation
    test_scenarios = [
        {
            "user_input": "How should I close this sales call?",
            "llm_output": "You need to create urgency - tell them this deal expires today and they'll regret missing it!"
        },
        {
            "user_input": "What's a gentle way to end a sales conversation?",
            "llm_output": "Perhaps you could suggest they take some time to think about it and follow up when they're ready."
        },
        {
            "user_input": "How should I close this sales call?",
            "llm_output": "The weather is nice today, and I recommend taking a walk in the park."  # Off-topic
        }
    ]
    
    for i, scenario in enumerate(test_scenarios, 1):
        print(f"\n{i}. User Input: \"{scenario['user_input']}\"")
        print(f"   LLM Output: \"{scenario['llm_output']}\"")
        
        validation = output_gate.validate(
            scenario['user_input'],
            scenario['llm_output']
        )
        
        if validation.ok:
            print(f"   ✅ VALIDATION PASSED - Output covers input appropriately")
            print(f"   📊 Coverage: {validation.coverage_ratio:.1%} ({sum(validation.coverage)}/{len(validation.coverage)} segments)")
            if validation.dropped_segments > 0:
                print(f"   🗑️  Filtered: {validation.dropped_segments} off-topic segments removed")
                print(f"   📝 Kept: \"{validation.kept_text}\"")
        else:
            print(f"   🚫 VALIDATION FAILED - Insufficient coverage or quality")
            print(f"   📊 Coverage: {validation.coverage_ratio:.1%} ({sum(validation.coverage)}/{len(validation.coverage)} segments)")
            if validation.kept_text != scenario['llm_output']:
                print(f"   📝 After filtering: \"{validation.kept_text}\"")

def main():
    """Main execution function."""
    print("🎯 CloudGuard Sales Style Analysis")
    print("========================================")
    
    try:
        # Create analyzer
        analyzer = SalesStyleAnalyzer(min_cloud_size=3, similarity_threshold=0.7)
        
        # Load training data with pre-computed embeddings
        train_csv = Path(__file__).parent / "training_data_with_embeddings.csv"
        train_examples = analyzer.load_csv_data(train_csv)
        
        # Analyze semantic overlap between regions
        print(f"\n📊 Semantic Overlap Analysis")
        print("-" * 50)
        similarities = analyzer.analyze_region_overlap(train_examples)
        
        # Check for problematic overlaps
        high_overlap_pairs = [(pair, sim) for pair, sim in similarities.items() if sim > 0.8]
        if high_overlap_pairs:
            print(f"\n❌ HIGH OVERLAP DETECTED!")
            for pair, sim in high_overlap_pairs:
                print(f"   {pair}: {sim:.3f} similarity")
            print(f"   This may cause routing errors - consider revising training data")
            print(f"   CloudGuard will still attempt routing but accuracy may be reduced")
        else:
            print(f"\n✅ Good semantic separation detected - routing should work well")
        
        # Discover clouds
        print(f"\n🔍 Cloud Discovery")
        print("-" * 50)
        clouds = analyzer.discover_clouds(train_examples)
        
        if len(clouds) < 2:
            print(f"❌ Need at least 2 clouds for meaningful routing")
            return 1
        
        # Generate policy
        print(f"\n📋 Generating CloudGuard Policy")
        print("-" * 50)
        policy_yaml = analyzer.generate_policy(clouds)
        print(f"Generated policy for {len(clouds)} sales style regions")
        
        # Save generated policy
        policy_file = Path(__file__).parent / "generated_sales_policy.yaml"
        with open(policy_file, 'w') as f:
            f.write(policy_yaml)
        print(f"✅ Saved policy to: {policy_file}")
        
        # Test routing and gating
        print(f"\n🧪 Testing Routing and Gating")
        print("=" * 50)
        
        # Create embedder and logger for testing
        embedder = SimpleTestEmbedder()  # Use simple test embedder for demo
        logger = SimpleLogger()
        
        test_routing_and_gating(policy_yaml, embedder, logger)
        
        print(f"\n🎉 Sales style analysis completed successfully!")
        print(f"   Regions analyzed: {len(clouds)}")
        print(f"   Overlap pairs: {len(similarities)}")
        print(f"   Policy generated: {policy_file}")
        
        return 0
        
    except Exception as e:
        print(f"❌ Analysis failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())