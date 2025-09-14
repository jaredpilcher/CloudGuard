#!/usr/bin/env python3
"""
CloudGuard Cloud Discovery - Automatically discover semantic clouds from labeled embedding data.

Takes CSV files with embeddings and in_cloud/out_cloud labels, then:
1. Learns cloud boundaries from the labeled data
2. Automatically generates CloudGuard policies for multiple clouds
3. Validates the discovered clouds against test data
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
from cloudguard.core.distance import l2_normalize, cosine
import yaml

@dataclass
class EmbeddingExample:
    """Single labeled embedding example."""
    text: str
    embedding: np.ndarray
    cloud_label: str  # e.g., "billing", "tech_support", "out_of_cloud"
    confidence: float = 1.0

@dataclass
class DiscoveredCloud:
    """A discovered semantic cloud with its characteristics."""
    cloud_id: str
    centroid: np.ndarray
    examples: List[EmbeddingExample]
    radius: float  # Average distance from centroid
    threshold: float  # Recommended similarity threshold

class CloudDiscovery:
    """Discovers semantic clouds from labeled embedding data."""
    
    def __init__(self, min_cloud_size: int = 3, similarity_threshold: float = 0.7):
        self.min_cloud_size = min_cloud_size
        self.similarity_threshold = similarity_threshold
        self.discovered_clouds: List[DiscoveredCloud] = []
        
    def load_csv_data(self, csv_path: Path) -> List[EmbeddingExample]:
        """Load labeled embedding data from CSV file."""
        examples = []
        
        print(f"📄 Loading data from {csv_path}")
        
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
                    print("✅ Using real embeddings for cloud discovery")
                
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
            
            print(f"✅ Loaded {len(examples)} examples")
            return examples
            
        except Exception as e:
            raise ValueError(f"Failed to load CSV data: {e}")
    
    
    def discover_clouds(self, examples: List[EmbeddingExample]) -> List[DiscoveredCloud]:
        """Discover clouds from labeled examples using clustering."""
        
        # Group examples by cloud label
        cloud_groups = {}
        for example in examples:
            if example.cloud_label == "out_of_cloud":
                continue  # Skip negative examples for cloud discovery
                
            if example.cloud_label not in cloud_groups:
                cloud_groups[example.cloud_label] = []
            cloud_groups[example.cloud_label].append(example)
        
        print(f"🔍 Discovering clouds from {len(cloud_groups)} labeled groups")
        
        discovered = []
        for cloud_id, cloud_examples in cloud_groups.items():
            if len(cloud_examples) < self.min_cloud_size:
                print(f"⚠️  Skipping {cloud_id}: only {len(cloud_examples)} examples (min: {self.min_cloud_size})")
                continue
                
            # Compute centroid
            embeddings = np.array([ex.embedding for ex in cloud_examples])
            centroid = embeddings.mean(axis=0)
            centroid = centroid / (np.linalg.norm(centroid) + 1e-12)
            
            # Compute average distance (radius)
            distances = [cosine(centroid, ex.embedding) for ex in cloud_examples]
            avg_similarity = float(np.mean(distances))
            std_similarity = float(np.std(distances))
            
            # Set threshold based on data distribution (more conservative)
            threshold = max(0.5, avg_similarity - 2 * std_similarity)
            
            cloud = DiscoveredCloud(
                cloud_id=cloud_id,
                centroid=centroid,
                examples=cloud_examples,
                radius=avg_similarity,
                threshold=threshold
            )
            
            discovered.append(cloud)
            print(f"✅ Discovered cloud '{cloud_id}': {len(cloud_examples)} examples, "
                  f"avg_sim={avg_similarity:.3f}, threshold={threshold:.3f}")
        
        self.discovered_clouds = discovered
        return discovered
    
    def generate_policy(self, clouds: List[DiscoveredCloud], 
                       global_threshold: Optional[float] = None) -> CloudPolicy:
        """Generate CloudGuard policy from discovered clouds."""
        
        if global_threshold is None:
            # Use a reasonable threshold based on discovered cloud performance
            global_threshold = min(cloud.threshold for cloud in clouds) if clouds else 0.6
        
        # Create regions from discovered clouds
        regions = []
        for cloud in clouds:
            # Use representative examples as seeds
            seeds = [ex.text for ex in cloud.examples[:5]]  # Top 5 examples as seeds
            
            region = Region(
                id=cloud.cloud_id,
                label=cloud.cloud_id.replace("_", " ").title(),
                seeds=seeds,
                routes_to=f"agent_{cloud.cloud_id}"
            )
            regions.append(region)
        
        policy = CloudPolicy(
            version=1,
            thresholds=Thresholds(
                in_cloud=global_threshold,
                margin=0.1
            ),
            routing=RoutingCfg(
                abstain_action="fallback",
                default_target="general_agent"
            ),
            regions=regions
        )
        
        return policy
    
    def validate_discovery(self, policy: CloudPolicy, test_examples: List[EmbeddingExample],
                          embedder) -> Dict[str, float]:
        """Validate discovered clouds against test data."""
        
        print(f"\n🧪 Validating discovery with {len(test_examples)} test examples")
        
        # Build index and gate (embedder should not be None at this point)
        if embedder is None:
            raise RuntimeError("Embedder is required for validation")
        index = build_region_index(policy, embedder)
        gate = InputCloudGate(policy=policy, index=index, embedder=embedder)
        
        # Test predictions
        correct_routes = 0
        correct_abstains = 0
        total_in_cloud = 0
        total_out_cloud = 0
        
        for example in test_examples:
            decision = gate.route(example.text)
            
            if example.cloud_label == "out_of_cloud":
                total_out_cloud += 1
                if decision.decision == "abstain":
                    correct_abstains += 1
            else:
                total_in_cloud += 1
                if decision.decision == "route" and decision.region_id == example.cloud_label:
                    correct_routes += 1
        
        metrics = {
            "route_accuracy": correct_routes / max(total_in_cloud, 1),
            "abstain_accuracy": correct_abstains / max(total_out_cloud, 1),
            "overall_accuracy": (correct_routes + correct_abstains) / len(test_examples)
        }
        
        print(f"📊 Validation Results:")
        print(f"   Route Accuracy: {metrics['route_accuracy']:.1%} ({correct_routes}/{total_in_cloud})")
        print(f"   Abstain Accuracy: {metrics['abstain_accuracy']:.1%} ({correct_abstains}/{total_out_cloud})")
        print(f"   Overall Accuracy: {metrics['overall_accuracy']:.1%}")
        
        return metrics

def create_embedder():
    """Create the best available embedder with proper fallbacks."""
    from cloudguard.providers.openai_embedder import create_openai_embedder, is_openai_available
    
    # Try OpenAI first
    if is_openai_available():
        try:
            print("🔍 Testing OpenAI API connection...")
            test_embedder = create_openai_embedder()
            # Test with a small embedding to verify API is working
            test_result = test_embedder.embed(["test connection"])
            if test_result.shape[0] > 0 and test_result.shape[1] > 0:
                print("✅ Using OpenAI embeddings for best quality semantic understanding")
                return test_embedder
            else:
                raise RuntimeError("OpenAI API returned empty results")
        except Exception as api_error:
            print(f"⚠️  OpenAI API test failed: {api_error}")
    else:
        print("⚠️  OpenAI not available (missing package or API key)")
    
    # Try SentenceTransformers as fallback
    try:
        print("🔍 Initializing SentenceTransformers...")
        from sentence_transformers import SentenceTransformer
        import numpy as np
        
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
        
    except ImportError:
        print("⚠️  SentenceTransformers not available")
    except Exception as sbert_error:
        print(f"⚠️  SentenceTransformers initialization failed: {sbert_error}")
    
    # No fallback - exit with error
    print("❌ No real embedding provider available.")
    print("   Install sentence-transformers for local embeddings:")
    print("   pip install sentence-transformers")
    print("   Or configure OpenAI API key: export OPENAI_API_KEY=your_key")
    raise RuntimeError("No real embedding provider available")


def main():
    print("🔍 CloudGuard Cloud Discovery")
    print("=" * 40)
    
    discovery = CloudDiscovery(min_cloud_size=2, similarity_threshold=0.7)
    
    # Load training data
    train_csv = Path(__file__).parent / "training_data.csv"
    if not train_csv.exists():
        print(f"❌ Training data not found: {train_csv}")
        print("Creating sample training data...")
        create_sample_data()
    
    train_examples = discovery.load_csv_data(train_csv)
    
    # Discover clouds
    clouds = discovery.discover_clouds(train_examples)
    print(f"\n✅ Discovered {len(clouds)} clouds")
    
    # Generate policy
    policy = discovery.generate_policy(clouds)
    
    # Save generated policy
    policy_yaml = yaml.dump(policy.model_dump(), default_flow_style=False, sort_keys=False)
    
    policy_file = Path(__file__).parent / "discovered_policy.yaml"
    with open(policy_file, 'w') as f:
        f.write(policy_yaml)
    
    print(f"\n📝 Generated policy saved to: {policy_file}")
    
    # Test with validation data if available
    test_csv = Path(__file__).parent / "test_data.csv"
    if test_csv.exists():
        test_examples = discovery.load_csv_data(test_csv)
        try:
            embedder = create_embedder()
            metrics = discovery.validate_discovery(policy, test_examples, embedder)
        except RuntimeError as e:
            print(f"⚠️  Skipping validation: {e}")
    else:
        print("📝 No test data found, skipping validation")
    
    # Demo the discovered policy
    print(f"\n🚀 Testing discovered policy:")
    try:
        embedder = create_embedder()
        index = build_region_index(policy, embedder)
        gate = InputCloudGate(policy=policy, index=index, embedder=embedder)
    except RuntimeError as e:
        print(f"❌ Cannot demo policy: {e}")
        return
    
    test_queries = [
        "I need help with my invoice",
        "Software installation problems", 
        "Reset my account password",
        "I want to buy your product",
        "What's the weather today?"
    ]
    
    for query in test_queries:
        decision = gate.route(query)
        if decision.decision == "route":
            print(f"✅ '{query}' → {decision.target} (cloud: {decision.region_id})")
        else:
            print(f"⚠️  '{query}' → abstain (score: {decision.score:.3f})")

def create_sample_data():
    """Create sample training and test data for demonstration."""
    
    training_data = [
        # Billing examples
        ("I need help with my invoice", "billing", 1.0),
        ("Payment processing issues", "billing", 1.0),
        ("Refund request for overcharge", "billing", 1.0),
        ("Credit card billing questions", "billing", 0.9),
        ("Invoice payment due date", "billing", 0.8),
        
        # Tech support examples
        ("Software won't install properly", "tech_support", 1.0),
        ("Application keeps crashing", "tech_support", 1.0),
        ("Error code 404 not found", "tech_support", 1.0),
        ("Technical troubleshooting needed", "tech_support", 0.9),
        ("Computer compatibility issues", "tech_support", 0.8),
        
        # Account examples  
        ("Reset my password please", "account", 1.0),
        ("Update profile information", "account", 1.0),
        ("Account login problems", "account", 1.0),
        ("Change account settings", "account", 0.9),
        ("Profile security options", "account", 0.8),
        
        # Sales examples
        ("I want to buy your product", "sales", 1.0),
        ("Pricing information request", "sales", 1.0),
        ("Demo scheduling inquiry", "sales", 1.0),
        ("Purchase consultation needed", "sales", 0.9),
        ("Product features comparison", "sales", 0.8),
        
        # Out of cloud examples
        ("What's the weather today?", "out_of_cloud", 1.0),
        ("Tell me a joke", "out_of_cloud", 1.0),
        ("Recipe for chocolate cake", "out_of_cloud", 1.0),
        ("Movie recommendations", "out_of_cloud", 0.9),
        ("Sports scores update", "out_of_cloud", 0.8),
    ]
    
    test_data = [
        ("Billing statement questions", "billing", 1.0),
        ("Software bug report", "tech_support", 1.0),
        ("Forgot my login credentials", "account", 1.0),
        ("Interested in your services", "sales", 1.0),
        ("Random trivia question", "out_of_cloud", 1.0),
    ]
    
    # Create training CSV
    train_file = Path(__file__).parent / "training_data.csv"
    with open(train_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['text', 'cloud_label', 'confidence'])
        writer.writerows(training_data)
    
    # Create test CSV
    test_file = Path(__file__).parent / "test_data.csv"
    with open(test_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['text', 'cloud_label', 'confidence'])
        writer.writerows(test_data)
    
    print(f"📄 Created sample data:")
    print(f"   Training: {train_file} ({len(training_data)} examples)")
    print(f"   Test: {test_file} ({len(test_data)} examples)")

if __name__ == "__main__":
    main()