"""Command-line interface for CloudGuard policy management."""

import argparse
import sys
import json
from pathlib import Path
from typing import Optional

import yaml

from cloudguard.policy.loader import load_policy, PolicyLoadError
from cloudguard.policy.index import build_region_index
from cloudguard.providers.mock_embedder import create_mock_embedder


def validate_policy_command(args):
    """Validate a CloudGuard policy file."""
    try:
        policy_path = Path(args.policy_file)
        if not policy_path.exists():
            print(f"Error: Policy file not found: {policy_path}")
            return 1
        
        print(f"Validating policy: {policy_path}")
        policy = load_policy(policy_path)
        
        print("✅ Policy validation successful!")
        print(f"   Version: {policy.version}")
        print(f"   Regions: {len(policy.regions)}")
        print(f"   Thresholds: in_cloud={policy.thresholds.in_cloud}, margin={policy.thresholds.margin}")
        print(f"   Routing: {policy.routing.abstain_action} -> {policy.routing.default_target or 'none'}")
        
        if args.verbose:
            print("\nRegions:")
            for region in policy.regions:
                print(f"   {region.id}: {len(region.seeds)} seeds -> {region.routes_to or 'none'}")
        
        return 0
        
    except PolicyLoadError as e:
        print(f"❌ Policy validation failed: {e}")
        return 1
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return 1


def test_routing_command(args):
    """Test routing with a policy file."""
    try:
        policy_path = Path(args.policy_file)
        if not policy_path.exists():
            print(f"Error: Policy file not found: {policy_path}")
            return 1
        
        print(f"Loading policy: {policy_path}")
        policy = load_policy(policy_path)
        
        # Create embedder
        if args.mock_embedder:
            print("Using mock embedder for testing")
            embedder = create_mock_embedder()
        else:
            # Try to load real embedder
            try:
                from cloudguard.examples.utils import create_embedder_with_fallback
                embedder = create_embedder_with_fallback()
            except ImportError:
                print("Real embedders not available, falling back to mock")
                embedder = create_mock_embedder()
        
        # Build index and create gate
        from cloudguard.runtime.input_gate import InputCloudGate
        index = build_region_index(policy, embedder)
        gate = InputCloudGate(policy=policy, index=index, embedder=embedder)
        
        # Test queries
        if args.query:
            queries = [args.query]
        else:
            queries = [
                "I need help with my invoice and billing",
                "The app crashes when I try to login", 
                "How do I update my account settings?",
                "What's the weather like today?"
            ]
        
        print("\n🧪 Testing routing:")
        for query in queries:
            decision = gate.route(query)
            print(f"\nQuery: \"{query}\"")
            if decision.decision == 'route':
                print(f"   ✅ ROUTE -> {decision.target} (region: {decision.region_id})")
                print(f"   📊 Score: {decision.score:.3f}")
            else:
                print(f"   ⚠️  ABSTAIN -> {policy.routing.default_target or 'fallback'}")
                print(f"   📊 Score: {decision.score:.3f}, Reason: {decision.reason}")
        
        return 0
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return 1


def info_command(args):
    """Display CloudGuard version and system information."""
    print("CloudGuard CLI")
    print("=" * 50)
    
    # Try to get version from package
    try:
        import importlib.metadata
        version = importlib.metadata.version("langgraph-cloudguard")
        print(f"Version: {version}")
    except Exception:
        print("Version: development")
    
    print(f"Python: {sys.version.split()[0]}")
    
    # Check for optional dependencies
    print("\nOptional dependencies:")
    
    try:
        import sentence_transformers
        print(f"   ✅ sentence-transformers: {sentence_transformers.__version__}")
    except ImportError:
        print("   ❌ sentence-transformers: not installed")
    
    try:
        import openai
        print(f"   ✅ openai: {openai.__version__}")
    except ImportError:
        print("   ❌ openai: not installed")
    
    try:
        import langchain_core
        print(f"   ✅ langchain-core: {langchain_core.__version__}")
    except ImportError:
        print("   ❌ langchain-core: not installed")
    
    return 0


def create_parser():
    """Create the argument parser."""
    parser = argparse.ArgumentParser(
        prog="cloudguard",
        description="CloudGuard policy management and testing CLI"
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Validate command
    validate_parser = subparsers.add_parser(
        "validate",
        help="Validate a CloudGuard policy file"
    )
    validate_parser.add_argument(
        "policy_file",
        help="Path to the policy YAML file"
    )
    validate_parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Show detailed validation results"
    )
    
    # Test command  
    test_parser = subparsers.add_parser(
        "test",
        help="Test routing with a policy file"
    )
    test_parser.add_argument(
        "policy_file", 
        help="Path to the policy YAML file"
    )
    test_parser.add_argument(
        "-q", "--query",
        help="Test a specific query (default: run built-in test queries)"
    )
    test_parser.add_argument(
        "--mock-embedder",
        action="store_true",
        help="Force use of mock embedder for testing"
    )
    
    # Info command
    info_parser = subparsers.add_parser(
        "info",
        help="Display version and system information"
    )
    
    return parser


def main():
    """Main CLI entry point."""
    parser = create_parser()
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return 1
    
    if args.command == "validate":
        return validate_policy_command(args)
    elif args.command == "test":
        return test_routing_command(args)
    elif args.command == "info":
        return info_command(args)
    else:
        print(f"Unknown command: {args.command}")
        return 1


if __name__ == "__main__":
    sys.exit(main())