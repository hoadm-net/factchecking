#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
MINT TextGraph CLI
Command line interface for MINT TextGraph library
"""

import argparse
import sys
import os
from pathlib import Path
import py_vncorenlp
from .text_graph import TextGraph
from .helpers import (
    setup_vncorenlp, 
    load_sample_data, 
    process_text_data,
    build_complete_graph,
    print_statistics,
    save_outputs,
    validate_inputs,
    load_config
)

def create_parser():
    """Create command line argument parser"""
    # Load defaults from config
    config = load_config()
    
    parser = argparse.ArgumentParser(
        description="MINT TextGraph - Vietnamese Text Graph Analysis with Entity Extraction and Semantic Similarity",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --context "SAWACO thông báo..." --claim "SAWACO ngưng cấp nước..."
  %(prog)s --input-file data.json --similarity-threshold 0.7 --top-k 3
  %(prog)s --demo --export-image graph.png --verbose
  %(prog)s --context "..." --claim "..." --disable-entities --enable-semantic

Note: GPU/CPU auto-detection enabled. System will automatically optimize for your hardware.
        """
    )
    
    # Input options
    input_group = parser.add_argument_group('Input Options')
    input_group.add_argument(
        '--context', '-c',
        type=str,
        help='Context text for analysis'
    )
    input_group.add_argument(
        '--claim', '-l', 
        type=str,
        help='Claim text to verify against context'
    )
    input_group.add_argument(
        '--input-file', '-f',
        type=str,
        help='JSON file containing context and claim data'
    )
    input_group.add_argument(
        '--demo', '-d',
        action='store_true',
        help=f'Run with demo data (from {config["demo_data_path"]})'
    )
    
    # Semantic similarity parameters
    semantic_group = parser.add_argument_group('Semantic Similarity Parameters (auto-optimized for GPU/CPU)')
    semantic_group.add_argument(
        '--similarity-threshold', '-st',
        type=float,
        default=config['similarity_threshold'],
        help=f'Cosine similarity threshold for semantic edges (default: auto-optimized, base: {config["similarity_threshold"]})'
    )
    semantic_group.add_argument(
        '--top-k', '-k',
        type=int,
        default=config['top_k'],
        help=f'Number of most similar words to connect (default: auto-optimized, base: {config["top_k"]})'
    )
    semantic_group.add_argument(
        '--pca-dimensions', '-pca',
        type=int,
        default=config['pca_dimensions'],
        help=f'PCA dimensions for embeddings reduction (default: auto-optimized, base: {config["pca_dimensions"]})'
    )
    semantic_group.add_argument(
        '--disable-pca',
        action='store_true',
        help='Disable PCA dimensionality reduction'
    )
    semantic_group.add_argument(
        '--disable-faiss',
        action='store_true',
        help='Disable FAISS indexing (use brute force search)'
    )
    
    # OpenAI parameters
    openai_group = parser.add_argument_group('OpenAI Parameters')
    openai_group.add_argument(
        '--openai-model', '-om',
        type=str,
        default=config['openai_model'],
        choices=['gpt-4o-mini', 'gpt-4o', 'gpt-3.5-turbo', 'gpt-4'],
        help=f'OpenAI model for entity extraction (default: {config["openai_model"]})'
    )
    openai_group.add_argument(
        '--openai-temperature', '-ot',
        type=float,
        default=config['openai_temperature'],
        help=f'OpenAI temperature (default: {config["openai_temperature"]})'
    )
    openai_group.add_argument(
        '--openai-max-tokens', '-om-tokens',
        type=int,
        default=config['openai_max_tokens'],
        help=f'OpenAI max tokens (default: {config["openai_max_tokens"]})'
    )
    
    # Feature toggles
    feature_group = parser.add_argument_group('Feature Toggles')
    feature_group.add_argument(
        '--disable-entities',
        action='store_true',
        help='Disable entity extraction'
    )
    feature_group.add_argument(
        '--disable-semantic',
        action='store_true',
        help='Disable semantic similarity edges'
    )
    feature_group.add_argument(
        '--disable-dependencies',
        action='store_true',
        help='Disable dependency parsing edges'
    )
    feature_group.add_argument(
        '--enable-statistics', '-s',
        action='store_true',
        default=config['enable_statistics'],
        help=f'Enable detailed statistics (default: {config["enable_statistics"]})'
    )
    feature_group.add_argument(
        '--disable-statistics',
        action='store_true',
        help='Disable statistics output'
    )
    
    # Output options
    output_group = parser.add_argument_group('Output Options')
    output_group.add_argument(
        '--export-image', '-ei',
        type=str,
        default=None,
        help='Export visualization to image file (e.g., graph.png)'
    )
    output_group.add_argument(
        '--export-graph', '-eg',
        type=str,
        default=config['export_graph'],
        help=f'Export graph to file (default: {config["export_graph"]})'
    )
    output_group.add_argument(
        '--export-json', '-ej',
        type=str,
        default=None,
        help='Export graph data to JSON file'
    )
    output_group.add_argument(
        '--disable-visualization',
        action='store_true',
        help='Disable graph visualization'
    )
    output_group.add_argument(
        '--figure-size', '-fs',
        type=str,
        default=f'{config["figure_width"]},{config["figure_height"]}',
        help=f'Figure size for visualization (width,height) (default: {config["figure_width"]},{config["figure_height"]})'
    )
    
    # System and paths
    system_group = parser.add_argument_group('System Options')
    system_group.add_argument('--vncorenlp-path', 
                            default=config['vncorenlp_path'],
                            help=f'Path to VnCoreNLP directory (default: {config["vncorenlp_path"]})')
    system_group.add_argument('--no-auto-download', action='store_true',
                            help='Disable automatic VnCoreNLP download')
    system_group.add_argument('--force-download', action='store_true',
                            help='Force re-download VnCoreNLP even if exists')
    system_group.add_argument('--verbose', '-v', action='store_true',
                            help='Enable verbose output')
    system_group.add_argument('--quiet', '-q', action='store_true',
                            help='Suppress all output except errors')
    
    return parser

def track_user_overrides(args, parser):
    """Track which parameters user explicitly specified"""
    # Check if user provided custom values (not defaults)
    defaults = vars(parser.parse_args([]))  # Parse empty args to get defaults
    current = vars(args)
    
    # Mark which values were overridden by user
    args.similarity_threshold_overridden = current['similarity_threshold'] != defaults['similarity_threshold']
    args.top_k_overridden = current['top_k'] != defaults['top_k'] 
    args.pca_dimensions_overridden = current['pca_dimensions'] != defaults['pca_dimensions']

def main():
    """Main CLI entry point"""
    parser = create_parser()
    args = parser.parse_args()
    
    # Track user overrides before device optimization
    track_user_overrides(args, parser)
    
    # Handle conflicting options
    if args.quiet and args.verbose:
        print("Error: Cannot use --quiet and --verbose together")
        sys.exit(1)
    
    if args.disable_statistics:
        args.enable_statistics = False
    
    # Validate inputs and get device info
    try:
        context, claim, device_info = validate_inputs(args)
    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)
    
    if args.verbose:
        print(f"🖥️ Device detected: {device_info['type']} ({device_info['name']})")
        if device_info['type'] == 'GPU':
            print(f"  Memory: {device_info['memory_gb']}")
        print()
        
        print(f"📋 Configuration:")
        print(f"  Context length: {len(context)} chars")
        print(f"  Claim length: {len(claim)} chars")
        print(f"  Similarity threshold: {args.similarity_threshold}")
        print(f"  Top-K: {args.top_k}")
        print(f"  OpenAI model: {args.openai_model}")
        print(f"  PCA dimensions: {args.pca_dimensions}")
        print(f"  Use FAISS: {not args.disable_faiss}")
        print()
    
    try:
        # Handle VnCoreNLP download options
        auto_download = not args.no_auto_download
        
        # Force download if requested
        if args.force_download:
            if args.verbose:
                print("🔄 Force downloading VnCoreNLP...")
            try:
                from .helpers import download_vncorenlp
                download_vncorenlp(args.vncorenlp_path, verbose=args.verbose)
            except Exception as e:
                print(f"❌ Failed to download VnCoreNLP: {e}")
                sys.exit(1)
        
        # Setup VnCoreNLP
        if not args.quiet:
            print("🤖 Setting up VnCoreNLP...")
        model = setup_vncorenlp(args.vncorenlp_path, args.verbose, auto_download)
        
        # Process text data
        if not args.quiet:
            print("📝 Processing text data...")
        context_sentences, claim_sentences = process_text_data(
            model, context, claim, args.verbose
        )
        
        # Build complete graph
        if not args.quiet:
            print("🔗 Building text graph...")
        text_graph = build_complete_graph(
            context, claim, context_sentences, claim_sentences, args
        )
        
        # Print statistics
        if args.enable_statistics and not args.quiet:
            print_statistics(text_graph, args.verbose)
        
        # Save outputs
        if not args.quiet:
            print("💾 Saving outputs...")
        save_outputs(text_graph, args)
        
        if not args.quiet:
            print("✅ Analysis completed successfully!")
            
    except KeyboardInterrupt:
        print("\n❌ Analysis interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error during analysis: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)

def download_command():
    """Separate command to download VnCoreNLP"""
    import argparse
    import os
    import shutil
    
    parser = argparse.ArgumentParser(
        description='Download VnCoreNLP models and tools',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  mint-download                           # Download to default location
  mint-download --path ./my-vncorenlp     # Download to specific path
  mint-download --force                   # Force re-download
        """
    )
    
    parser.add_argument('--path', '-p', 
                       default='vncorenlp',
                       help='Download path for VnCoreNLP (default: vncorenlp)')
    parser.add_argument('--force', '-f', 
                       action='store_true',
                       help='Force download even if already exists')
    parser.add_argument('--verbose', '-v', 
                       action='store_true',
                       help='Enable verbose output')
    
    args = parser.parse_args()
    
    try:
        from .helpers import download_vncorenlp
        
        if args.force:
            if os.path.exists(args.path):
                if args.verbose:
                    print(f"🗑️ Removing existing VnCoreNLP at: {args.path}")
                shutil.rmtree(args.path)
        
        download_vncorenlp(args.path, verbose=args.verbose)
        print(f"✅ VnCoreNLP downloaded successfully to: {os.path.abspath(args.path)}")
        
    except Exception as e:
        print(f"❌ Failed to download VnCoreNLP: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 