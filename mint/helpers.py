#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
MINT TextGraph Helper Functions
Helper functions for CLI and text processing
"""

import json
import os
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import py_vncorenlp
from dotenv import load_dotenv
from .text_graph import TextGraph

def detect_device():
    """Automatically detect and configure optimal device (GPU/CPU)"""
    try:
        import torch
        
        if torch.cuda.is_available():
            device = 'cuda'
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            device_info = {
                'type': 'GPU',
                'name': gpu_name,
                'memory_gb': f"{gpu_memory:.1f}GB",
                'device': device,
                'use_gpu_optimizations': True
            }
        else:
            device = 'cpu'
            cpu_count = torch.get_num_threads()
            device_info = {
                'type': 'CPU',
                'name': f'{cpu_count} threads',
                'memory_gb': 'N/A',
                'device': device,
                'use_gpu_optimizations': False
            }
            
    except ImportError:
        # Fallback nếu torch không có
        device_info = {
            'type': 'CPU',
            'name': 'Unknown (torch not available)',
            'memory_gb': 'N/A', 
            'device': 'cpu',
            'use_gpu_optimizations': False
        }
    
    return device_info

def get_optimized_config_for_device(device_info, base_config):
    """Get optimized configuration based on detected device (PCA removed)"""
    if device_info['use_gpu_optimizations']:
        # GPU optimizations - sử dụng full embeddings với FAISS
        return {
            'similarity_threshold': base_config.get('similarity_threshold', 0.85),
            'top_k': base_config.get('top_k', 5),
            'use_faiss': base_config.get('use_faiss', True)
        }
    else:
        # CPU optimizations - giảm top_k, tăng threshold, có thể tắt FAISS
        return {
            'similarity_threshold': base_config.get('cpu_similarity_threshold', 0.9),
            'top_k': base_config.get('cpu_top_k', 3),
            'use_faiss': base_config.get('cpu_use_faiss', False)  # FAISS có thể problematic trên một số CPU
        }

def load_config():
    """Load configuration from .env file with defaults"""
    load_dotenv()
    
    config = {
        # Semantic Similarity defaults (PCA removed)
        'similarity_threshold': float(os.getenv('DEFAULT_SIMILARITY_THRESHOLD', '0.85')),
        'top_k': int(os.getenv('DEFAULT_TOP_K', '5')),
        'use_faiss': os.getenv('DEFAULT_USE_FAISS', 'true').lower() == 'true',
        
        # OpenAI defaults
        'openai_model': os.getenv('DEFAULT_OPENAI_MODEL', 'gpt-4o-mini'),
        'openai_temperature': float(os.getenv('DEFAULT_OPENAI_TEMPERATURE', '0.0')),
        'openai_max_tokens': int(os.getenv('DEFAULT_OPENAI_MAX_TOKENS', '1000')),
        
        # Visualization defaults
        'figure_width': float(os.getenv('DEFAULT_FIGURE_WIDTH', '15')),
        'figure_height': float(os.getenv('DEFAULT_FIGURE_HEIGHT', '10')),
        'dpi': int(os.getenv('DEFAULT_DPI', '300')),
        
        # System defaults
        'vncorenlp_path': os.getenv('DEFAULT_VNCORENLP_PATH', 'vncorenlp'),
        'export_graph': os.getenv('DEFAULT_EXPORT_GRAPH', 'text_graph.gexf'),
        'enable_statistics': os.getenv('DEFAULT_ENABLE_STATISTICS', 'true').lower() == 'true',
        'enable_visualization': os.getenv('DEFAULT_ENABLE_VISUALIZATION', 'true').lower() == 'true',
        
        # Demo data
        'demo_data_path': os.getenv('DEMO_DATA_PATH', 'data/demo.json'),
        
        # CPU mode (low performance fallback - no PCA)
        'cpu_similarity_threshold': float(os.getenv('CPU_SIMILARITY_THRESHOLD', '0.9')),
        'cpu_top_k': int(os.getenv('CPU_TOP_K', '3')),
        'cpu_use_faiss': os.getenv('CPU_USE_FAISS', 'false').lower() == 'true',
    }
    
    return config

def load_demo_data():
    """Load demo data from JSON file"""
    config = load_config()
    demo_path = config['demo_data_path']
    
    # Fallback SAWACO data if file not found
    fallback_data = {
        "context": """(PLO)- Theo Tổng Công ty Cấp nước Sài Gòn (SAWACO) việc cúp nước là để thực hiện công tác bảo trì, bảo dưỡng định kỳ Nhà máy nước Tân Hiệp. SAWACO cho biết đây là phương án để đảm bảo cung cấp nước sạch an toàn, liên tục phục vụ cho người dân TP. Vì vậy, SAWACO thông báo tạm ngưng cung cấp nước để thực hiện công tác nêu trên. Thời gian thực hiện dự kiến từ 22 giờ ngày 25-3 (thứ bảy) đến 4 giờ ngày 26-3 (chủ nhật). Các khu vực tạm ngưng cung cấp nước gồm quận 6, 8, 12, Gò Vấp, Tân Bình, Tân Phú, Bình Tân và huyện Hóc Môn, Bình Chánh.""",
        "claim": """SAWACO thông báo tạm ngưng cung cấp nước để thực hiện công tác bảo trì, bảo dưỡng định kỳ Nhà máy nước Tân Hiệp, thời gian thực hiện dự kiến từ 12 giờ ngày 25-3 (thứ bảy) đến 4 giờ ngày 26-3 (chủ nhật)."""
    }
    
    try:
        if os.path.exists(demo_path):
            with open(demo_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            return data.get('context', ''), data.get('claim', '')
        else:
            print(f"⚠️ Demo file not found: {demo_path}, using fallback SAWACO data")
            return fallback_data["context"], fallback_data["claim"]
    except Exception as e:
        print(f"⚠️ Error loading demo data: {e}, using fallback")
        return fallback_data["context"], fallback_data["claim"]

def apply_device_optimizations(args, device_info, verbose=False):
    """Apply device-specific optimizations to arguments (PCA removed)"""
    config = load_config()
    optimized_config = get_optimized_config_for_device(device_info, config)
    
    # Only override if user didn't specify custom values
    if not getattr(args, 'similarity_threshold_overridden', False):
        args.similarity_threshold = optimized_config['similarity_threshold']
    if not getattr(args, 'top_k_overridden', False):
        args.top_k = optimized_config['top_k']
    
    # Set technical flags
    args.disable_faiss = not optimized_config['use_faiss']
    
    if verbose:
        optimization_type = "GPU" if device_info['use_gpu_optimizations'] else "CPU"
        print(f"🔧 {optimization_type} optimizations applied (full embeddings - no PCA):")
        print(f"  Similarity threshold: {args.similarity_threshold}")
        print(f"  Top-K: {args.top_k}")
        print(f"  Use FAISS: {not args.disable_faiss}")
        print(f"  Embedding dimensions: 768 (full PhoBERT)")

def validate_inputs(args):
    """Validate and extract input data from arguments"""
    context = None
    claim = None
    
    if args.demo:
        context, claim = load_demo_data()
        if args.verbose:
            demo_name = "Bánh cuốn Thụy Khuê" if "bánh cuốn" in context.lower() else "SAWACO"
            print(f"📋 Using demo data ({demo_name} example)")
    
    elif args.input_file:
        if not os.path.exists(args.input_file):
            raise ValueError(f"Input file not found: {args.input_file}")
        
        try:
            with open(args.input_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            context = data.get('context', '')
            claim = data.get('claim', '')
            
            if not context or not claim:
                raise ValueError("Input file must contain 'context' and 'claim' fields")
                
        except json.JSONDecodeError:
            raise ValueError(f"Invalid JSON format in {args.input_file}")
    
    elif args.context and args.claim:
        context = args.context
        claim = args.claim
    
    else:
        raise ValueError("Must provide either --demo, --input-file, or both --context and --claim")
    
    if not context.strip() or not claim.strip():
        raise ValueError("Context and claim cannot be empty")
    
    # Auto-detect device and apply optimizations
    device_info = detect_device()
    apply_device_optimizations(args, device_info, args.verbose)
    
    return context.strip(), claim.strip(), device_info

def setup_vncorenlp(vncorenlp_path, verbose=False, auto_download=True):
    """Setup VnCoreNLP model with automatic download if needed"""
    
    # Convert to absolute path
    if not os.path.isabs(vncorenlp_path):
        vncorenlp_path = os.path.abspath(vncorenlp_path)
    
    # Check if VnCoreNLP exists
    jar_path = os.path.join(vncorenlp_path, "VnCoreNLP-1.2.jar")
    models_path = os.path.join(vncorenlp_path, "models")
    
    if not (os.path.exists(jar_path) and os.path.exists(models_path)):
        if auto_download:
            if verbose:
                print(f"  VnCoreNLP not found at: {vncorenlp_path}")
                print(f"  Auto-downloading VnCoreNLP...")
            try:
                vncorenlp_path = download_vncorenlp(vncorenlp_path, verbose)
            except Exception as e:
                raise RuntimeError(f"Failed to auto-download VnCoreNLP: {e}")
        else:
            raise RuntimeError(f"VnCoreNLP not found at: {vncorenlp_path}")
    
    try:
        if verbose:
            print(f"  Loading VnCoreNLP from: {vncorenlp_path}")
        
        model = py_vncorenlp.VnCoreNLP(save_dir=vncorenlp_path)
        
        if verbose:
            print("  ✅ VnCoreNLP loaded successfully")
        
        return model
    
    except Exception as e:
        raise RuntimeError(f"Failed to load VnCoreNLP: {e}")

def process_text_data(model, context, claim, verbose=False):
    """Process context and claim with VnCoreNLP"""
    try:
        if verbose:
            print("  Processing context...")
        context_sentences = model.annotate_text(context)
        
        if verbose:
            print("  Processing claim...")
        claim_sentences = model.annotate_text(claim)
        
        if verbose:
            print(f"  ✅ Processed {len(context_sentences)} context sentences")
            print(f"  ✅ Processed {len(claim_sentences)} claim sentences")
        
        return context_sentences, claim_sentences
    
    except Exception as e:
        raise RuntimeError(f"Failed to process text data: {e}")

def configure_textgraph_parameters(text_graph, args):
    """Configure TextGraph parameters from arguments with .env defaults (PCA removed)"""
    config = load_config()
    
    # Use CLI args if provided, otherwise use .env defaults
    text_graph.similarity_threshold = getattr(args, 'similarity_threshold', config['similarity_threshold'])
    text_graph.top_k_similar = getattr(args, 'top_k', config['top_k'])
    
    # OpenAI parameters
    if hasattr(text_graph, '_update_openai_model'):
        text_graph._update_openai_model(
            model=getattr(args, 'openai_model', config['openai_model']),
            temperature=getattr(args, 'openai_temperature', config['openai_temperature']),
            max_tokens=getattr(args, 'openai_max_tokens', config['openai_max_tokens'])
        )

def build_complete_graph(context, claim, context_sentences, claim_sentences, args):
    """Build complete text graph with all features"""
    # Initialize TextGraph
    text_graph = TextGraph()
    
    # Configure parameters
    configure_textgraph_parameters(text_graph, args)
    
    # Build basic graph
    if args.verbose:
        print("  Building basic graph structure...")
    text_graph.build_from_vncorenlp_output(context_sentences, claim, claim_sentences)
    
    # Entity extraction
    if not args.disable_entities:
        if args.verbose:
            print("  Extracting entities with OpenAI...")
        try:
            entity_nodes = text_graph.extract_and_add_entities(context, context_sentences)
            if args.verbose:
                print(f"  ✅ Added {len(entity_nodes)} entity nodes")
        except Exception as e:
            if args.verbose:
                print(f"  ⚠️ Entity extraction failed: {e}")
    
    # Semantic similarity (without PCA)
    if not args.disable_semantic:
        if args.verbose:
            print("  Building semantic similarity edges (full embeddings - no PCA)...")
        try:
            use_faiss = not args.disable_faiss
            
            edges_added = text_graph.build_semantic_similarity_edges(
                use_faiss=use_faiss
            )
            if args.verbose:
                print(f"  ✅ Added {edges_added} semantic edges")
        except Exception as e:
            if args.verbose:
                print(f"  ⚠️ Semantic similarity failed: {e}")
    
    return text_graph

def print_statistics(text_graph, verbose=False):
    """Print detailed statistics"""
    print("\n" + "="*50)
    print("📊 DETAILED STATISTICS")
    print("="*50)
    
    stats = text_graph.get_detailed_statistics()
    
    # Basic statistics
    print(f"📈 Graph Overview:")
    print(f"  Total nodes: {stats['total_nodes']}")
    print(f"  Total edges: {stats['total_edges']}")
    print(f"    - Structural edges: {stats['structural_edges']}")
    print(f"    - Dependency edges: {stats['dependency_edges']}")
    print(f"    - Entity edges: {stats.get('entity_structural_edges', 0)}")
    print(f"    - Semantic edges: {stats.get('semantic_edges', 0)}")
    
    print(f"\n📍 Node Types:")
    print(f"  Word nodes: {stats['word_nodes']}")
    print(f"  Sentence nodes: {stats['sentence_nodes']}")
    print(f"  Claim nodes: {stats['claim_nodes']}")
    print(f"  Entity nodes: {stats.get('entity_nodes', 0)}")
    
    # Word analysis
    print(f"\n📝 Text Analysis:")
    print(f"  Unique words: {stats['unique_words']}")
    print(f"  Shared words (context & claim): {stats['shared_words_count']}")
    print(f"  Average words per sentence: {stats['average_words_per_sentence']:.1f}")
    
    # Entity information
    if stats.get('entities'):
        print(f"\n🏷️ Entities Extracted:")
        for entity in stats['entities'][:10]:  # Show first 10
            print(f"  '{entity['name']}' - {entity['connected_sentences']} connections")
        if len(stats['entities']) > 10:
            print(f"  ... and {len(stats['entities']) - 10} more entities")
    
    # Semantic similarity info
    if stats.get('semantic_edges', 0) > 0:
        semantic_stats = stats['semantic_statistics']
        print(f"\n🔗 Semantic Similarity:")
        print(f"  Total semantic edges: {semantic_stats['total_semantic_edges']}")
        print(f"  Average similarity: {semantic_stats['average_similarity']:.3f}")
        print(f"  Similarity range: {semantic_stats['min_similarity']:.3f} - {semantic_stats['max_similarity']:.3f}")
        
        print("  Similarity distribution:")
        for range_key, count in semantic_stats['similarity_distribution'].items():
            if count > 0:
                print(f"    {range_key}: {count} edges")
    
    # Dependency statistics
    if verbose:
        dep_stats = stats['dependency_statistics']
        print(f"\n🔗 Dependency Parsing:")
        print(f"  Total dependency edges: {dep_stats['total_dependency_edges']}")
        print(f"  Dependency types: {len(dep_stats['dependency_types'])}")
        
        print("  Most common dependencies:")
        for dep_type, count in dep_stats['most_common_dependencies'][:8]:
            print(f"    '{dep_type}': {count} edges")
        
        print(f"\n📊 Most Frequent Words:")
        for word, freq in stats['most_frequent_words']:
            print(f"  '{word}': {freq} times")

def save_outputs(text_graph, args):
    """Save various output formats"""
    config = load_config()
    outputs_saved = []
    
    # Save graph file
    export_graph = getattr(args, 'export_graph', config['export_graph'])
    if export_graph:
        try:
            text_graph.save_graph(export_graph)
            outputs_saved.append(f"Graph: {export_graph}")
        except Exception as e:
            print(f"⚠️ Failed to save graph: {e}")
    
    # Save JSON data
    if getattr(args, 'export_json', None):
        try:
            json_data = text_graph.export_to_json()
            with open(args.export_json, 'w', encoding='utf-8') as f:
                f.write(json_data)
            outputs_saved.append(f"JSON: {args.export_json}")
        except Exception as e:
            print(f"⚠️ Failed to save JSON: {e}")
    
    # Handle visualization
    if not getattr(args, 'disable_visualization', False):
        try:
            # Parse figure size
            if hasattr(args, 'figure_size'):
                fig_width, fig_height = map(float, args.figure_size.split(','))
            else:
                fig_width, fig_height = config['figure_width'], config['figure_height']
            
            # Create visualization
            text_graph.visualize(
                figsize=(fig_width, fig_height),
                show_dependencies=not getattr(args, 'disable_dependencies', False),
                show_semantic=not getattr(args, 'disable_semantic', False)
            )
            
            # Save image if specified
            if getattr(args, 'export_image', None):
                plt.savefig(
                    args.export_image, 
                    dpi=config['dpi'], 
                    bbox_inches='tight', 
                    facecolor='white'
                )
                outputs_saved.append(f"Image: {args.export_image}")
                
                # Don't show plot if saving to file
                plt.close()
            else:
                # Show plot
                plt.show()
                
        except Exception as e:
            print(f"⚠️ Visualization failed: {e}")
    
    # Print saved outputs
    if outputs_saved and not getattr(args, 'quiet', False):
        print(f"💾 Saved outputs:")
        for output in outputs_saved:
            print(f"  ✅ {output}")

def load_sample_data():
    """Load sample data for demo (deprecated, use load_demo_data instead)"""
    return load_demo_data()

def download_vncorenlp(target_dir="vncorenlp", verbose=False):
    """Download and setup VnCoreNLP automatically using py_vncorenlp"""
    
    # Convert to absolute path
    if not os.path.isabs(target_dir):
        target_dir = os.path.abspath(target_dir)
    
    # Check if already exists
    jar_path = os.path.join(target_dir, "VnCoreNLP-1.2.jar")
    models_path = os.path.join(target_dir, "models")
    
    if os.path.exists(jar_path) and os.path.exists(models_path):
        if verbose:
            print(f"✅ VnCoreNLP already exists at: {target_dir}")
        return target_dir
    
    if verbose:
        print(f"📥 Downloading VnCoreNLP to: {target_dir}")
    
    # Create directory if it doesn't exist
    os.makedirs(target_dir, exist_ok=True)
    
    try:
        # Use py_vncorenlp's built-in download function
        import py_vncorenlp
        py_vncorenlp.download_model(save_dir=target_dir)
        
        if verbose:
            print("  ✅ VnCoreNLP downloaded successfully!")
        
        return target_dir
        
    except Exception as e:
        raise RuntimeError(f"Failed to download VnCoreNLP: {e}") 