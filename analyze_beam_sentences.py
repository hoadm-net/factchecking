#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Analyze beam search results from output directory
Provides comprehensive statistical analysis and visualizations
"""

import os
import json
import pandas as pd
import numpy as np
from glob import glob
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional

def load_beam_search_results(output_dir: str = "output") -> List[Dict]:
    """Load all beam search JSON files from output directory"""
    results = []
    pattern = os.path.join(output_dir, "beam_search_*.json")
    
    for json_file in sorted(glob(pattern)):
        # Skip summary and patterns files
        if any(skip in json_file for skip in ["summary", "patterns", "analysis"]):
            continue
            
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
                # Extract sample index from filename
                filename = os.path.basename(json_file)
                sample_idx = filename.replace("beam_search_", "").replace(".json", "")
                
                # Add metadata
                data['filename'] = filename
                data['sample_index'] = sample_idx
                results.append(data)
                
            print(f"✅ Loaded {json_file}")
        except Exception as e:
            print(f"❌ Error loading {json_file}: {e}")
    
    if not results:
        print("⚠️ No beam search results found. Make sure you have run beam search first.")
        return []
        
    print(f"\n📊 Successfully loaded {len(results)} beam search samples")
    return results

def analyze_paths(beam_results: Dict) -> Dict:
    """Analyze paths from a single beam search result"""
    paths = beam_results.get('paths', [])
    if not paths:
        return {
            'sample_index': beam_results.get('sample_index', 'unknown'),
            'filename': beam_results.get('filename', 'unknown'),
            'total_paths': 0,
            'avg_score': 0.0,
            'max_score': 0.0,
            'min_score': 0.0,
            'avg_length': 0.0,
            'max_length': 0,
            'min_length': 0,
            'paths_to_sentences': 0,
            'paths_through_entities': 0,
            'sentence_reach_rate': 0.0,
            'entity_visit_rate': 0.0,
            'unique_sentences_reached': 0,
            'claim_word_coverage': 0.0
        }
    
    # Basic statistics
    scores = [p.get('score', 0.0) for p in paths]
    lengths = [len(p.get('nodes', [])) for p in paths]
    
    # Advanced analysis
    sentences_reached = set()
    entities_visited = set()
    paths_with_sentences = 0
    paths_with_entities = 0
    total_claim_words = 0
    total_matched_words = 0
    
    for path in paths:
        nodes = path.get('nodes', [])
        
        # Check for sentences and entities in path
        has_sentence = False
        has_entity = False
        
        for node in nodes:
            if node.startswith('sentence_'):
                sentences_reached.add(node)
                has_sentence = True
            elif node.startswith('entity_'):
                entities_visited.add(node)
                has_entity = True
        
        if has_sentence:
            paths_with_sentences += 1
        if has_entity:
            paths_with_entities += 1
            
        # Word matching analysis
        claim_words = path.get('total_claim_words', 0)
        matched_words = path.get('claim_words_matched', 0)
        total_claim_words = max(total_claim_words, claim_words)
        total_matched_words += matched_words
    
    # Calculate coverage
    avg_word_coverage = (total_matched_words / (len(paths) * max(total_claim_words, 1))) * 100
    
    analysis = {
        'sample_index': beam_results.get('sample_index', 'unknown'),
        'filename': beam_results.get('filename', 'unknown'),
        'total_paths': len(paths),
        'avg_score': np.mean(scores) if scores else 0.0,
        'max_score': max(scores) if scores else 0.0,
        'min_score': min(scores) if scores else 0.0,
        'std_score': np.std(scores) if scores else 0.0,
        'avg_length': np.mean(lengths) if lengths else 0.0,
        'max_length': max(lengths) if lengths else 0,
        'min_length': min(lengths) if lengths else 0,
        'std_length': np.std(lengths) if lengths else 0.0,
        'paths_to_sentences': paths_with_sentences,
        'paths_through_entities': paths_with_entities,
        'unique_sentences_reached': len(sentences_reached),
        'unique_entities_visited': len(entities_visited),
        'sentence_reach_rate': paths_with_sentences / len(paths) if paths else 0.0,
        'entity_visit_rate': paths_with_entities / len(paths) if paths else 0.0,
        'claim_word_coverage': avg_word_coverage,
        'beam_width': beam_results.get('metadata', {}).get('beam_width', 10),
        'max_depth': beam_results.get('metadata', {}).get('max_depth', 6)
    }
    
    return analysis

def create_summary_dataframe(all_analyses: List[Dict]) -> pd.DataFrame:
    """Create a DataFrame from all analyses"""
    if not all_analyses:
        return pd.DataFrame()
    return pd.DataFrame(all_analyses)

def plot_metrics_over_time(df: pd.DataFrame, save_dir: str = "output"):
    """Plot various metrics over time with improved visualizations"""
    if df.empty:
        print("⚠️ No data to plot")
        return
        
    # Ensure save directory exists
    os.makedirs(save_dir, exist_ok=True)
    
    # Set style
    plt.style.use('seaborn-v0_8' if 'seaborn-v0_8' in plt.style.available else 'default')
    
    metrics_groups = [
        ('scores', ['avg_score', 'max_score', 'min_score'], 'Beam Search Scores'),
        ('lengths', ['avg_length', 'max_length', 'min_length'], 'Path Lengths'),
        ('rates', ['sentence_reach_rate', 'entity_visit_rate'], 'Success Rates'),
        ('coverage', ['claim_word_coverage'], 'Word Coverage (%)'),
        ('diversity', ['unique_sentences_reached', 'unique_entities_visited'], 'Unique Nodes Reached')
    ]
    
    for group_name, columns, title in metrics_groups:
        # Check if columns exist in dataframe
        available_columns = [col for col in columns if col in df.columns]
        if not available_columns:
            continue
            
        plt.figure(figsize=(14, 8))
        
        for col in available_columns:
            plt.plot(df.index, df[col], marker='o', label=col.replace('_', ' ').title(), linewidth=2, markersize=4)
        
        plt.title(f'{title} Over Sample Index', fontsize=16, fontweight='bold')
        plt.xlabel('Sample Index', fontsize=12)
        plt.ylabel(title.split()[0], fontsize=12)
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # Save plot
        filename = os.path.join(save_dir, f'beam_search_{group_name}_analysis.png')
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"📊 Saved {filename}")

def analyze_path_patterns(beam_results: List[Dict]) -> Dict[str, Dict]:
    """Analyze common patterns in paths with enhanced pattern recognition"""
    patterns = defaultdict(int)
    detailed_patterns = defaultdict(int)
    
    for result in beam_results:
        for path in result.get('paths', []):
            nodes = path.get('nodes', [])
            
            # Basic node type sequence
            node_types = []
            for node in nodes:
                if node.startswith('claim'):
                    node_types.append('C')
                elif node.startswith('word_'):
                    node_types.append('W')
                elif node.startswith('sentence_'):
                    node_types.append('S')
                elif node.startswith('entity_'):
                    node_types.append('E')
                else:
                    node_types.append('?')
            
            # Create pattern string
            pattern = '->'.join(node_types)
            patterns[pattern] += 1
            
            # Detailed pattern with path success
            reaches_sentence = any(n.startswith('S') for n in node_types)
            has_entity = any(n.startswith('E') for n in node_types)
            
            pattern_key = f"{pattern} ({'SUCCESS' if reaches_sentence else 'PARTIAL'})"
            if has_entity:
                pattern_key += " +ENTITY"
            detailed_patterns[pattern_key] += 1
    
    return {
        'basic_patterns': dict(sorted(patterns.items(), key=lambda x: x[1], reverse=True)),
        'detailed_patterns': dict(sorted(detailed_patterns.items(), key=lambda x: x[1], reverse=True))
    }

def create_comprehensive_report(df: pd.DataFrame, patterns: Dict, save_dir: str = "output") -> str:
    """Create a comprehensive analysis report"""
    if df.empty:
        return "No data available for report generation."
    
    report_lines = [
        "📊 COMPREHENSIVE BEAM SEARCH ANALYSIS REPORT",
        "=" * 60,
        f"Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"Total Samples Analyzed: {len(df)}",
        "",
        "🎯 SUMMARY STATISTICS",
        "-" * 30
    ]
    
    # Overall statistics
    if not df.empty:
        report_lines.extend([
            f"Average Score: {df['avg_score'].mean():.3f} ± {df['avg_score'].std():.3f}",
            f"Average Path Length: {df['avg_length'].mean():.2f} ± {df['avg_length'].std():.2f}",
            f"Sentence Reach Rate: {df['sentence_reach_rate'].mean():.2%} ± {df['sentence_reach_rate'].std():.2%}",
            f"Entity Visit Rate: {df['entity_visit_rate'].mean():.2%} ± {df['entity_visit_rate'].std():.2%}",
            f"Word Coverage: {df['claim_word_coverage'].mean():.2f}% ± {df['claim_word_coverage'].std():.2f}%",
            ""
        ])
    
    # Top performing samples
    if 'avg_score' in df.columns:
        top_samples = df.nlargest(5, 'avg_score')[['sample_index', 'avg_score', 'sentence_reach_rate']]
        report_lines.extend([
            "🏆 TOP 5 PERFORMING SAMPLES",
            "-" * 30
        ])
        for _, row in top_samples.iterrows():
            report_lines.append(f"Sample {row['sample_index']}: Score {row['avg_score']:.3f}, Success Rate {row['sentence_reach_rate']:.1%}")
        report_lines.append("")
    
    # Pattern analysis
    basic_patterns = patterns.get('basic_patterns', {})
    if basic_patterns:
        report_lines.extend([
            "🔍 MOST COMMON PATH PATTERNS",
            "-" * 30
        ])
        for pattern, count in list(basic_patterns.items())[:10]:
            percentage = count / sum(basic_patterns.values()) * 100
            report_lines.append(f"{pattern}: {count} occurrences ({percentage:.1f}%)")
        report_lines.append("")
    
    # Performance insights
    if not df.empty:
        report_lines.extend([
            "💡 PERFORMANCE INSIGHTS",
            "-" * 30
        ])
        
        high_performers = df[df['sentence_reach_rate'] > 0.8]
        if not high_performers.empty:
            report_lines.append(f"• {len(high_performers)} samples achieved >80% sentence reach rate")
            report_lines.append(f"• These samples had average score: {high_performers['avg_score'].mean():.3f}")
        
        entity_rich = df[df['entity_visit_rate'] > 0.5]
        if not entity_rich.empty:
            report_lines.append(f"• {len(entity_rich)} samples had >50% entity visit rate")
            report_lines.append(f"• Entity-rich paths average score: {entity_rich['avg_score'].mean():.3f}")
        
        report_lines.append("")
    
    report_content = '\n'.join(report_lines)
    
    # Save report
    report_file = os.path.join(save_dir, "beam_search_comprehensive_report.txt")
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(report_content)
    
    return report_file

def main():
    """Main analysis function with enhanced error handling"""
    print("🔍 Analyzing beam search results...")
    
    try:
        # Load all results
        results = load_beam_search_results()
        if not results:
            print("❌ No beam search results found. Please run beam search first.")
            return
        
        # Analyze each result
        print("\n📊 Analyzing individual samples...")
        analyses = []
        for i, result in enumerate(results):
            try:
                analysis = analyze_paths(result)
                analyses.append(analysis)
                if (i + 1) % 10 == 0:
                    print(f"   Processed {i + 1}/{len(results)} samples...")
            except Exception as e:
                print(f"⚠️ Error analyzing sample {i}: {e}")
                continue
        
        if not analyses:
            print("❌ No valid analyses could be performed")
            return
        
        # Create DataFrame
        df = create_summary_dataframe(analyses)
        
        # Save detailed statistics
        stats_file = os.path.join("output", "beam_search_detailed_stats.csv")
        df.to_csv(stats_file, index=False)
        print(f"\n💾 Saved detailed statistics to {stats_file}")
        
        # Print summary
        print("\n📈 SUMMARY STATISTICS:")
        if not df.empty:
            summary_stats = df[['avg_score', 'avg_length', 'sentence_reach_rate', 'entity_visit_rate']].describe()
            print(summary_stats.round(3))
        
        # Create visualizations
        print("\n📊 Creating visualizations...")
        plot_metrics_over_time(df)
        
        # Analyze patterns
        print("\n🔍 Analyzing path patterns...")
        patterns = analyze_path_patterns(results)
        
        # Save patterns
        patterns_file = os.path.join("output", "beam_search_patterns.json")
        with open(patterns_file, 'w', encoding='utf-8') as f:
            json.dump(patterns, f, indent=2, ensure_ascii=False)
        print(f"💾 Saved path patterns to {patterns_file}")
        
        # Create comprehensive report
        report_file = create_comprehensive_report(df, patterns)
        print(f"📄 Saved comprehensive report to {report_file}")
        
        print("\n✅ Analysis completed successfully!")
        print(f"📁 All outputs saved to 'output/' directory")
        
    except Exception as e:
        print(f"❌ Fatal error during analysis: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 