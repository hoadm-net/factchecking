#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
SHORTCUT SCRIPT: Extract sentences từ kết quả beam search
Tự động tìm file beam search mới nhất và extract sentences
"""

import os
import sys
from analyze_beam_sentences import BeamSearchSentenceAnalyzer, find_latest_beam_file

def main():
    print("🎯 EXTRACT SENTENCES FROM BEAM SEARCH")
    print("="*50)
    
    # Find latest beam search file
    print("🔍 Looking for latest beam search file...")
    
    # Try multiple possible locations and find the most interesting file
    possible_dirs = [
        "output",
        "/home/hoadm/FactChecking/output",
        "vncorenlp/output"
    ]
    
    # Find all beam search files and pick the one with most data
    all_files = []
    for dir_path in possible_dirs:
        if os.path.exists(dir_path):
            try:
                beam_files = [
                    os.path.join(dir_path, f) for f in os.listdir(dir_path) 
                    if f.startswith('beam_search_') and f.endswith('.json')
                ]
                all_files.extend(beam_files)
            except:
                continue
    
    # Sort by modification time and pick most recent
    if all_files:
        all_files.sort(key=lambda x: os.path.getmtime(x), reverse=True)
        
        # Try to find file with more than 1 path
        input_file = None
        for file in all_files:
            try:
                import json
                with open(file, 'r') as f:
                    data = json.load(f)
                paths_count = data.get('total_paths_found', len(data.get('paths', [])))
                if paths_count > 1:  # Prefer files with multiple paths
                    input_file = file
                    break
            except:
                continue
        
        # Fallback to most recent file if no multi-path file found
        if not input_file:
            input_file = all_files[0]
    else:
        input_file = None
    
    if not input_file:
        print("❌ No beam search files found!")
        print("💡 Run beam search first:")
        print("   python3 main.py --demo --beam-search --verbose")
        return
    
    print(f"✅ Found: {input_file}")
    
    # Analyze sentences
    analyzer = BeamSearchSentenceAnalyzer()
    
    # Load data
    beam_data = analyzer.load_beam_results(input_file)
    if not beam_data:
        return
    
    # Analyze sentences
    sentences_stats = analyzer.analyze_sentences(beam_data)
    if not sentences_stats:
        return
    
    # Rank by frequency (most common method)
    ranked_sentences = analyzer.rank_sentences(sentences_stats, 'frequency')
    
    # Show results in terminal
    print(f"\n🏆 TOP 10 SENTENCES (by frequency):")
    print("="*80)
    
    for i, (sentence_id, stats) in enumerate(ranked_sentences[:10], 1):
        freq = stats['frequency']
        text = stats['sentence_text']
        
        # Truncate long sentences
        if len(text) > 70:
            text = text[:67] + "..."
        
        print(f"{i:2d}. [freq:{freq}] {text}")
    
    # Export simple list
    simple_file = analyzer.export_simple_list(ranked_sentences, ranking_method='frequency')
    
    print(f"\n💾 Sentences extracted and saved to:")
    print(f"   {simple_file}")
    
    # Also show statistics
    total_sentences = len(ranked_sentences)
    max_freq = max(stats['frequency'] for _, stats in ranked_sentences)
    avg_freq = sum(stats['frequency'] for _, stats in ranked_sentences) / total_sentences
    
    print(f"\n📊 Statistics:")
    print(f"   Total unique sentences: {total_sentences}")
    print(f"   Max frequency: {max_freq}")
    print(f"   Average frequency: {avg_freq:.1f}")
    
    # Show path analysis
    total_paths = beam_data.get('total_paths_found', len(beam_data.get('paths', [])))
    print(f"   Total paths analyzed: {total_paths}")
    
    print(f"\n✅ Done! Check {simple_file} for complete ranked list.")

if __name__ == "__main__":
    main() 