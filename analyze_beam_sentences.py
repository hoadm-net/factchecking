#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Phân tích kết quả Beam Search và trích xuất danh sách câu được xếp hạng
Đọc file JSON từ beam search, lấy sentences không trùng lặp, xếp theo thứ tự quan trọng
"""

import json
import os
import argparse
from collections import defaultdict, Counter
from datetime import datetime
from typing import Dict, List, Tuple


class BeamSearchSentenceAnalyzer:
    """Phân tích sentences từ kết quả beam search"""
    
    def __init__(self):
        self.sentences_data = {}  # sentence_id -> sentence_info
        self.sentence_stats = defaultdict(lambda: {
            'frequency': 0,
            'total_score': 0.0,
            'avg_score': 0.0,
            'max_score': 0.0,
            'min_score': float('inf'),
            'paths_count': 0,
            'sentence_text': '',
            'sentence_id': ''
        })
    
    def load_beam_results(self, json_file: str) -> Dict:
        """Đọc file JSON kết quả beam search"""
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            print(f"✅ Loaded beam search results from: {json_file}")
            print(f"   Total paths: {data.get('total_paths_found', 0)}")
            print(f"   Search config: beam_width={data.get('search_config', {}).get('beam_width', 'N/A')}")
            return data
        except FileNotFoundError:
            print(f"❌ File not found: {json_file}")
            return {}
        except json.JSONDecodeError as e:
            print(f"❌ JSON decode error: {e}")
            return {}
    
    def analyze_sentences(self, beam_data: Dict) -> Dict[str, Dict]:
        """Phân tích và xếp hạng sentences từ beam search paths"""
        if not beam_data or 'paths' not in beam_data:
            print("❌ No valid beam search data to analyze")
            return {}
        
        print(f"\n🔍 Analyzing {len(beam_data['paths'])} paths...")
        
        # Reset data
        self.sentence_stats.clear()
        
        # Process each path
        for path_idx, path in enumerate(beam_data['paths']):
            path_score = path.get('score', 0.0)
            node_details = path.get('node_details', [])
            
            # Find sentence nodes in this path
            sentence_nodes = [
                node for node in node_details 
                if node.get('type', '').lower() == 'sentence'
            ]
            
            # Update stats for each sentence in this path
            for sentence_node in sentence_nodes:
                sentence_id = sentence_node.get('id', '')
                sentence_text = sentence_node.get('text', '').strip()
                
                if sentence_id and sentence_text:
                    stats = self.sentence_stats[sentence_id]
                    
                    # Update statistics
                    stats['frequency'] += 1
                    stats['total_score'] += path_score
                    stats['max_score'] = max(stats['max_score'], path_score)
                    stats['min_score'] = min(stats['min_score'], path_score)
                    stats['paths_count'] += 1
                    stats['sentence_text'] = sentence_text
                    stats['sentence_id'] = sentence_id
        
        # Calculate average scores
        for sentence_id, stats in self.sentence_stats.items():
            if stats['frequency'] > 0:
                stats['avg_score'] = stats['total_score'] / stats['frequency']
                if stats['min_score'] == float('inf'):
                    stats['min_score'] = 0.0
        
        print(f"✅ Found {len(self.sentence_stats)} unique sentences")
        return dict(self.sentence_stats)
    
    def rank_sentences(self, sentences_stats: Dict, ranking_method: str = 'frequency') -> List[Tuple]:
        """Xếp hạng sentences theo các tiêu chí khác nhau"""
        if not sentences_stats:
            return []
        
        ranking_functions = {
            'frequency': lambda x: x[1]['frequency'],
            'avg_score': lambda x: x[1]['avg_score'],
            'max_score': lambda x: x[1]['max_score'],
            'total_score': lambda x: x[1]['total_score'],
            'combined': lambda x: x[1]['frequency'] * x[1]['avg_score']  # Frequency × Average Score
        }
        
        if ranking_method not in ranking_functions:
            print(f"⚠️ Unknown ranking method: {ranking_method}. Using 'frequency'")
            ranking_method = 'frequency'
        
        # Sort sentences by the chosen ranking method
        ranked_sentences = sorted(
            sentences_stats.items(),
            key=ranking_functions[ranking_method],
            reverse=True
        )
        
        print(f"📊 Ranked sentences by: {ranking_method}")
        return ranked_sentences
    
    def export_ranked_sentences(self, ranked_sentences: List[Tuple], 
                               output_file: str = None, 
                               ranking_method: str = 'frequency') -> str:
        """Export danh sách sentences đã xếp hạng"""
        if output_file is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = f"output/ranked_sentences_{ranking_method}_{timestamp}.txt"
        
        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("📊 BEAM SEARCH - RANKED SENTENCES ANALYSIS\n")
            f.write("="*70 + "\n\n")
            
            f.write(f"Ranking Method: {ranking_method.upper()}\n")
            f.write(f"Total Unique Sentences: {len(ranked_sentences)}\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # Summary statistics
            if ranked_sentences:
                frequencies = [stats['frequency'] for _, stats in ranked_sentences]
                avg_scores = [stats['avg_score'] for _, stats in ranked_sentences]
                
                f.write("📈 SUMMARY STATISTICS:\n")
                f.write("-" * 40 + "\n")
                f.write(f"  Max frequency: {max(frequencies)}\n")
                f.write(f"  Min frequency: {min(frequencies)}\n")
                f.write(f"  Avg frequency: {sum(frequencies)/len(frequencies):.1f}\n")
                f.write(f"  Max avg_score: {max(avg_scores):.3f}\n")
                f.write(f"  Min avg_score: {min(avg_scores):.3f}\n")
                f.write(f"  Overall avg_score: {sum(avg_scores)/len(avg_scores):.3f}\n\n")
            
            # Detailed ranking
            f.write("🏆 RANKED SENTENCES:\n")
            f.write("="*70 + "\n\n")
            
            for rank, (sentence_id, stats) in enumerate(ranked_sentences, 1):
                f.write(f"RANK #{rank:2d} │ ")
                
                if ranking_method == 'frequency':
                    f.write(f"Frequency: {stats['frequency']:2d}")
                elif ranking_method == 'avg_score':
                    f.write(f"Avg Score: {stats['avg_score']:5.3f}")
                elif ranking_method == 'max_score':
                    f.write(f"Max Score: {stats['max_score']:5.3f}")
                elif ranking_method == 'total_score':
                    f.write(f"Total Score: {stats['total_score']:6.3f}")
                elif ranking_method == 'combined':
                    combined_score = stats['frequency'] * stats['avg_score']
                    f.write(f"Combined: {combined_score:6.3f}")
                
                f.write(f" │ Paths: {stats['paths_count']:2d}\n")
                f.write("-" * 70 + "\n")
                
                # Sentence text (truncated for readability)
                sentence_text = stats['sentence_text']
                if len(sentence_text) > 80:
                    sentence_text = sentence_text[:77] + "..."
                f.write(f"Text: {sentence_text}\n")
                
                # Detailed stats
                f.write(f"Stats: freq={stats['frequency']}, ")
                f.write(f"avg_score={stats['avg_score']:.3f}, ")
                f.write(f"score_range=({stats['min_score']:.3f}-{stats['max_score']:.3f})\n")
                f.write(f"ID: {sentence_id}\n")
                f.write("\n" + "="*70 + "\n\n")
        
        print(f"💾 Exported ranked sentences to: {output_file}")
        return output_file
    
    def export_simple_list(self, ranked_sentences: List[Tuple], 
                          output_file: str = None,
                          ranking_method: str = 'frequency') -> str:
        """Export danh sách sentences đơn giản (chỉ text)"""
        if output_file is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = f"output/simple_sentences_{ranking_method}_{timestamp}.txt"
        
        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(f"# Sentences ranked by {ranking_method}\n")
            f.write(f"# Total: {len(ranked_sentences)} unique sentences\n")
            f.write(f"# Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            for rank, (sentence_id, stats) in enumerate(ranked_sentences, 1):
                ranking_value = stats.get(ranking_method, stats['frequency'])
                f.write(f"{rank:2d}. [{ranking_value:5.1f}] {stats['sentence_text']}\n")
        
        print(f"📝 Exported simple sentence list to: {output_file}")
        return output_file


def find_latest_beam_file(output_dir: str = "output") -> str:
    """Tìm file beam search mới nhất"""
    try:
        beam_files = [
            f for f in os.listdir(output_dir) 
            if f.startswith('beam_search_') and f.endswith('.json')
        ]
        if not beam_files:
            return ""
        
        # Sort by timestamp in filename
        beam_files.sort(reverse=True)
        latest_file = os.path.join(output_dir, beam_files[0])
        print(f"🔍 Found latest beam search file: {latest_file}")
        return latest_file
    except FileNotFoundError:
        return ""


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Analyze beam search results and rank sentences')
    parser.add_argument('--input', '-i', type=str, help='Input beam search JSON file')
    parser.add_argument('--output-dir', '-o', type=str, default='output', 
                       help='Output directory (default: output)')
    parser.add_argument('--ranking', '-r', type=str, default='frequency',
                       choices=['frequency', 'avg_score', 'max_score', 'total_score', 'combined'],
                       help='Ranking method (default: frequency)')
    parser.add_argument('--simple', '-s', action='store_true',
                       help='Also export simple sentence list')
    parser.add_argument('--auto', '-a', action='store_true',
                       help='Auto-find latest beam search file')
    
    args = parser.parse_args()
    
    print("📊 BEAM SEARCH SENTENCE ANALYZER")
    print("="*50)
    
    # Determine input file
    input_file = args.input
    if args.auto or not input_file:
        input_file = find_latest_beam_file(args.output_dir)
        if not input_file:
            print("❌ No beam search files found. Run beam search first!")
            print("   Example: python3 main.py --demo --beam-search --verbose")
            return
    
    if not os.path.exists(input_file):
        print(f"❌ Input file not found: {input_file}")
        return
    
    # Analyze sentences
    analyzer = BeamSearchSentenceAnalyzer()
    
    # Load and analyze
    beam_data = analyzer.load_beam_results(input_file)
    if not beam_data:
        return
    
    sentences_stats = analyzer.analyze_sentences(beam_data)
    if not sentences_stats:
        return
    
    # Rank sentences
    ranked_sentences = analyzer.rank_sentences(sentences_stats, args.ranking)
    
    # Show top 5 sentences
    print(f"\n🏆 TOP 5 SENTENCES (by {args.ranking}):")
    print("-" * 60)
    for i, (sentence_id, stats) in enumerate(ranked_sentences[:5], 1):
        ranking_value = stats.get(args.ranking, stats['frequency'])
        sentence_preview = stats['sentence_text'][:50] + "..." if len(stats['sentence_text']) > 50 else stats['sentence_text']
        print(f"{i}. [{ranking_value:5.1f}] {sentence_preview}")
    
    # Export results
    print(f"\n💾 Exporting results...")
    
    # Detailed analysis
    detailed_file = analyzer.export_ranked_sentences(
        ranked_sentences, 
        ranking_method=args.ranking
    )
    
    # Simple list (optional)
    if args.simple:
        simple_file = analyzer.export_simple_list(
            ranked_sentences,
            ranking_method=args.ranking
        )
    
    print(f"\n✅ Analysis completed!")
    print(f"   Input: {input_file}")
    print(f"   Detailed output: {detailed_file}")
    if args.simple:
        print(f"   Simple list: {simple_file}")


if __name__ == "__main__":
    main() 