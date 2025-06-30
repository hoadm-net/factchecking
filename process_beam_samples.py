#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Process beam search samples and organize them into structured data
Extract claims and related sentences from beam search results for further analysis
"""

import os
import json
from glob import glob
from typing import Dict, List, Set, Any
from collections import defaultdict

def extract_claim_and_sentences(data: Dict) -> Dict[str, List[str]]:
    """
    Extract claim text and associated sentences from beam search data
    Returns a dictionary with claim as key and list of unique sentences as value
    """
    # Extract claim text
    claim_text = "Unknown Claim"
    sentences_set: Set[str] = set()
    
    # Get claim from metadata or paths
    metadata = data.get('metadata', {})
    if 'claim' in metadata:
        claim_text = metadata['claim']
    
    # If no claim in metadata, try to extract from node_details
    if claim_text == "Unknown Claim":
        for path in data.get('paths', []):
            for node in path.get('node_details', []):
                if node.get('id', '').startswith('claim_'):
                    claim_text = node.get('text', 'Unknown Claim')
                    break
            if claim_text != "Unknown Claim":
                break
    
    # Extract all unique sentence texts from paths
    for path in data.get('paths', []):
        # From node_details (preferred method as it has full text)
        for node in path.get('node_details', []):
            if node.get('id', '').startswith('sentence_'):
                sentence_text = node.get('text', '').strip()
                if sentence_text:
                    sentences_set.add(sentence_text)
        
        # Fallback: if no node_details, try to extract from nodes list
        # This is less reliable but provides backup
        if not path.get('node_details'):
            nodes = path.get('nodes', [])
            for node_id in nodes:
                if node_id.startswith('sentence_'):
                    # We don't have the actual text here, just mark that this sentence exists
                    sentences_set.add(f"[Sentence {node_id}]")
    
    # Convert set to sorted list
    sentences_list = sorted(list(sentences_set))
    
    return {claim_text: sentences_list}

def process_beam_samples(output_dir: str = "output") -> Dict[str, List[str]]:
    """
    Process beam search files and organize them into samples
    Returns a dictionary where:
    - key: claim text
    - value: list of sentence texts associated with that claim
    """
    if not os.path.exists(output_dir):
        print(f"❌ Output directory '{output_dir}' does not exist")
        return {}
    
    samples_dict = {}
    pattern = os.path.join(output_dir, "beam_search_*.json")
    json_files = glob(pattern)
    
    # Filter out summary, analysis, and pattern files
    json_files = [f for f in json_files if not any(
        skip in f for skip in ["summary", "analysis", "patterns", "stats"]
    )]
    
    if not json_files:
        print(f"⚠️ No beam search JSON files found in '{output_dir}' directory")
        return {}
    
    print(f"📁 Found {len(json_files)} beam search files to process")
    
    processed_count = 0
    error_count = 0
    
    for json_file in sorted(json_files):
        try:
            # Extract sample index from filename for tracking
            filename = os.path.basename(json_file)
            sample_idx = filename.replace("beam_search_", "").replace(".json", "")
            
            # Read and process the json file
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Extract claim and sentences
            claim_sentences = extract_claim_and_sentences(data)
            
            # Merge into main dictionary
            for claim, sentences in claim_sentences.items():
                if claim in samples_dict:
                    # Merge sentences, avoiding duplicates
                    existing_sentences = set(samples_dict[claim])
                    new_sentences = set(sentences)
                    combined_sentences = existing_sentences.union(new_sentences)
                    samples_dict[claim] = sorted(list(combined_sentences))
                else:
                    samples_dict[claim] = sentences
            
            processed_count += 1
            if processed_count % 10 == 0:
                print(f"   Processed {processed_count}/{len(json_files)} files...")
            
        except Exception as e:
            print(f"❌ Error processing {json_file}: {e}")
            error_count += 1
            continue
    
    print(f"\n✅ Processing completed:")
    print(f"   Successfully processed: {processed_count} files")
    print(f"   Errors encountered: {error_count} files")
    print(f"   Unique claims found: {len(samples_dict)}")
    
    return samples_dict

def validate_samples_dict(samples_dict: Dict[str, List[str]]) -> Dict[str, Any]:
    """Validate and analyze the processed samples dictionary"""
    validation_stats = {
        'total_claims': len(samples_dict),
        'claims_with_sentences': 0,
        'total_sentences': 0,
        'avg_sentences_per_claim': 0.0,
        'claims_without_sentences': [],
        'longest_claim': '',
        'most_sentences_claim': '',
        'max_sentences_count': 0
    }
    
    total_sentences = 0
    claims_with_sentences = 0
    max_sentences = 0
    longest_claim = ''
    most_sentences_claim = ''
    
    for claim, sentences in samples_dict.items():
        sentence_count = len(sentences)
        total_sentences += sentence_count
        
        if sentence_count > 0:
            claims_with_sentences += 1
        else:
            validation_stats['claims_without_sentences'].append(claim)
        
        if sentence_count > max_sentences:
            max_sentences = sentence_count
            most_sentences_claim = claim
        
        if len(claim) > len(longest_claim):
            longest_claim = claim
    
    validation_stats.update({
        'claims_with_sentences': claims_with_sentences,
        'total_sentences': total_sentences,
        'avg_sentences_per_claim': total_sentences / max(len(samples_dict), 1),
        'longest_claim': longest_claim,
        'most_sentences_claim': most_sentences_claim,
        'max_sentences_count': max_sentences
    })
    
    return validation_stats

def save_samples_dict(samples_dict: Dict[str, List[str]], output_file: str = "beam_samples_organized.json"):
    """Save the organized samples dictionary to a JSON file with validation"""
    try:
        # Create backup if file exists
        if os.path.exists(output_file):
            backup_file = output_file.replace('.json', '_backup.json')
            os.rename(output_file, backup_file)
            print(f"📁 Created backup: {backup_file}")
        
        # Save the main file
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(samples_dict, f, indent=2, ensure_ascii=False)
        
        print(f"✅ Saved organized samples to {output_file}")
        
        # Save validation report
        validation_stats = validate_samples_dict(samples_dict)
        
        report_file = output_file.replace('.json', '_report.txt')
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("📊 BEAM SAMPLES PROCESSING REPORT\n")
            f.write("=" * 40 + "\n\n")
            f.write(f"Total Claims: {validation_stats['total_claims']}\n")
            f.write(f"Claims with Sentences: {validation_stats['claims_with_sentences']}\n")
            f.write(f"Total Sentences: {validation_stats['total_sentences']}\n")
            f.write(f"Average Sentences per Claim: {validation_stats['avg_sentences_per_claim']:.2f}\n")
            f.write(f"Max Sentences in Single Claim: {validation_stats['max_sentences_count']}\n\n")
            
            if validation_stats['claims_without_sentences']:
                f.write(f"⚠️ Claims without sentences ({len(validation_stats['claims_without_sentences'])}):\n")
                for claim in validation_stats['claims_without_sentences']:
                    f.write(f"  - {claim[:100]}...\n")
                f.write("\n")
            
            f.write(f"📏 Longest claim: {validation_stats['longest_claim'][:200]}...\n")
            f.write(f"📈 Claim with most sentences: {validation_stats['most_sentences_claim'][:200]}...\n")
        
        print(f"📊 Saved processing report to {report_file}")
        
    except Exception as e:
        print(f"❌ Error saving file: {e}")

def create_summary_statistics(samples_dict: Dict[str, List[str]]) -> None:
    """Print summary statistics about the processed samples"""
    if not samples_dict:
        print("⚠️ No samples to analyze")
        return
    
    validation_stats = validate_samples_dict(samples_dict)
    
    print("\n📊 PROCESSING SUMMARY:")
    print("=" * 40)
    print(f"📋 Total unique claims: {validation_stats['total_claims']}")
    print(f"📝 Claims with sentences: {validation_stats['claims_with_sentences']}")
    print(f"📄 Total sentences across all claims: {validation_stats['total_sentences']}")
    print(f"📊 Average sentences per claim: {validation_stats['avg_sentences_per_claim']:.2f}")
    print(f"📈 Maximum sentences in a single claim: {validation_stats['max_sentences_count']}")
    
    if validation_stats['claims_without_sentences']:
        print(f"⚠️ Claims without sentences: {len(validation_stats['claims_without_sentences'])}")
    
    print("\n🔍 SAMPLE PREVIEW:")
    print("-" * 40)
    for i, (claim, sentences) in enumerate(list(samples_dict.items())[:3]):
        print(f"\nSample {i+1}:")
        print(f"Claim: {claim[:100]}{'...' if len(claim) > 100 else ''}")
        print(f"Sentences ({len(sentences)}):")
        for j, sentence in enumerate(sentences[:2]):  # Show first 2 sentences
            print(f"  {j+1}. {sentence[:80]}{'...' if len(sentence) > 80 else ''}")
        if len(sentences) > 2:
            print(f"  ... and {len(sentences) - 2} more sentences")

def main():
    """Main processing function"""
    print("🔄 Starting Beam Search Samples Processing...")
    
    try:
        # Process all samples
        samples_dict = process_beam_samples()
        
        if not samples_dict:
            print("❌ No samples were processed successfully")
            return
        
        # Show summary statistics
        create_summary_statistics(samples_dict)
        
        # Save to file
        save_samples_dict(samples_dict)
        
        print("\n✅ Processing completed successfully!")
        
    except Exception as e:
        print(f"❌ Fatal error during processing: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 