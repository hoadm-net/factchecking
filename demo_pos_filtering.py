#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Demo script cho tính năng lọc từ loại (POS filtering) trong MINT TextGraph
Minh họa sự khác biệt giữa đồ thị có và không có lọc từ loại
"""

import json
from mint.text_graph import TextGraph
from mint.helpers import setup_vncorenlp, process_text_data

def compare_graphs():
    """So sánh đồ thị với và không có POS filtering"""
    
    # Load demo data
    with open('data/demo.json', 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    context = data['context']
    claim = data['claim']
    
    print("🔥 DEMO: So sánh đồ thị có và không có POS filtering")
    print("="*60)
    
    # Setup VnCoreNLP
    print("📥 Đang setup VnCoreNLP...")
    model = setup_vncorenlp("vncorenlp", verbose=True)
    
    # Process text
    print("🔍 Đang xử lý văn bản...")
    context_sentences, claim_sentences = process_text_data(model, context, claim, verbose=True)
    
    # Build graph WITHOUT POS filtering
    print("\n📊 Xây dựng đồ thị KHÔNG có lọc từ loại...")
    graph_no_filter = TextGraph()
    graph_no_filter.set_pos_filtering(enable=False)
    graph_no_filter.build_from_vncorenlp_output(context_sentences, claim, claim_sentences)
    stats_no_filter = graph_no_filter.get_statistics()
    
    # Build graph WITH POS filtering  
    print("📊 Xây dựng đồ thị CÓ lọc từ loại...")
    graph_with_filter = TextGraph()
    graph_with_filter.set_pos_filtering(enable=True)
    graph_with_filter.build_from_vncorenlp_output(context_sentences, claim, claim_sentences)
    stats_with_filter = graph_with_filter.get_statistics()
    
    # Compare results
    print("\n" + "="*60)
    print("📈 KẾT QUẢ SO SÁNH")
    print("="*60)
    
    print("🔸 KHÔNG lọc từ loại:")
    print(f"  Total nodes: {stats_no_filter['total_nodes']}")
    print(f"  Word nodes: {stats_no_filter['word_nodes']}")
    print(f"  Total edges: {stats_no_filter['total_edges']}")
    
    print("\n🔹 CÓ lọc từ loại (N,Np,V,A,Nc,M,R,P):")
    print(f"  Total nodes: {stats_with_filter['total_nodes']}")
    print(f"  Word nodes: {stats_with_filter['word_nodes']}")
    print(f"  Total edges: {stats_with_filter['total_edges']}")
    
    # Calculate reduction
    node_reduction = ((stats_no_filter['total_nodes'] - stats_with_filter['total_nodes']) / 
                     stats_no_filter['total_nodes'] * 100)
    word_reduction = ((stats_no_filter['word_nodes'] - stats_with_filter['word_nodes']) / 
                     stats_no_filter['word_nodes'] * 100)
    edge_reduction = ((stats_no_filter['total_edges'] - stats_with_filter['total_edges']) / 
                     stats_no_filter['total_edges'] * 100)
    
    print(f"\n✨ HIỆU QUẢ GIẢM NHIỄU:")
    print(f"  Giảm {node_reduction:.1f}% tổng số nodes")
    print(f"  Giảm {word_reduction:.1f}% word nodes")
    print(f"  Giảm {edge_reduction:.1f}% tổng số edges")
    
    # Show filtered vs kept words
    print(f"\n🔍 PHÂN TÍCH CHI TIẾT:")
    show_filtered_words_analysis(context_sentences, claim_sentences)
    
    print(f"\n💡 KẾT LUẬN:")
    print(f"  - POS filtering giúp giảm đáng kể số lượng nodes và edges")
    print(f"  - Loại bỏ từ chức năng, dấu câu, giữ lại từ có ý nghĩa")
    print(f"  - Cải thiện hiệu suất xử lý và chất lượng phân tích")

def show_filtered_words_analysis(context_sentences, claim_sentences):
    """Phân tích chi tiết các từ được lọc và giữ lại"""
    
    important_pos_tags = {'N', 'Np', 'V', 'A', 'Nc', 'M', 'R', 'P'}
    
    kept_words = []
    filtered_words = []
    
    # Analyze all tokens
    all_tokens = []
    for sent_tokens in context_sentences.values():
        all_tokens.extend(sent_tokens)
    for sent_tokens in claim_sentences.values():
        all_tokens.extend(sent_tokens)
    
    for token in all_tokens:
        word = token["wordForm"]
        pos_tag = token.get("posTag", "")
        
        if pos_tag in important_pos_tags:
            kept_words.append((word, pos_tag))
        else:
            filtered_words.append((word, pos_tag))
    
    print(f"  Từ được GIỮ LẠI: {len(kept_words)} từ")
    kept_sample = kept_words[:10]
    for word, pos in kept_sample:
        print(f"    '{word}' ({pos})")
    if len(kept_words) > 10:
        print(f"    ... và {len(kept_words) - 10} từ khác")
    
    print(f"  Từ bị LỌC BỎ: {len(filtered_words)} từ")
    filtered_sample = filtered_words[:10]
    for word, pos in filtered_sample:
        print(f"    '{word}' ({pos})")
    if len(filtered_words) > 10:
        print(f"    ... và {len(filtered_words) - 10} từ khác")

if __name__ == "__main__":
    compare_graphs() 