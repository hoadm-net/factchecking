#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import py_vncorenlp
from os import path, getcwd
from mint import TextGraph

BASE_DIR = getcwd()
VNCORENLP_PATH = path.join(BASE_DIR, "vncorenlp")

def demo_semantic_similarity():
    """Demo tính năng semantic similarity trên một câu nhỏ"""
    
    # Sử dụng một câu nhỏ để test trên server không GPU
    context = """SAWACO thông báo tạm ngưng cung cấp nước để thực hiện công tác bảo trì."""
    
    claim = """SAWACO ngưng cấp nước để bảo trì hệ thống."""
    
    print("=== DEMO SEMANTIC SIMILARITY ===")
    print(f"Context: {context}")
    print(f"Claim: {claim}")
    print()
    
    try:
        # Khởi tạo VnCoreNLP
        model = py_vncorenlp.VnCoreNLP(save_dir=VNCORENLP_PATH)
        
        # Xử lý text
        print("Đang xử lý text với VnCoreNLP...")
        context_sentences = model.annotate_text(context)
        claim_sentences = model.annotate_text(claim)
        
        # Tạo TextGraph
        print("Đang khởi tạo TextGraph...")
        text_graph = TextGraph()
        
        # Build graph cơ bản
        print("Đang xây dựng graph cơ bản...")
        text_graph.build_from_vncorenlp_output(context_sentences, claim, claim_sentences)
        
        # In thống kê ban đầu
        stats = text_graph.get_detailed_statistics()
        print(f"\n=== THỐNG KÊ BAN ĐẦU ===")
        print(f"Tổng nodes: {stats['total_nodes']}")
        print(f"Word nodes: {stats['word_nodes']}")
        print(f"Tổng edges: {stats['total_edges']}")
        print(f"  - Structural: {stats['structural_edges']}")
        print(f"  - Dependency: {stats['dependency_edges']}")
        
        # Test semantic similarity với parameters thấp hơn
        print(f"\n=== DEMO SEMANTIC SIMILARITY ===")
        print("Đang thiết lập parameters cho demo...")
        
        # Giảm threshold và top_k để dễ test
        text_graph.similarity_threshold = 0.7  # Giảm từ 0.85 xuống 0.7
        text_graph.top_k_similar = 3  # Giảm từ 5 xuống 3
        text_graph.reduced_dim = 64   # Giảm từ 128 xuống 64
        
        print(f"Threshold: {text_graph.similarity_threshold}")
        print(f"Top-k: {text_graph.top_k_similar}")
        print(f"PCA dimensions: {text_graph.reduced_dim}")
        
        # Build semantic edges
        print("\nBắt đầu xây dựng semantic similarity edges...")
        edges_added = text_graph.build_semantic_similarity_edges(use_pca=True, use_faiss=True)
        
        # In thống kê sau khi thêm semantic edges
        stats_after = text_graph.get_detailed_statistics()
        semantic_stats = stats_after['semantic_statistics']
        
        print(f"\n=== KẾT QUẢ SEMANTIC SIMILARITY ===")
        print(f"Số edges được thêm: {edges_added}")
        print(f"Tổng semantic edges: {semantic_stats['total_semantic_edges']}")
        
        if semantic_stats['total_semantic_edges'] > 0:
            print(f"Similarity trung bình: {semantic_stats['average_similarity']:.3f}")
            print(f"Similarity cao nhất: {semantic_stats['max_similarity']:.3f}")
            print(f"Similarity thấp nhất: {semantic_stats['min_similarity']:.3f}")
            
            print("\nPhân bố similarity:")
            for range_key, count in semantic_stats['similarity_distribution'].items():
                if count > 0:
                    print(f"  {range_key}: {count} edges")
            
            # In một vài examples
            print(f"\n=== MỘT SỐ EDGES SEMANTIC ===")
            semantic_edges = [
                (u, v, data) for u, v, data in text_graph.graph.edges(data=True) 
                if data.get('edge_type') == 'semantic'
            ]
            
            for i, (u, v, data) in enumerate(semantic_edges[:5]):  # In tối đa 5 edges
                word1 = text_graph.graph.nodes[u]['text']
                word2 = text_graph.graph.nodes[v]['text']
                similarity = data.get('similarity', 0.0)
                print(f"  '{word1}' <-> '{word2}' (similarity: {similarity:.3f})")
        else:
            print("Không có semantic edges nào được tạo với threshold hiện tại.")
            print("Có thể giảm threshold hoặc tăng số từ trong text để có kết quả tốt hơn.")
        
        # Thống kê tổng quát
        print(f"\n=== THỐNG KÊ TỔNG QUÁT ===")
        print(f"Tổng nodes: {stats_after['total_nodes']}")
        print(f"Tổng edges: {stats_after['total_edges']}")
        print(f"  - Structural: {stats_after['structural_edges']}")
        print(f"  - Dependency: {stats_after['dependency_edges']}")
        print(f"  - Semantic: {stats_after['semantic_edges']}")
        
        # Visualization (optional - comment out nếu không muốn hiển thị)
        print(f"\n=== VISUALIZATION ===")
        print("Đang tạo visualization...")
        text_graph.visualize(figsize=(12, 8), show_dependencies=True, show_semantic=True)
        
        print("Demo hoàn thành!")
        
    except Exception as e:
        print(f"Lỗi trong quá trình demo: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    demo_semantic_similarity() 