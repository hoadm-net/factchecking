import py_vncorenlp
from os import path, getcwd
from mint import TextGraph


BASE_DIR = getcwd()
VNCORENLP_PATH = path.join(BASE_DIR, "vncorenlp")


if __name__ == "__main__":
    context = """(PLO)- Theo Tổng Công ty Cấp nước Sài Gòn (SAWACO) việc cúp nước là để thực hiện công tác bảo trì, bảo dưỡng định kỳ Nhà máy nước Tân Hiệp. SAWACO cho biết đây là phương án để đảm bảo cung cấp nước sạch an toàn, liên tục phục vụ cho người dân TP. Vì vậy, SAWACO thông báo tạm ngưng cung cấp nước để thực hiện công tác nêu trên. Thời gian thực hiện dự kiến từ 22 giờ ngày 25-3 (thứ bảy) đến 4 giờ ngày 26-3 (chủ nhật). Các khu vực tạm ngưng cung cấp nước gồm quận 6, 8, 12, Gò Vấp, Tân Bình, Tân Phú, Bình Tân và huyện Hóc Môn, Bình Chánh. SAWACO cho biết do điều kiện đặc thù của vùng cung cấp nước nên thời gian phục hồi nước trên mạng lưới cấp nước tại một số nơi xa nguồn sẽ chậm hơn so với mốc thời gian chính nêu trên. Theo đó, để hạn chế đến mức thấp nhất ảnh hưởng đến sinh hoạt của người dân, SAWACO đã có phương án tăng cường cấp nước bằng xe bồn tại các khu vực trọng yếu; điều tiết hỗ trợ từ mạng truyền tải, theo dõi chặt chẽ diễn biến trên mạng lưới cấp nước để điều phối nguồn nước theo tình hình thực tế. Đồng thời, SAWACO chủ động phối hợp giải quyết các vấn đề phát sinh xảy ra trên mạng lưới cấp nước."""

    claim = """SAWACO thông báo tạm ngưng cung cấp nước để thực hiện công tác bảo trì, bảo dưỡng định kỳ Nhà máy nước Tân Hiệp, thời gian thực hiện dự kiến từ 12 giờ ngày 25-3 (thứ bảy) đến 4 giờ ngày 26-3 (chủ nhật)."""

    evidence = """SAWACO thông báo tạm ngưng cung cấp nước để thực hiện công tác nêu trên. Thời gian thực hiện dự kiến từ 22 giờ ngày 25-3 (thứ bảy) đến 4 giờ ngày 26-3 (chủ nhật)."""

    # py_vncorenlp.download_model(save_dir=VNCORENLP_PATH)
    model = py_vncorenlp.VnCoreNLP(save_dir=VNCORENLP_PATH)

    # Xử lý context và claim
    print("Đang xử lý context...")
    context_sentences = model.annotate_text(context)
    
    print("Đang xử lý claim...")
    claim_sentences = model.annotate_text(claim)
    
    # Tạo đồ thị text
    print("Đang xây dựng đồ thị text...")
    text_graph = TextGraph()
    text_graph.build_from_vncorenlp_output(context_sentences, claim, claim_sentences)
    
    # Demo tính năng mới: Trích xuất entities từ OpenAI
    print("\n=== TRÍCH XUẤT ENTITIES TỪ OPENAI ===")
    entity_nodes = text_graph.extract_and_add_entities(context, context_sentences)
    
    # Demo tính năng Semantic Similarity (có thể comment out trên server không GPU)
    print("\n=== DEMO SEMANTIC SIMILARITY ===")
    try:
        # Giảm parameters cho demo trên server không GPU
        text_graph.similarity_threshold = 0.7  # Giảm threshold
        text_graph.top_k_similar = 3  # Giảm top-k
        text_graph.reduced_dim = 64   # Giảm PCA dimensions
        
        print(f"Parameters: threshold={text_graph.similarity_threshold}, top_k={text_graph.top_k_similar}, pca_dim={text_graph.reduced_dim}")
        
        # Build semantic similarity edges
        semantic_edges_added = text_graph.build_semantic_similarity_edges(use_pca=True, use_faiss=True)
        print(f"Đã thêm {semantic_edges_added} semantic similarity edges.")
        
    except Exception as e:
        print(f"Lỗi khi build semantic similarity: {e}")
        print("Có thể do thiếu GPU hoặc dependencies. Tiếp tục với demo khác...")
    
    # In thống kê chi tiết
    stats = text_graph.get_detailed_statistics()
    print("\n=== THỐNG KÊ CHI TIẾT ĐỒ THỊ TEXT ===")
    print(f"Tổng số nodes: {stats['total_nodes']}")
    print(f"Tổng số edges: {stats['total_edges']}")
    print(f"  - Structural edges: {stats['structural_edges']}")
    print(f"  - Dependency edges: {stats['dependency_edges']}")
    print(f"  - Entity edges: {stats['entity_structural_edges']}")
    print(f"  - Semantic edges: {stats.get('semantic_edges', 0)}")
    print(f"Word nodes: {stats['word_nodes']}")
    print(f"Sentence nodes: {stats['sentence_nodes']}")
    print(f"Claim nodes: {stats['claim_nodes']}")
    print(f"Entity nodes: {stats['entity_nodes']}")
    print(f"Số từ duy nhất: {stats['unique_words']}")
    print(f"Số entities duy nhất: {stats['unique_entities']}")
    print(f"Số từ chung giữa context và claim: {stats['shared_words_count']}")
    print(f"Trung bình từ mỗi câu: {stats['average_words_per_sentence']:.1f}")
    
    # In thông tin về entities được trích xuất
    if stats['entities']:
        print(f"\n=== ENTITIES ĐƯỢC TRÍCH XUẤT ===")
        for entity in stats['entities']:
            print(f"'{entity['name']}' (Type: {entity['type']}) - Kết nối với {entity['connected_sentences']} câu")
    
    # In thông tin về semantic similarity
    if stats.get('semantic_edges', 0) > 0:
        semantic_stats = stats['semantic_statistics']
        print(f"\n=== SEMANTIC SIMILARITY STATISTICS ===")
        print(f"Tổng semantic edges: {semantic_stats['total_semantic_edges']}")
        print(f"Similarity trung bình: {semantic_stats['average_similarity']:.3f}")
        print(f"Similarity cao nhất: {semantic_stats['max_similarity']:.3f}")
        print(f"Similarity thấp nhất: {semantic_stats['min_similarity']:.3f}")
        
        print("Phân bố similarity:")
        for range_key, count in semantic_stats['similarity_distribution'].items():
            if count > 0:
                print(f"  {range_key}: {count} edges")
    
    # In thống kê dependency
    dep_stats = stats['dependency_statistics']
    print(f"\n=== THỐNG KÊ DEPENDENCY PARSING ===")
    print(f"Tổng số dependency relationships: {dep_stats['total_dependency_edges']}")
    print(f"Số loại dependency khác nhau: {len(dep_stats['dependency_types'])}")
    
    # In các dependency phổ biến nhất
    print("\n=== DEPENDENCY RELATIONSHIPS PHỔ BIẾN NHẤT ===")
    for dep_type, count in dep_stats['most_common_dependencies'][:8]:
        print(f"'{dep_type}': {count} lần")
    
    # In các từ xuất hiện nhiều nhất
    print("\n=== TỪ XUẤT HIỆN NHIỀU NHẤT ===")
    for word, freq in stats['most_frequent_words']:
        print(f"'{word}': {freq} lần")
    
    # In các từ chung giữa context và claim
    print("\n=== TỪ CHUNG GIỮA CONTEXT VÀ CLAIM ===")
    if stats['shared_words']:
        for shared_word in stats['shared_words'][:10]:  # Chỉ in 10 từ đầu tiên
            print(f"'{shared_word['word']}' (POS: {shared_word['pos']})")
    else:
        print("Không có từ chung nào được tìm thấy.")
    
    # Demo phân tích dependency của một từ cụ thể
    print("\n=== PHÂN TÍCH DEPENDENCY CỦA TỪ 'SAWACO' ===")
    sawaco_deps = text_graph.get_word_dependencies("SAWACO")
    if sawaco_deps['heads']:
        print("Heads (từ mà SAWACO phụ thuộc vào):")
        for head in sawaco_deps['heads']:
            print(f"  -> {head['word']} (relation: {head['relation']})")
    if sawaco_deps['dependents']:
        print("Dependents (từ phụ thuộc vào SAWACO):")
        for dep in sawaco_deps['dependents']:
            print(f"  <- {dep['word']} (relation: {dep['relation']})")
    
    # In một số ví dụ về nodes
    print("\n=== MỘT SỐ VÍ DỤ VỀ NODES ===")
    node_count = 0
    for node in text_graph.graph.nodes():
        if node_count >= 3:  # Chỉ in 3 node đầu tiên
            break
        node_data = text_graph.graph.nodes[node]
        print(f"Node ID: {node}")
        print(f"  Type: {node_data['type']}")
        print(f"  Text: {node_data['text'][:50]}...")
        if 'pos' in node_data and node_data['pos']:
            print(f"  POS: {node_data['pos']}")
        print()
        node_count += 1
    
    # Vẽ đồ thị (uncomment dòng dưới nếu muốn hiển thị)
    text_graph.visualize()
    
    # Vẽ chỉ dependency graph (uncomment dòng dưới nếu muốn hiển thị)
    # text_graph.visualize_dependencies_only()
    
    # Lưu đồ thị vào file (demo tính năng mới)
    text_graph.save_graph("text_graph.gexf")
    
    # Xuất ra JSON (demo tính năng mới)
    # print("\n=== XUẤT RA JSON ===")
    # json_data = text_graph.export_to_json()
    # print(json_data[:500] + "...")  # Chỉ in 500 ký tự đầu
    
    print("Đã hoàn thành xây dựng đồ thị text với dependency parsing!")
    