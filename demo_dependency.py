"""
Demo script để trực quan hóa dependency parsing trong TextGraph
"""

import py_vncorenlp
from os import path, getcwd
from mint import TextGraph

BASE_DIR = getcwd()
VNCORENLP_PATH = path.join(BASE_DIR, "vncorenlp")

# Khởi tạo model một lần duy nhất
MODEL = None

def get_model():
    """Lấy model py_vncorenlp (singleton pattern)"""
    global MODEL
    if MODEL is None:
        MODEL = py_vncorenlp.VnCoreNLP(save_dir=VNCORENLP_PATH)
    return MODEL

def demo_simple_sentence():
    """Demo với câu đơn giản"""
    print("🔍 DEMO VỚI CÂU ĐÚN GIẢN")
    print("=" * 50)
    
    simple_context = "SAWACO thông báo tạm ngưng cung cấp nước."
    simple_claim = "SAWACO ngưng cung cấp nước."
    
    model = get_model()
    
    context_sentences = model.annotate_text(simple_context)
    claim_sentences = model.annotate_text(simple_claim)
    
    text_graph = TextGraph()
    text_graph.build_from_vncorenlp_output(context_sentences, simple_claim, claim_sentences)
    
    stats = text_graph.get_detailed_statistics()
    
    print(f"📊 Thống kê cơ bản:")
    print(f"  - Nodes: {stats['total_nodes']}")
    print(f"  - Edges: {stats['total_edges']} (Structural: {stats['structural_edges']}, Dependency: {stats['dependency_edges']})")
    print(f"  - Shared words: {stats['shared_words_count']}")
    
    print(f"\n🔗 Dependency relationships:")
    dep_stats = stats['dependency_statistics']
    for dep_type, count in dep_stats['most_common_dependencies']:
        print(f"  {dep_type}: {count}")
    
    print(f"\n📝 Shared words giữa context và claim:")
    for word_info in stats['shared_words']:
        print(f"  '{word_info['word']}' ({word_info['pos']})")
    
    # Phân tích dependency của từ SAWACO
    print(f"\n🎯 Dependency analysis của 'SAWACO':")
    deps = text_graph.get_word_dependencies("SAWACO")
    if deps['heads']:
        print("  Heads:")
        for head in deps['heads']:
            print(f"    -> {head['word']} ({head['relation']})")
    if deps['dependents']:
        print("  Dependents:")
        for dep in deps['dependents']:
            print(f"    <- {dep['word']} ({dep['relation']})")
    
    # Uncomment để xem visualization
    # text_graph.visualize(show_dependencies=True)
    # text_graph.visualize_dependencies_only()
    
    return text_graph

def analyze_complex_dependencies():
    """Phân tích dependency relationships phức tạp"""
    print("\n🧠 PHÂN TÍCH DEPENDENCY PHỨC TẠP")
    print("=" * 50)
    
    complex_text = "Công ty SAWACO đã thông báo việc tạm ngưng cung cấp nước sạch để thực hiện bảo trì hệ thống."
    
    model = get_model()
    sentences = model.annotate_text(complex_text)
    
    text_graph = TextGraph()
    # Tạo đồ thị chỉ với context
    text_graph.build_from_vncorenlp_output(sentences, "", {})
    
    dep_stats = text_graph.get_dependency_statistics()
    
    print(f"📈 Dependency statistics:")
    print(f"  Total dependency edges: {dep_stats['total_dependency_edges']}")
    print(f"  Unique dependency types: {len(dep_stats['dependency_types'])}")
    
    print(f"\n🏷️ Dependency types:")
    for dep_type, count in sorted(dep_stats['dependency_types'].items()):
        print(f"  {dep_type}: {count}")
    
    # Phân tích một số từ quan trọng
    important_words = ["SAWACO", "thông_báo", "cung_cấp", "nước"]
    
    print(f"\n🔍 Dependency analysis cho từ khóa:")
    for word in important_words:
        if word in text_graph.word_nodes:
            deps = text_graph.get_word_dependencies(word)
            print(f"\n  '{word}':")
            if deps['heads']:
                heads_str = [f"{h['word']}({h['relation']})" for h in deps['heads']]
                print(f"    Heads: {heads_str}")
            if deps['dependents']:
                deps_str = [f"{d['word']}({d['relation']})" for d in deps['dependents']]
                print(f"    Dependents: {deps_str}")
    
    return text_graph

def compare_structures():
    """So sánh cấu trúc dependency giữa context và claim"""
    print("\n⚖️ SO SÁNH CẤU TRÚC DEPENDENCY")
    print("=" * 50)
    
    context = "SAWACO thông báo tạm ngưng cung cấp nước từ 22 giờ ngày 25-3."
    claim = "SAWACO ngưng cung cấp nước từ 12 giờ ngày 25-3."
    
    model = get_model()
    
    context_sentences = model.annotate_text(context)
    claim_sentences = model.annotate_text(claim)
    
    text_graph = TextGraph()
    text_graph.build_from_vncorenlp_output(context_sentences, claim, claim_sentences)
    
    stats = text_graph.get_detailed_statistics()
    
    print(f"📊 Graph overview:")
    print(f"  Shared words: {stats['shared_words_count']}/{stats['unique_words']}")
    print(f"  Dependency edges: {stats['dependency_edges']}")
    
    print(f"\n🎯 Từ chung và dependency của chúng:")
    for word_info in stats['shared_words'][:5]:  # Top 5 shared words
        word = word_info['word']
        if len(word) > 1:  # Skip punctuation
            deps = text_graph.get_word_dependencies(word)
            print(f"  '{word}' ({word_info['pos']}):")
            if deps['heads']:
                print(f"    -> {[h['word'] for h in deps['heads']]}")
            if deps['dependents']:
                print(f"    <- {[d['word'] for d in deps['dependents']]}")

if __name__ == "__main__":
    print("🚀 MINT TextGraph - Dependency Parsing Demo")
    print("=" * 60)
    
    # Demo 1: Câu đơn giản
    graph1 = demo_simple_sentence()
    
    # Demo 2: Dependency phức tạp  
    graph2 = analyze_complex_dependencies()
    
    # Demo 3: So sánh cấu trúc
    compare_structures()
    
    print(f"\n✅ Demo hoàn thành!")
    print(f"💡 Để xem visualization, uncomment các dòng:")
    print(f"   - text_graph.visualize()")
    print(f"   - text_graph.visualize_dependencies_only()") 