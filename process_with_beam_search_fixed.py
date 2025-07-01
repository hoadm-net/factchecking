import json
import py_vncorenlp
from mint.text_graph import TextGraph
import os
from datetime import datetime

# IMPROVED ENTITY MATCHING FUNCTIONS
def segment_entity_with_vncorenlp(entity, model):
    """
    Segment entity sử dụng VnCoreNLP để match với segmented text
    """
    try:
        result = model.annotate_text(entity)
        segmented_words = []
        if result and len(result) > 0:
            first_sentence = list(result.values())[0]
            for token in first_sentence:
                segmented_words.append(token["wordForm"])
        
        segmented_entity = "_".join(segmented_words)
        return segmented_entity
        
    except Exception as e:
        # Fallback: simple space to underscore replacement
        return entity.replace(" ", "_")

def improved_entity_matching(entity, sentence_text, model=None):
    """
    Improved entity matching với support cho segmented text
    """
    entity_lower = entity.lower()
    sentence_lower = sentence_text.lower()
    
    # Method 1: Direct matching
    if entity_lower in sentence_lower:
        return True
    
    # Method 2: Simple space->underscore replacement
    entity_simple_seg = entity.replace(" ", "_").lower()
    if entity_simple_seg in sentence_lower:
        return True
    
    # Method 3: VnCoreNLP segmentation
    if model:
        try:
            entity_vncorenlp_seg = segment_entity_with_vncorenlp(entity, model).lower()
            if entity_vncorenlp_seg in sentence_lower:
                return True
        except:
            pass
    
    # Method 4: Fuzzy matching cho partial matches
    entity_words = entity.split()
    if len(entity_words) > 1:
        all_words_found = True
        for word in entity_words:
            word_variants = [
                word.lower(),
                word.replace(" ", "_").lower()
            ]
            
            word_found = any(variant in sentence_lower for variant in word_variants)
            if not word_found:
                all_words_found = False
                break
        
        if all_words_found:
            return True
    
    return False

def improved_add_entities_to_graph(text_graph, entities, context_sentences, model):
    """
    Improved version của add_entities_to_graph với better matching
    """
    entity_nodes_added = []
    
    for entity in entities:
        # Thêm entity node
        entity_node = text_graph.add_entity_node(entity)
        entity_nodes_added.append(entity_node)
        
        connections_made = 0
        # Tìm các sentences có chứa entity này
        for sent_idx, sentence_node in text_graph.sentence_nodes.items():
            sentence_text = text_graph.graph.nodes[sentence_node]['text']
            
            # SỬ DỤNG IMPROVED MATCHING
            if improved_entity_matching(entity, sentence_text, model):
                text_graph.connect_entity_to_sentence(entity_node, sentence_node)
                connections_made += 1
                print(f"✅ Improved: Kết nối entity '{entity}' với sentence {sent_idx}")
        
        if connections_made == 0:
            print(f"⚠️ Entity '{entity}' không match với sentence nào")
    
    print(f"Đã thêm {len(entity_nodes_added)} entity nodes vào graph với improved matching.")
    return entity_nodes_added

def process_sample_with_beam_search(sample_data, model, output_dir="beam_output"):
    """
    Xử lý một sample: xây dựng TextGraph và chạy Beam Search với improved entity matching
    """
    context = sample_data["context"]
    claim = sample_data["claim"]
    evidence = sample_data["evidence"]
    label = sample_data["label"]
    
    print(f"Processing claim: {claim[:100]}...")
    
    try:
        # Xử lý context và claim với VnCoreNLP
        context_sentences = model.annotate_text(context)
        claim_sentences = model.annotate_text(claim)
        
        # Tạo TextGraph
        text_graph = TextGraph()
        
        # Build basic graph từ VnCoreNLP output
        text_graph.build_from_vncorenlp_output(
            context_sentences, claim, claim_sentences
        )
        
        # 🔧 IMPROVED: Extract và add entities với better matching
        print("🤖 Extracting entities with improved matching...")
        try:
            entities = text_graph.extract_entities_with_openai(context)
            if entities:
                entity_nodes = improved_add_entities_to_graph(
                    text_graph, entities, context_sentences, model
                )
                print(f"✅ Added {len(entity_nodes)} entity nodes with improved matching")
        except Exception as e:
            print(f"⚠️ Entity extraction failed: {e}")
        
        # Chạy Beam Search để tìm paths từ claim đến sentences
        paths = text_graph.beam_search_paths(
            beam_width=10,
            max_depth=6,
            max_paths=20
        )
        
        # Trích xuất sentences từ paths
        beam_sentences = extract_sentences_from_paths(paths, text_graph)
        
        # Xử lý format sentences (replace _ thành space)
        processed_sentences = []
        for sentence in beam_sentences:
            processed_sentence = sentence.replace("_", " ")
            processed_sentences.append(processed_sentence)
        
        # Tạo result
        result = {
            "context": context,
            "claim": claim,
            "evidence": evidence,
            "beam_evidence": processed_sentences,
            "label": label
        }
        
        return result, True
        
    except Exception as e:
        print(f"❌ Error processing sample: {e}")
        return None, False

def extract_sentences_from_paths(paths, text_graph):
    """
    Trích xuất danh sách sentences từ beam search paths
    """
    sentence_frequency = {}
    
    # Kiểm tra paths structure
    if not paths:
        print("⚠️  No paths found")
        return []
    
    # Đếm tần suất xuất hiện của mỗi sentence
    for path_obj in paths:
        visited_sentences = set()
        
        # Kiểm tra cấu trúc path object
        if hasattr(path_obj, 'nodes'):
            path_nodes = path_obj.nodes
        elif hasattr(path_obj, 'path'):
            path_nodes = path_obj.path
        elif isinstance(path_obj, dict) and 'nodes' in path_obj:
            path_nodes = path_obj['nodes']
        elif isinstance(path_obj, dict) and 'path' in path_obj:
            path_nodes = path_obj['path']
        elif hasattr(path_obj, '__dict__'):
            path_dict = path_obj.__dict__
            if 'nodes' in path_dict:
                path_nodes = path_dict['nodes']
            elif 'path' in path_dict:
                path_nodes = path_dict['path']
            else:
                continue
        else:
            continue
            
        for node_id in path_nodes:
            if node_id in text_graph.graph.nodes:
                node_data = text_graph.graph.nodes[node_id]
                if node_data.get('type') == 'sentence':
                    sentence_text = node_data.get('text', '')
                    if sentence_text and sentence_text not in visited_sentences:
                        sentence_frequency[sentence_text] = sentence_frequency.get(sentence_text, 0) + 1
                        visited_sentences.add(sentence_text)
    
    # Sắp xếp theo tần suất giảm dần
    sorted_sentences = sorted(sentence_frequency.items(), key=lambda x: x[1], reverse=True)
    
    print(f"📊 Found {len(sentence_frequency)} unique sentences")
    
    # Trả về top sentences (giới hạn 10 sentences)
    return [sentence for sentence, freq in sorted_sentences[:10]]

def main():
    """
    Main function để xử lý toàn bộ dataset với improved entity matching
    """
    print("🚀 Starting Beam Search processing with IMPROVED ENTITY MATCHING...")
    
    # Lưu working directory hiện tại
    original_cwd = os.getcwd()
    print(f"📂 Original working directory: {original_cwd}")
    
    # Khởi tạo VnCoreNLP model với đường dẫn tuyệt đối
    print("📖 Loading VnCoreNLP model...")
    vncorenlp_path = os.path.abspath("vncorenlp")
    model = py_vncorenlp.VnCoreNLP(save_dir=vncorenlp_path)
    
    # Khôi phục working directory
    os.chdir(original_cwd)
    print(f"📂 Restored working directory: {os.getcwd()}")
    
    # Đọc input file
    input_file = "raw_test.json"
    input_file_path = os.path.abspath(input_file)
    if not os.path.exists(input_file_path):
        print(f"❌ File {input_file_path} not found!")
        return
    
    print(f"📁 Input file: {input_file_path}")
    
    with open(input_file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"📊 Total samples: {len(data)}")
    
    # Tạo output directory
    os.makedirs("beam_output", exist_ok=True)
    
    # Xử lý samples (test với 5 samples đầu tiên)
    results = []
    success_count = 0
    total_beam_sentences = 0
    
    for i, sample in enumerate(data[:5]):  # Test với 5 samples
        print(f"\n📝 Processing sample {i+1}/5...")
        
        result, success = process_sample_with_beam_search(sample, model)
        
        if success and result:
            results.append(result)
            success_count += 1
            
            beam_count = len(result.get("beam_evidence", []))
            total_beam_sentences += beam_count
            
            print(f"✅ Sample {i+1}: Found {beam_count} sentences")
            # Show first 3 sentences
            for j, sentence in enumerate(result["beam_evidence"][:3], 1):
                print(f"   {j}. {sentence[:100]}...")
        else:
            print(f"❌ Sample {i+1}: Failed to process")
    
    # Lưu kết quả
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"beam_output/processed_with_improved_entities_{timestamp}.json"
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    # In thống kê
    print(f"\n🎉 Processing completed!")
    print(f"📁 Results saved to: {output_file}")
    print(f"📊 Success rate: {success_count}/{len(data[:5])} ({success_count/5*100:.1f}%)")
    print(f"📈 Statistics:")
    print(f"   - Total beam sentences found: {total_beam_sentences}")
    print(f"   - Average sentences per sample: {total_beam_sentences/max(success_count,1):.1f}")
    
    if success_count > 0:
        print(f"\n📋 Sample result:")
        sample_result = results[0]
        print(f"   Claim: {sample_result['claim'][:100]}...")
        print(f"   Beam evidence found: {len(sample_result['beam_evidence'])} sentences")
        for i, sentence in enumerate(sample_result["beam_evidence"][:2], 1):
            print(f"      {i}. {sentence[:100]}...")

if __name__ == "__main__":
    main() 