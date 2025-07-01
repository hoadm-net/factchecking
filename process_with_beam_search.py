import json
import py_vncorenlp
from mint.text_graph import TextGraph
import os
from datetime import datetime

def process_sample_with_beam_search(sample_data, model, output_dir="beam_output"):
    """
    Xử lý một sample: xây dựng TextGraph và chạy Beam Search
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
        
        # Build graph từ VnCoreNLP output
        text_graph.build_from_vncorenlp_output(
            context_sentences, claim, claim_sentences
        )
        
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
        print(f"Error processing sample: {e}")
        # Trả về kết quả rỗng nếu có lỗi
        result = {
            "context": context,
            "claim": claim,
            "evidence": evidence,
            "beam_evidence": [],
            "label": label
        }
        return result, False

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
            # Nếu là object có attributes
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
    Main function để xử lý toàn bộ dataset
    """
    print("🚀 Starting Beam Search processing...")
    
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
    
    print(f"📊 Loaded {len(data)} samples from {input_file}")
    
    # Tạo thư mục output
    os.makedirs("beam_output", exist_ok=True)
    
    # Xử lý samples (test với 5 samples đầu)
    test_samples = data[:5]  # Chỉ test 5 samples đầu
    results = []
    success_count = 0
    
    for i, sample in enumerate(test_samples, 1):
        print(f"\n📝 Processing sample {i}/{len(test_samples)}...")
        
        result, success = process_sample_with_beam_search(sample, model)
        results.append(result)
        
        if success:
            success_count += 1
            print(f"✅ Sample {i}: Found {len(result['beam_evidence'])} sentences")
            # Hiển thị một số sentences tìm được
            for j, sentence in enumerate(result['beam_evidence'][:3], 1):
                print(f"   {j}. {sentence[:100]}...")
        else:
            print(f"❌ Sample {i}: Failed to process")
    
    # Lưu kết quả
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"beam_output/processed_with_beam_{timestamp}.json"
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"\n🎉 Processing completed!")
    print(f"📁 Results saved to: {output_file}")
    print(f"📊 Success rate: {success_count}/{len(test_samples)} ({success_count/len(test_samples)*100:.1f}%)")
    
    # Hiển thị thống kê tổng quan
    total_beam_sentences = sum(len(r['beam_evidence']) for r in results)
    avg_beam_sentences = total_beam_sentences / len(results) if results else 0
    
    print(f"📈 Statistics:")
    print(f"   - Total beam sentences found: {total_beam_sentences}")
    print(f"   - Average sentences per sample: {avg_beam_sentences:.1f}")
    
    # Hiển thị một sample kết quả
    if results and results[0]['beam_evidence']:
        print(f"\n📋 Sample result:")
        sample_result = results[0]
        print(f"   Claim: {sample_result['claim'][:100]}...")
        print(f"   Beam evidence found: {len(sample_result['beam_evidence'])} sentences")
        for i, sentence in enumerate(sample_result['beam_evidence'][:2], 1):
            print(f"      {i}. {sentence[:80]}...")

if __name__ == "__main__":
    main() 