import json
import py_vncorenlp
from mint.text_graph import TextGraph
import os
from datetime import datetime
import statistics

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
    total_connections = 0
    
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
                total_connections += 1
        
        if connections_made == 0:
            print(f"⚠️ Entity '{entity}' không match với sentence nào")
    
    print(f"📊 Entity stats: {len(entity_nodes_added)} entities, {total_connections} total connections")
    return entity_nodes_added, total_connections

def count_sentences_in_context(context, model):
    """
    Đếm số câu trong context bằng VnCoreNLP
    """
    try:
        context_sentences = model.annotate_text(context)
        return len(context_sentences)
    except Exception as e:
        print(f"⚠️ Error counting sentences: {e}")
        # Fallback: đếm câu bằng dấu câu
        return len([s for s in context.split('.') if s.strip()])

def process_sample_with_beam_search(sample_data, model, sample_index, output_dir="beam_output"):
    """
    Xử lý một sample: xây dựng TextGraph và chạy Beam Search với improved entity matching
    """
    context = sample_data["context"]
    claim = sample_data["claim"]
    evidence = sample_data["evidence"]
    label = sample_data["label"]
    
    print(f"📝 Processing sample {sample_index}: {claim[:60]}...")
    
    # Đếm tổng số câu trong context
    total_context_sentences = count_sentences_in_context(context, model)
    
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
        
        # Extract và add entities với better matching
        entity_stats = {"entities_count": 0, "connections_count": 0}
        try:
            entities = text_graph.extract_entities_with_openai(context)
            if entities:
                entity_nodes, total_connections = improved_add_entities_to_graph(
                    text_graph, entities, context_sentences, model
                )
                entity_stats["entities_count"] = len(entities)
                entity_stats["connections_count"] = total_connections
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
        
        # Tính tỷ lệ beam_evidence / context_sentences
        beam_evidence_count = len(processed_sentences)
        coverage_ratio = beam_evidence_count / total_context_sentences if total_context_sentences > 0 else 0
        
        # Tạo result với thống kê chi tiết
        result = {
            "context": context,
            "claim": claim,
            "evidence": evidence,
            "beam_evidence": processed_sentences,
            "label": label,
            "statistics": {
                "total_context_sentences": total_context_sentences,
                "beam_evidence_count": beam_evidence_count,
                "coverage_ratio": round(coverage_ratio, 4),
                "coverage_percentage": round(coverage_ratio * 100, 2),
                "entities_extracted": entity_stats["entities_count"],
                "entity_connections": entity_stats["connections_count"],
                "paths_found": len(paths) if paths else 0
            }
        }
        
        return result, True
        
    except Exception as e:
        print(f"❌ Error processing sample {sample_index}: {e}")
        return None, False

def extract_sentences_from_paths(paths, text_graph):
    """
    Trích xuất danh sách sentences từ beam search paths
    """
    sentence_frequency = {}
    
    # Kiểm tra paths structure
    if not paths:
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
    
    # Trả về top sentences (giới hạn 10 sentences)
    return [sentence for sentence, freq in sorted_sentences[:10]]

def calculate_dataset_statistics(results):
    """
    Tính toán thống kê tổng quan cho toàn bộ dataset
    """
    if not results:
        return {}
    
    # Collect metrics
    coverage_ratios = []
    beam_counts = []
    context_sentence_counts = []
    entity_counts = []
    connection_counts = []
    path_counts = []
    
    for result in results:
        stats = result.get("statistics", {})
        coverage_ratios.append(stats.get("coverage_ratio", 0))
        beam_counts.append(stats.get("beam_evidence_count", 0))
        context_sentence_counts.append(stats.get("total_context_sentences", 0))
        entity_counts.append(stats.get("entities_extracted", 0))
        connection_counts.append(stats.get("entity_connections", 0))
        path_counts.append(stats.get("paths_found", 0))
    
    # Calculate statistics
    dataset_stats = {
        "total_samples": len(results),
        "coverage_ratio": {
            "mean": round(statistics.mean(coverage_ratios), 4),
            "median": round(statistics.median(coverage_ratios), 4),
            "min": round(min(coverage_ratios), 4),
            "max": round(max(coverage_ratios), 4),
            "std": round(statistics.stdev(coverage_ratios) if len(coverage_ratios) > 1 else 0, 4)
        },
        "coverage_percentage": {
            "mean": round(statistics.mean(coverage_ratios) * 100, 2),
            "median": round(statistics.median(coverage_ratios) * 100, 2),
            "min": round(min(coverage_ratios) * 100, 2),
            "max": round(max(coverage_ratios) * 100, 2)
        },
        "beam_evidence_count": {
            "total": sum(beam_counts),
            "mean": round(statistics.mean(beam_counts), 2),
            "median": statistics.median(beam_counts),
            "min": min(beam_counts),
            "max": max(beam_counts)
        },
        "context_sentences": {
            "total": sum(context_sentence_counts),
            "mean": round(statistics.mean(context_sentence_counts), 2),
            "median": statistics.median(context_sentence_counts),
            "min": min(context_sentence_counts),
            "max": max(context_sentence_counts)
        },
        "entities": {
            "total_extracted": sum(entity_counts),
            "total_connections": sum(connection_counts),
            "mean_per_sample": round(statistics.mean(entity_counts), 2),
            "mean_connections_per_sample": round(statistics.mean(connection_counts), 2)
        },
        "paths": {
            "total": sum(path_counts),
            "mean_per_sample": round(statistics.mean(path_counts), 2),
            "median": statistics.median(path_counts)
        }
    }
    
    return dataset_stats

def main():
    """
    Main function để xử lý toàn bộ 300 samples với thống kê chi tiết
    """
    print("🚀 Starting FULL DATASET processing with detailed statistics...")
    print("📊 Target: Process all 300 samples with beam_evidence coverage analysis")
    
    # Lưu working directory hiện tại
    original_cwd = os.getcwd()
    
    # Khởi tạo VnCoreNLP model với đường dẫn tuyệt đối
    print("📖 Loading VnCoreNLP model...")
    vncorenlp_path = os.path.abspath("vncorenlp")
    model = py_vncorenlp.VnCoreNLP(save_dir=vncorenlp_path)
    
    # Khôi phục working directory
    os.chdir(original_cwd)
    
    # Đọc input file
    input_file = "raw_test.json"
    input_file_path = os.path.abspath(input_file)
    if not os.path.exists(input_file_path):
        print(f"❌ File {input_file_path} not found!")
        return
    
    with open(input_file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"📊 Dataset loaded: {len(data)} samples")
    
    # Tạo output directory
    os.makedirs("beam_output", exist_ok=True)
    
    # Xử lý toàn bộ samples
    results = []
    success_count = 0
    
    # Thống kê realtime
    total_beam_sentences = 0
    total_context_sentences = 0
    
    print(f"\n🔄 Processing {len(data)} samples...")
    
    for i, sample in enumerate(data):
        print(f"\n{'='*60}")
        
        result, success = process_sample_with_beam_search(sample, model, i+1)
        
        if success and result:
            results.append(result)
            success_count += 1
            
            # Update running statistics
            stats = result.get("statistics", {})
            beam_count = stats.get("beam_evidence_count", 0)
            context_count = stats.get("total_context_sentences", 0)
            coverage = stats.get("coverage_percentage", 0)
            
            total_beam_sentences += beam_count
            total_context_sentences += context_count
            
            print(f"✅ Sample {i+1}: {beam_count} beam evidence / {context_count} context sentences = {coverage}%")
            
        else:
            print(f"❌ Sample {i+1}: Failed to process")
        
        # Progress update every 50 samples
        if (i + 1) % 50 == 0:
            current_coverage = (total_beam_sentences / total_context_sentences * 100) if total_context_sentences > 0 else 0
            print(f"\n📈 Progress: {i+1}/{len(data)} ({(i+1)/len(data)*100:.1f}%)")
            print(f"📊 Running stats: {success_count} successful, {total_beam_sentences} total beam evidence")
            print(f"📊 Overall coverage so far: {current_coverage:.2f}%")
    
    # Tính toán thống kê tổng quan
    print(f"\n🧮 Calculating comprehensive statistics...")
    dataset_stats = calculate_dataset_statistics(results)
    
    # Lưu kết quả
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"beam_output/full_dataset_with_stats_{timestamp}.json"
    stats_file = f"beam_output/dataset_statistics_{timestamp}.json"
    
    # Lưu data + individual stats
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    # Lưu tổng thống kê riêng
    with open(stats_file, 'w', encoding='utf-8') as f:
        json.dump(dataset_stats, f, ensure_ascii=False, indent=2)
    
    # In báo cáo tổng kết
    print(f"\n{'='*80}")
    print(f"🎉 FULL DATASET PROCESSING COMPLETED!")
    print(f"{'='*80}")
    
    print(f"\n📁 Output files:")
    print(f"   - Data: {output_file}")
    print(f"   - Statistics: {stats_file}")
    
    print(f"\n📊 OVERALL RESULTS:")
    print(f"   ✅ Success rate: {success_count}/{len(data)} ({success_count/len(data)*100:.1f}%)")
    print(f"   📈 Total beam evidence found: {dataset_stats['beam_evidence_count']['total']}")
    print(f"   📚 Total context sentences: {dataset_stats['context_sentences']['total']}")
    
    overall_coverage = dataset_stats['coverage_percentage']['mean']
    print(f"\n🎯 KEY METRIC - BEAM EVIDENCE COVERAGE:")
    print(f"   📊 Mean coverage: {overall_coverage:.2f}%")
    print(f"   📊 Median coverage: {dataset_stats['coverage_percentage']['median']:.2f}%")
    print(f"   📊 Range: {dataset_stats['coverage_percentage']['min']:.2f}% - {dataset_stats['coverage_percentage']['max']:.2f}%")
    
    print(f"\n📈 DETAILED STATISTICS:")
    print(f"   🔍 Average beam evidence per sample: {dataset_stats['beam_evidence_count']['mean']:.1f}")
    print(f"   📄 Average context sentences per sample: {dataset_stats['context_sentences']['mean']:.1f}")
    print(f"   🏷️ Total entities extracted: {dataset_stats['entities']['total_extracted']}")
    print(f"   🔗 Total entity-sentence connections: {dataset_stats['entities']['total_connections']}")
    print(f"   🛤️ Average paths found per sample: {dataset_stats['paths']['mean_per_sample']:.1f}")
    
    if overall_coverage >= 30:
        print(f"\n🌟 EXCELLENT: Coverage rate ≥30% indicates strong beam search performance!")
    elif overall_coverage >= 20:
        print(f"\n👍 GOOD: Coverage rate ≥20% shows decent beam search results.")
    elif overall_coverage >= 10:
        print(f"\n⚠️ MODERATE: Coverage rate ≥10% suggests room for improvement.")
    else:
        print(f"\n🔧 NEEDS IMPROVEMENT: Coverage rate <10% indicates beam search optimization needed.")

if __name__ == "__main__":
    main() 