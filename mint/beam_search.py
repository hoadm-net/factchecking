#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
MINT TextGraph - Beam Search Path Finding
Tìm đường đi từ claim đến sentence nodes bằng Beam Search
"""

import json
import os
from collections import defaultdict
from typing import List, Dict, Tuple, Set
import heapq
from datetime import datetime


class Path:
    """Đại diện cho một đường đi trong đồ thị"""
    
    def __init__(self, nodes: List[str], edges: List[Tuple[str, str, str]] = None, score: float = 0.0):
        self.nodes = nodes  # Danh sách node IDs
        self.edges = edges or []  # Danh sách (from_node, to_node, relation)
        self.score = score  # Điểm đánh giá path
        self.claim_words = set()  # Từ trong claim để so sánh
        self.path_words = set()   # Từ trong path
        self.entities_visited = set()  # Entities đã đi qua
        
    def __lt__(self, other):
        """So sánh paths dựa trên score (để dùng trong heap)"""
        return self.score > other.score  # Reverse order - score cao hơn có priority cao hơn
        
    def add_node(self, node_id: str, edge_info: Tuple[str, str, str] = None):
        """Thêm node vào path"""
        self.nodes.append(node_id)
        if edge_info:
            self.edges.append(edge_info)
            
    def copy(self):
        """Tạo bản copy của path"""
        new_path = Path(self.nodes.copy(), self.edges.copy(), self.score)
        new_path.claim_words = self.claim_words.copy()
        new_path.path_words = self.path_words.copy()
        new_path.entities_visited = self.entities_visited.copy()
        return new_path
        
    def get_current_node(self):
        """Lấy node hiện tại (cuối path)"""
        return self.nodes[-1] if self.nodes else None
        
    def contains_node(self, node_id: str):
        """Kiểm tra path có chứa node này không"""
        return node_id in self.nodes
        
    def to_dict(self):
        """Convert path thành dictionary để export"""
        return {
            'nodes': self.nodes,
            'edges': self.edges,
            'score': self.score,
            'length': len(self.nodes),
            'claim_words_matched': len(self.claim_words.intersection(self.path_words)),
            'total_claim_words': len(self.claim_words),
            'entities_visited': list(self.entities_visited),
            'path_summary': self._get_path_summary()
        }
        
    def _get_path_summary(self):
        """Tạo summary ngắn gọn của path"""
        node_types = []
        for node in self.nodes:
            if node.startswith('claim'):
                node_types.append('CLAIM')
            elif node.startswith('word'):
                node_types.append('WORD')
            elif node.startswith('sentence'):
                node_types.append('SENTENCE')
            elif node.startswith('entity'):
                node_types.append('ENTITY')
            else:
                node_types.append('UNKNOWN')
        return ' -> '.join(node_types)


class BeamSearchPathFinder:
    """Beam Search để tìm đường đi từ claim đến sentence nodes"""
    
    def __init__(self, text_graph, beam_width: int = 10, max_depth: int = 6):
        self.graph = text_graph
        self.beam_width = beam_width
        self.max_depth = max_depth
        self.claim_words = set()  # Words trong claim
        
        # Scoring weights
        self.word_match_weight = 3.0    # Trọng số cho từ khớp với claim
        self.entity_bonus = 2.0         # Bonus cho path đi qua entity
        self.length_penalty = 0.1       # Penalty cho path dài
        self.sentence_bonus = 5.0       # Bonus lớn khi đến sentence
        
    def extract_claim_words(self):
        """Trích xuất tất cả từ trong claim để so sánh"""
        claim_words = set()
        
        if self.graph.claim_node:
            # Lấy tất cả word nodes connected đến claim
            for neighbor in self.graph.graph.neighbors(self.graph.claim_node):
                node_data = self.graph.graph.nodes[neighbor]
                if node_data.get('type') == 'word':
                    claim_words.add(node_data.get('text', '').lower())
                    
        self.claim_words = claim_words
        return claim_words
        
    def score_path(self, path: Path, graph_data: Dict) -> float:
        """
        Đánh giá điểm cho path dựa trên các tiêu chí:
        1. Số từ khớp với claim
        2. Path đi qua entity quan trọng  
        3. Path đến sentence node
        4. Độ dài path (penalty)
        """
        score = 0.0
        
        # 1. Word matching score
        path_words = set()
        entities_in_path = set()
        reached_sentence = False
        
        for node_id in path.nodes:
            node_data = graph_data.get(node_id, {})
            node_type = node_data.get('type', '')
            
            if node_type == 'word':
                word_text = node_data.get('text', '').lower()
                path_words.add(word_text)
                
            elif node_type == 'entity':
                entity_name = node_data.get('text', '').lower()
                entities_in_path.add(entity_name)
                
            elif node_type == 'sentence':
                reached_sentence = True
                
        # Score based on word overlap với claim
        word_matches = len(self.claim_words.intersection(path_words))
        if len(self.claim_words) > 0:
            word_match_ratio = word_matches / len(self.claim_words)
            score += word_match_ratio * self.word_match_weight
            
        # Bonus cho entity nodes
        score += len(entities_in_path) * self.entity_bonus
        
        # Bonus lớn nếu đạt sentence
        if reached_sentence:
            score += self.sentence_bonus
            
        # Length penalty (path càng dài càng bị trừ điểm)
        score -= len(path.nodes) * self.length_penalty
        
        # Update path attributes
        path.claim_words = self.claim_words
        path.path_words = path_words
        path.entities_visited = entities_in_path
        
        return score
        
    def beam_search(self, start_node: str = None) -> List[Path]:
        """
        Thực hiện Beam Search từ claim node đến sentence nodes
        
        Returns:
            List[Path]: Danh sách các paths tốt nhất tìm được
        """
        if start_node is None:
            start_node = self.graph.claim_node
            
        if not start_node:
            print("⚠️ Không tìm thấy claim node để bắt đầu beam search")
            return []
            
        # Extract claim words để scoring
        self.extract_claim_words()
        
        # Prepare graph data for faster lookup
        graph_data = dict(self.graph.graph.nodes(data=True))
        
        # Initialize beam với path từ claim node
        beam = [Path([start_node])]
        completed_paths = []  # Paths đã đến sentence nodes
        
        print(f"🎯 Starting Beam Search from {start_node}")
        print(f"📊 Beam width: {self.beam_width}, Max depth: {self.max_depth}")
        print(f"💭 Claim words: {self.claim_words}")
        
        for depth in range(self.max_depth):
            if not beam:
                break
                
            print(f"\n🔍 Depth {depth + 1}/{self.max_depth} - Current beam size: {len(beam)}")
            
            new_candidates = []
            
            # Expand mỗi path trong beam hiện tại
            for path in beam:
                current_node = path.get_current_node()
                
                # Lấy tất cả neighbors của current node
                neighbors = list(self.graph.graph.neighbors(current_node))
                
                for neighbor in neighbors:
                    # Tránh cycle - không quay lại node đã visit
                    if path.contains_node(neighbor):
                        continue
                        
                    # Tạo path mới
                    new_path = path.copy()
                    
                    # Lấy edge info
                    edge_data = self.graph.graph.get_edge_data(current_node, neighbor)
                    relation = edge_data.get('relation', 'unknown') if edge_data else 'unknown'
                    edge_info = (current_node, neighbor, relation)
                    
                    new_path.add_node(neighbor, edge_info)
                    
                    # Score path mới
                    new_path.score = self.score_path(new_path, graph_data)
                    
                    # Kiểm tra nếu đạt sentence node
                    neighbor_data = graph_data.get(neighbor, {})
                    if neighbor_data.get('type') == 'sentence':
                        completed_paths.append(new_path)
                        print(f"  ✅ Found path to sentence: {neighbor} (score: {new_path.score:.3f})")
                    else:
                        new_candidates.append(new_path)
                        
            # Chọn top K candidates cho beam tiếp theo
            if new_candidates:
                # Sort by score descending và chọn top beam_width
                new_candidates.sort(key=lambda p: p.score, reverse=True)
                beam = new_candidates[:self.beam_width]
                
                # Debug info
                print(f"  📈 Top scores in beam: {[f'{p.score:.3f}' for p in beam[:5]]}")
            else:
                beam = []
                
        # Combine completed paths và sort theo score
        all_paths = completed_paths
        all_paths.sort(key=lambda p: p.score, reverse=True)
        
        print(f"\n🎉 Beam Search completed!")
        print(f"  Found {len(completed_paths)} paths to sentences")
        print(f"  Top path score: {all_paths[0].score:.3f}" if all_paths else "  No paths found")
        
        return all_paths
        
    def find_best_paths(self, max_paths: int = 20) -> List[Path]:
        """
        Tìm các path tốt nhất từ claim đến sentences
        
        Args:
            max_paths: Số lượng paths tối đa để trả về
            
        Returns:
            List[Path]: Danh sách paths được sắp xếp theo score
        """
        all_paths = self.beam_search()
        return all_paths[:max_paths]
        
    def export_paths_to_file(self, paths: List[Path], output_file: str = None) -> str:
        """
        Export paths ra file JSON để khảo sát
        
        Args:
            paths: Danh sách paths cần export
            output_file: Đường dẫn file output (nếu None sẽ tự generate)
            
        Returns:
            str: Đường dẫn file đã lưu
        """
        if output_file is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            # Use absolute path to ensure correct directory
            current_dir = os.getcwd()
            if current_dir.endswith('vncorenlp'):
                # If we're in vncorenlp directory, go back to parent
                current_dir = os.path.dirname(current_dir)
            output_file = os.path.join(current_dir, "output", f"beam_search_paths_{timestamp}.json")
            
        # Tạo thư mục nếu chưa có
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        # Prepare data for export
        export_data = {
            'search_config': {
                'beam_width': self.beam_width,
                'max_depth': self.max_depth,
                'word_match_weight': self.word_match_weight,
                'entity_bonus': self.entity_bonus,
                'length_penalty': self.length_penalty,
                'sentence_bonus': self.sentence_bonus
            },
            'claim_words': list(self.claim_words),
            'total_paths_found': len(paths),
            'paths': []
        }
        
        # Prepare graph data for node details
        graph_data = dict(self.graph.graph.nodes(data=True))
        
        for i, path in enumerate(paths):
            path_data = path.to_dict()
            
            # Thêm thông tin chi tiết về nodes
            path_data['node_details'] = []
            for node_id in path.nodes:
                node_info = graph_data.get(node_id, {})
                path_data['node_details'].append({
                    'id': node_id,
                    'type': node_info.get('type', 'unknown'),
                    'text': node_info.get('text', ''),
                    'pos': node_info.get('pos', ''),
                    'lemma': node_info.get('lemma', '')
                })
                
            export_data['paths'].append(path_data)
            
        # Write to file
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, ensure_ascii=False, indent=2)
            
        print(f"💾 Exported {len(paths)} paths to: {output_file}")
        return output_file
        
    def export_paths_summary(self, paths: List[Path], output_file: str = None) -> str:
        """
        Export summary dễ đọc của paths
        
        Args:
            paths: Danh sách paths
            output_file: File output (nếu None sẽ tự generate)
            
        Returns:
            str: Đường dẫn file đã lưu
        """
        if output_file is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            # Use absolute path to ensure correct directory
            current_dir = os.getcwd()
            if current_dir.endswith('vncorenlp'):
                # If we're in vncorenlp directory, go back to parent
                current_dir = os.path.dirname(current_dir)
            output_file = os.path.join(current_dir, "output", f"beam_search_summary_{timestamp}.txt")
            
        # Tạo thư mục nếu chưa có
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        # Prepare graph data
        graph_data = dict(self.graph.graph.nodes(data=True))
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("🎯 BEAM SEARCH PATH ANALYSIS\n")
            f.write("="*60 + "\n\n")
            
            f.write(f"Search Configuration:\n")
            f.write(f"  Beam Width: {self.beam_width}\n")
            f.write(f"  Max Depth: {self.max_depth}\n")
            f.write(f"  Claim Words: {', '.join(self.claim_words)}\n")
            f.write(f"  Total Paths Found: {len(paths)}\n\n")
            
            for i, path in enumerate(paths[:10]):  # Top 10 paths
                f.write(f"PATH #{i+1} (Score: {path.score:.3f})\n")
                f.write("-" * 40 + "\n")
                
                f.write(f"Length: {len(path.nodes)} nodes\n")
                f.write(f"Word Matches: {len(path.claim_words.intersection(path.path_words))}/{len(path.claim_words)}\n")
                f.write(f"Entities Visited: {', '.join(path.entities_visited) if path.entities_visited else 'None'}\n")
                f.write(f"Path Type: {path._get_path_summary()}\n\n")
                
                f.write("Detailed Path:\n")
                for j, node_id in enumerate(path.nodes):
                    node_info = graph_data.get(node_id, {})
                    node_type = node_info.get('type', 'unknown').upper()
                    node_text = node_info.get('text', '')[:50]  # Truncate long text
                    
                    prefix = "  START: " if j == 0 else f"  {j:2d}: "
                    f.write(f"{prefix}[{node_type}] {node_text}\n")
                    
                    if j < len(path.edges):
                        edge_info = path.edges[j]
                        f.write(f"       └─ ({edge_info[2]}) ─>\n")
                        
                f.write("\n" + "="*60 + "\n\n")
                
        print(f"📄 Exported paths summary to: {output_file}")
        return output_file 