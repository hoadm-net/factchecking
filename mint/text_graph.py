import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os
import json
from openai import OpenAI
from dotenv import load_dotenv
import numpy as np
from transformers import AutoTokenizer, AutoModel
import torch
import faiss
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity


class TextGraph:
    """
    Lớp TextGraph để xây dựng và phân tích đồ thị văn bản từ context và claim
    
    Đồ thị bao gồm các loại node:
    - Word nodes: chứa từng từ trong context và claim
    - Sentence nodes: các câu trong context  
    - Claim node: giá trị claim
    """
    
    def __init__(self):
        self.graph = nx.Graph()
        self.word_nodes = {}
        self.sentence_nodes = {}
        self.claim_node = None
        self.entity_nodes = {}  # Thêm dictionary để quản lý entity nodes
        
        # Load environment variables
        load_dotenv()
        self.openai_client = None
        self._init_openai_client()
        
        # Semantic similarity components
        self.phobert_tokenizer = None
        self.phobert_model = None
        self.word_embeddings = {}  # Cache embeddings
        self.embedding_dim = 768  # PhoBERT base dimension
        self.pca_model = None
        self.faiss_index = None
        self.word_to_index = {}  # Mapping từ word -> index trong faiss
        self.index_to_word = {}  # Mapping ngược lại
        
        # Semantic similarity parameters
        self.similarity_threshold = 0.85
        self.top_k_similar = 5
        self.reduced_dim = 128  # Dimension after PCA reduction
        
        self._init_phobert_model()
    
    def add_word_node(self, word, pos_tag=None, lemma=None):
        """Thêm word node vào đồ thị"""
        if word not in self.word_nodes:
            node_id = f"word_{len(self.word_nodes)}"
            self.word_nodes[word] = node_id
            self.graph.add_node(node_id, 
                              type="word", 
                              text=word, 
                              pos=pos_tag, 
                              lemma=lemma)
        return self.word_nodes[word]
    
    def add_sentence_node(self, sentence_id, sentence_text):
        """Thêm sentence node vào đồ thị"""
        node_id = f"sentence_{sentence_id}"
        self.sentence_nodes[sentence_id] = node_id
        self.graph.add_node(node_id, 
                          type="sentence", 
                          text=sentence_text)
        return node_id
    
    def add_claim_node(self, claim_text):
        """Thêm claim node vào đồ thị"""
        self.claim_node = "claim_0"
        self.graph.add_node(self.claim_node, 
                          type="claim", 
                          text=claim_text)
        return self.claim_node
    
    def connect_word_to_sentence(self, word_node, sentence_node):
        """Kết nối word với sentence"""
        self.graph.add_edge(word_node, sentence_node, relation="belongs_to", edge_type="structural")
    
    def connect_word_to_claim(self, word_node, claim_node):
        """Kết nối word với claim"""
        self.graph.add_edge(word_node, claim_node, relation="belongs_to", edge_type="structural")
    
    def connect_dependency(self, dependent_word_node, head_word_node, dep_label):
        """Kết nối dependency giữa hai từ"""
        self.graph.add_edge(dependent_word_node, head_word_node, 
                          relation=dep_label, edge_type="dependency")
    
    def build_from_vncorenlp_output(self, context_sentences, claim_text, claim_sentences):
        """Xây dựng đồ thị từ kết quả py_vncorenlp"""
        
        # Thêm claim node
        claim_node = self.add_claim_node(claim_text)
        
        # Xử lý các câu trong context (context_sentences là dict)
        for sent_idx, sentence_tokens in context_sentences.items():
            sentence_text = " ".join([token["wordForm"] for token in sentence_tokens])
            sentence_node = self.add_sentence_node(sent_idx, sentence_text)
            
            # Dictionary để map index -> word_node_id cho việc tạo dependency links
            token_index_to_node = {}
            
            # Thêm các word trong sentence
            for token in sentence_tokens:
                word = token["wordForm"]
                pos_tag = token.get("posTag", "")
                lemma = token.get("lemma", "")
                token_index = token.get("index", 0)
                
                word_node = self.add_word_node(word, pos_tag, lemma)
                self.connect_word_to_sentence(word_node, sentence_node)
                
                # Lưu mapping để tạo dependency links sau
                token_index_to_node[token_index] = word_node
            
            # Tạo dependency connections giữa các từ trong câu
            for token in sentence_tokens:
                token_index = token.get("index", 0)
                head_index = token.get("head", 0)
                dep_label = token.get("depLabel", "")
                
                # Nếu có head (không phải root) và head tồn tại trong câu
                if head_index > 0 and head_index in token_index_to_node:
                    dependent_node = token_index_to_node[token_index]
                    head_node = token_index_to_node[head_index]
                    self.connect_dependency(dependent_node, head_node, dep_label)
        
        # Xử lý các word trong claim (claim_sentences cũng là dict)
        for sent_idx, sentence_tokens in claim_sentences.items():
            # Dictionary để map index -> word_node_id cho claim
            claim_token_index_to_node = {}
            
            # Thêm words
            for token in sentence_tokens:
                word = token["wordForm"]
                pos_tag = token.get("posTag", "")
                lemma = token.get("lemma", "")
                token_index = token.get("index", 0)
                
                word_node = self.add_word_node(word, pos_tag, lemma)
                self.connect_word_to_claim(word_node, claim_node)
                
                # Lưu mapping cho dependency links
                claim_token_index_to_node[token_index] = word_node
            
            # Tạo dependency connections trong claim
            for token in sentence_tokens:
                token_index = token.get("index", 0)
                head_index = token.get("head", 0)
                dep_label = token.get("depLabel", "")
                
                # Nếu có head (không phải root) và head tồn tại trong claim
                if head_index > 0 and head_index in claim_token_index_to_node:
                    dependent_node = claim_token_index_to_node[token_index]
                    head_node = claim_token_index_to_node[head_index]
                    self.connect_dependency(dependent_node, head_node, dep_label)
    
    def get_statistics(self):
        """Thống kê cơ bản về đồ thị"""
        word_count = len([n for n in self.graph.nodes() if self.graph.nodes[n]['type'] == 'word'])
        sentence_count = len([n for n in self.graph.nodes() if self.graph.nodes[n]['type'] == 'sentence'])
        claim_count = len([n for n in self.graph.nodes() if self.graph.nodes[n]['type'] == 'claim'])
        entity_count = len([n for n in self.graph.nodes() if self.graph.nodes[n]['type'] == 'entity'])
        
        return {
            "total_nodes": self.graph.number_of_nodes(),
            "total_edges": self.graph.number_of_edges(),
            "word_nodes": word_count,
            "sentence_nodes": sentence_count,
            "claim_nodes": claim_count,
            "entity_nodes": entity_count
        }
    
    def get_shared_words(self):
        """Tìm các từ xuất hiện cả trong context và claim"""
        shared_words = []
        
        for word_node_id in self.word_nodes.values():
            # Kiểm tra xem word node có kết nối với cả sentence nodes và claim node không
            neighbors = list(self.graph.neighbors(word_node_id))
            has_sentence_connection = any(
                self.graph.nodes[neighbor]['type'] == 'sentence' for neighbor in neighbors
            )
            has_claim_connection = any(
                self.graph.nodes[neighbor]['type'] == 'claim' for neighbor in neighbors
            )
            
            if has_sentence_connection and has_claim_connection:
                word_text = self.graph.nodes[word_node_id]['text']
                pos_tag = self.graph.nodes[word_node_id]['pos']
                shared_words.append({
                    'word': word_text,
                    'pos': pos_tag,
                    'node_id': word_node_id
                })
        
        return shared_words
    
    def get_word_frequency(self):
        """Đếm tần suất xuất hiện của từng từ"""
        word_freq = {}
        for word_node_id in self.word_nodes.values():
            word_text = self.graph.nodes[word_node_id]['text']
            word_freq[word_text] = word_freq.get(word_text, 0) + 1
        return word_freq
    
    def get_dependency_statistics(self):
        """Thống kê về các mối quan hệ dependency"""
        dependency_edges = [
            (u, v, data) for u, v, data in self.graph.edges(data=True) 
            if data.get('edge_type') == 'dependency'
        ]
        
        # Đếm các loại dependency
        dep_types = {}
        for u, v, data in dependency_edges:
            dep_label = data.get('relation', 'unknown')
            dep_types[dep_label] = dep_types.get(dep_label, 0) + 1
        
        return {
            "total_dependency_edges": len(dependency_edges),
            "dependency_types": dep_types,
            "most_common_dependencies": sorted(dep_types.items(), key=lambda x: x[1], reverse=True)[:10]
        }
    
    def get_word_dependencies(self, word):
        """Lấy tất cả dependencies của một từ"""
        if word not in self.word_nodes:
            return {"dependents": [], "heads": []}
        
        word_node_id = self.word_nodes[word]
        dependents = []
        heads = []
        
        for neighbor in self.graph.neighbors(word_node_id):
            edge_data = self.graph.edges[word_node_id, neighbor]
            if edge_data.get('edge_type') == 'dependency':
                dep_relation = edge_data.get('relation', '')
                neighbor_word = self.graph.nodes[neighbor]['text']
                
                # Kiểm tra xem word_node_id là head hay dependent
                # Trong NetworkX undirected graph, cần kiểm tra hướng dựa trên semantic
                # Giả sử edge được tạo từ dependent -> head
                if (word_node_id, neighbor) in self.graph.edges():
                    heads.append({"word": neighbor_word, "relation": dep_relation})
                else:
                    dependents.append({"word": neighbor_word, "relation": dep_relation})
        
        return {"dependents": dependents, "heads": heads}
    
    def get_detailed_statistics(self):
        """Thống kê chi tiết về đồ thị"""
        basic_stats = self.get_statistics()
        shared_words = self.get_shared_words()
        word_freq = self.get_word_frequency()
        dep_stats = self.get_dependency_statistics()
        semantic_stats = self.get_semantic_statistics()
        
        # Tìm từ xuất hiện nhiều nhất
        most_frequent_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:5]
        
        # Tính tổng edges theo loại
        structural_edges = len([
            (u, v) for u, v, data in self.graph.edges(data=True) 
            if data.get('edge_type') == 'structural'
        ])
        
        entity_structural_edges = len([
            (u, v) for u, v, data in self.graph.edges(data=True) 
            if data.get('edge_type') == 'entity_structural'
        ])
        
        # Thống kê entity
        entity_list = [
            {
                'name': self.graph.nodes[node_id]['text'],
                'type': self.graph.nodes[node_id].get('entity_type', 'ENTITY'),
                'connected_sentences': len([
                    neighbor for neighbor in self.graph.neighbors(node_id) 
                    if self.graph.nodes[neighbor]['type'] == 'sentence'
                ])
            }
            for node_id in self.graph.nodes() 
            if self.graph.nodes[node_id]['type'] == 'entity'
        ]
        
        return {
            **basic_stats,
            "shared_words_count": len(shared_words),
            "shared_words": shared_words,
            "unique_words": len(word_freq),
            "most_frequent_words": most_frequent_words,
            "average_words_per_sentence": basic_stats['word_nodes'] / max(basic_stats['sentence_nodes'], 1),
            "dependency_statistics": dep_stats,
            "structural_edges": structural_edges,
            "dependency_edges": dep_stats["total_dependency_edges"],
            "entity_structural_edges": entity_structural_edges,
            "entities": entity_list,
            "unique_entities": len(entity_list),
            "semantic_statistics": semantic_stats,
            "semantic_edges": semantic_stats["total_semantic_edges"]
        }
    
    def visualize(self, figsize=(15, 10), show_dependencies=True, show_semantic=True):
        """Vẽ đồ thị với phân biệt structural, dependency, entity và semantic edges"""
        plt.figure(figsize=figsize)
        
        # Định nghĩa màu sắc cho các loại node
        node_colors = []
        node_sizes = []
        for node in self.graph.nodes():
            node_type = self.graph.nodes[node]['type']
            if node_type == 'word':
                node_colors.append('lightblue')
                node_sizes.append(200)
            elif node_type == 'sentence':
                node_colors.append('lightgreen')
                node_sizes.append(500)
            elif node_type == 'claim':
                node_colors.append('lightcoral')
                node_sizes.append(600)
            elif node_type == 'entity':
                node_colors.append('gold')
                node_sizes.append(400)
        
        # Tạo layout
        pos = nx.spring_layout(self.graph, k=2, iterations=100)
        
        # Phân chia edges theo loại
        structural_edges = []
        dependency_edges = []
        entity_edges = []
        semantic_edges = []
        
        for u, v, data in self.graph.edges(data=True):
            edge_type = data.get('edge_type', 'structural')
            if edge_type == 'structural':
                structural_edges.append((u, v))
            elif edge_type == 'dependency':
                dependency_edges.append((u, v))
            elif edge_type == 'entity_structural':
                entity_edges.append((u, v))
            elif edge_type == 'semantic':
                semantic_edges.append((u, v))
        
        # Vẽ nodes
        nx.draw_networkx_nodes(self.graph, pos, 
                             node_color=node_colors,
                             node_size=node_sizes,
                             alpha=0.8)
        
        # Vẽ structural edges (word -> sentence/claim)
        if structural_edges:
            nx.draw_networkx_edges(self.graph, pos,
                                 edgelist=structural_edges,
                                 edge_color='gray',
                                 style='-',
                                 width=1,
                                 alpha=0.6)
        
        # Vẽ entity edges (entity -> sentence)
        if entity_edges:
            nx.draw_networkx_edges(self.graph, pos,
                                 edgelist=entity_edges,
                                 edge_color='orange',
                                 style='-',
                                 width=2,
                                 alpha=0.7)
        
        # Vẽ semantic edges (word -> word)
        if show_semantic and semantic_edges:
            nx.draw_networkx_edges(self.graph, pos,
                                 edgelist=semantic_edges,
                                 edge_color='purple',
                                 style=':',
                                 width=1.5,
                                 alpha=0.8)
        
        # Vẽ dependency edges (word -> word)
        if show_dependencies and dependency_edges:
            nx.draw_networkx_edges(self.graph, pos,
                                 edgelist=dependency_edges,
                                 edge_color='red',
                                 style='--',
                                 width=0.8,
                                 alpha=0.7,
                                 arrows=True,
                                 arrowsize=10)
        
        # Thêm legend
        legend_elements = [
            mpatches.Patch(color='lightblue', label='Word nodes'),
            mpatches.Patch(color='lightgreen', label='Sentence nodes'),
            mpatches.Patch(color='lightcoral', label='Claim node'),
            mpatches.Patch(color='gold', label='Entity nodes')
        ]
        
        edge_legend = []
        if structural_edges:
            edge_legend.append(plt.Line2D([0], [0], color='gray', label='Structural edges'))
        if entity_edges:
            edge_legend.append(plt.Line2D([0], [0], color='orange', label='Entity edges'))
        if show_semantic and semantic_edges:
            edge_legend.append(plt.Line2D([0], [0], color='purple', linestyle=':', label='Semantic edges'))
        if show_dependencies and dependency_edges:
            edge_legend.append(plt.Line2D([0], [0], color='red', linestyle='--', label='Dependency edges'))
        
        legend_elements.extend(edge_legend)
        
        plt.legend(handles=legend_elements, loc='upper right')
        
        title = f"Text Graph: Words, Sentences, Claim, Entities ({len(self.entity_nodes)} entities)"
        if show_semantic and semantic_edges:
            title += f", Semantic ({len(semantic_edges)} edges)"
        if show_dependencies and dependency_edges:
            title += f", Dependencies ({len(dependency_edges)} edges)"
        
        plt.title(title)
        plt.axis('off')
        plt.tight_layout()
        plt.show()
    
    def visualize_dependencies_only(self, figsize=(12, 8)):
        """Vẽ chỉ dependency graph giữa các từ"""
        # Tạo subgraph chỉ với word nodes và dependency edges
        word_nodes = [n for n in self.graph.nodes() if self.graph.nodes[n]['type'] == 'word']
        dependency_edges = [
            (u, v) for u, v, data in self.graph.edges(data=True) 
            if data.get('edge_type') == 'dependency'
        ]
        
        if not dependency_edges:
            print("Không có dependency edges để vẽ!")
            return
        
        # Tạo subgraph
        subgraph = self.graph.edge_subgraph(dependency_edges).copy()
        
        plt.figure(figsize=figsize)
        
        # Layout cho dependency graph
        pos = nx.spring_layout(subgraph, k=1.5, iterations=100)
        
        # Vẽ nodes với labels
        nx.draw_networkx_nodes(subgraph, pos, 
                             node_color='lightblue',
                             node_size=300,
                             alpha=0.8)
        
        # Vẽ edges với labels
        nx.draw_networkx_edges(subgraph, pos,
                             edge_color='red',
                             style='-',
                             width=1.5,
                             alpha=0.7,
                             arrows=True,
                             arrowsize=15)
        
        # Thêm node labels (từ)
        node_labels = {node: self.graph.nodes[node]['text'][:10] 
                      for node in subgraph.nodes()}
        nx.draw_networkx_labels(subgraph, pos, node_labels, font_size=8)
        
        # Thêm edge labels (dependency relations)
        edge_labels = {(u, v): data.get('relation', '') 
                      for u, v, data in subgraph.edges(data=True)}
        nx.draw_networkx_edge_labels(subgraph, pos, edge_labels, font_size=6)
        
        plt.title(f"Dependency Graph ({len(dependency_edges)} dependencies)")
        plt.axis('off')
        plt.tight_layout()
        plt.show()
    
    def save_graph(self, filepath):
        """Lưu đồ thị vào file"""
        # Đảm bảo lưu file vào thư mục gốc của project
        if not os.path.isabs(filepath):
            # Lấy thư mục cha của thư mục mint
            project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            filepath = os.path.join(project_root, filepath)
        
        # Tạo một bản copy của graph để xử lý None values
        graph_copy = self.graph.copy()
        
        # Xử lý None values trong node attributes
        for node_id in graph_copy.nodes():
            node_data = graph_copy.nodes[node_id]
            for key, value in node_data.items():
                if value is None:
                    graph_copy.nodes[node_id][key] = ""
        
        # Xử lý None values trong edge attributes
        for u, v in graph_copy.edges():
            edge_data = graph_copy.edges[u, v]
            for key, value in edge_data.items():
                if value is None:
                    graph_copy.edges[u, v][key] = ""
        
        nx.write_gexf(graph_copy, filepath)
        print(f"Đồ thị đã được lưu vào: {filepath}")
    
    def load_graph(self, filepath):
        """Tải đồ thị từ file"""
        self.graph = nx.read_gexf(filepath)
        
        # Rebuild node mappings
        self.word_nodes = {}
        self.sentence_nodes = {}
        self.entity_nodes = {}
        self.claim_node = None
        
        for node_id in self.graph.nodes():
            node_data = self.graph.nodes[node_id]
            if node_data['type'] == 'word':
                self.word_nodes[node_data['text']] = node_id
            elif node_data['type'] == 'sentence':
                # Extract sentence index from node_id
                sent_idx = int(node_id.split('_')[1])
                self.sentence_nodes[sent_idx] = node_id
            elif node_data['type'] == 'claim':
                self.claim_node = node_id
            elif node_data['type'] == 'entity':
                self.entity_nodes[node_data['text']] = node_id
        
        print(f"Đồ thị đã được tải từ: {filepath}")
    
    def export_to_json(self):
        """Xuất đồ thị ra định dạng JSON để dễ dàng phân tích"""
        graph_data = {
            "nodes": [],
            "edges": [],
            "statistics": self.get_detailed_statistics()
        }
        
        # Export nodes
        for node_id in self.graph.nodes():
            node_data = self.graph.nodes[node_id]
            graph_data["nodes"].append({
                "id": node_id,
                "type": node_data["type"],
                "text": node_data["text"],
                "pos": node_data.get("pos", ""),
                "lemma": node_data.get("lemma", "")
            })
        
        # Export edges
        for edge in self.graph.edges():
            edge_data = self.graph.edges[edge]
            graph_data["edges"].append({
                "source": edge[0],
                "target": edge[1],
                "relation": edge_data.get("relation", ""),
                "edge_type": edge_data.get("edge_type", "")
            })
        
        return json.dumps(graph_data, ensure_ascii=False, indent=2)
    
    def _init_openai_client(self):
        """Khởi tạo OpenAI client"""
        try:
            # Try multiple key names for backward compatibility
            api_key = os.getenv('OPENAI_KEY') or os.getenv('OPENAI_API_KEY')
            if api_key and api_key != 'your_openai_api_key_here':
                self.openai_client = OpenAI(api_key=api_key)
                print("OpenAI client đã được khởi tạo thành công.")
            else:
                print("Warning: OPENAI_KEY hoặc OPENAI_API_KEY không được tìm thấy trong .env file.")
                print("Vui lòng tạo file .env và thêm OPENAI_KEY=your_api_key")
        except Exception as e:
            print(f"Lỗi khi khởi tạo OpenAI client: {e}")
    
    def add_entity_node(self, entity_name, entity_type="ENTITY"):
        """Thêm entity node vào đồ thị"""
        if entity_name not in self.entity_nodes:
            node_id = f"entity_{len(self.entity_nodes)}"
            self.entity_nodes[entity_name] = node_id
            self.graph.add_node(node_id, 
                              type="entity", 
                              text=entity_name,
                              entity_type=entity_type)
        return self.entity_nodes[entity_name]
    
    def connect_entity_to_sentence(self, entity_node, sentence_node):
        """Kết nối entity với sentence"""
        self.graph.add_edge(entity_node, sentence_node, relation="mentioned_in", edge_type="entity_structural")
    
    def _update_openai_model(self, model=None, temperature=None, max_tokens=None):
        """Update OpenAI model parameters"""
        if model:
            self.openai_model = model
        if temperature is not None:
            self.openai_temperature = temperature  
        if max_tokens is not None:
            self.openai_max_tokens = max_tokens
    
    def extract_entities_with_openai(self, context_text):
        """Trích xuất entities từ context bằng OpenAI GPT-4o-mini"""
        if not self.openai_client:
            print("OpenAI client chưa được khởi tạo. Không thể trích xuất entities.")
            return []
        
        try:
            # Prompt để trích xuất entities
            prompt = f"""
Trích xuất tất cả các thông tin thực thể quan trọng từ văn bản sau đây. 
Chỉ trả về tên các thực thể, không giải thích gì thêm.
Trả về danh sách các thực thể dưới dạng JSON array với format: ["entity1", "entity2", "entity3"]

Các loại thực thể cần trích xuất:
- Tên người
- Tên tổ chức/công ty
- Địa điểm
- Ngày tháng/thời gian
- Số liệu quan trọng
- Sản phẩm/dịch vụ
- Sự kiện

Văn bản:
{context_text}
"""

            # Use parameters from CLI if available
            model = getattr(self, 'openai_model', 'gpt-4o-mini')
            temperature = getattr(self, 'openai_temperature', 0.0)
            max_tokens = getattr(self, 'openai_max_tokens', 1000)

            response = self.openai_client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "user", "content": prompt}
                ],
                temperature=temperature,
                max_tokens=max_tokens
            )
            
            # Parse response
            response_text = response.choices[0].message.content.strip()
            
            # Strip markdown code blocks if present
            if response_text.startswith('```json'):
                response_text = response_text[7:]  # Remove '```json'
            if response_text.startswith('```'):
                response_text = response_text[3:]   # Remove '```'
            if response_text.endswith('```'):
                response_text = response_text[:-3]  # Remove ending '```'
            response_text = response_text.strip()
            
            # Cố gắng parse JSON
            try:
                entities = json.loads(response_text)
                if isinstance(entities, list):
                    # Filter out empty strings and duplicates
                    entities = list(set([entity.strip() for entity in entities if entity.strip()]))
                    print(f"Đã trích xuất được {len(entities)} entities: {entities}")
                    return entities
                else:
                    print(f"Response không phải dạng list: {response_text}")
                    return []
            except json.JSONDecodeError:
                print(f"Không thể parse JSON từ OpenAI response: {response_text}")
                return []
                
        except Exception as e:
            print(f"Lỗi khi gọi OpenAI API: {e}")
            return []
    
    def add_entities_to_graph(self, entities, context_sentences):
        """Thêm entities vào graph và kết nối với sentences nếu được nhắc đến"""
        entity_nodes_added = []
        
        for entity in entities:
            # Thêm entity node
            entity_node = self.add_entity_node(entity)
            entity_nodes_added.append(entity_node)
            
            # Tìm các sentences có chứa entity này
            for sent_idx, sentence_node in self.sentence_nodes.items():
                sentence_text = self.graph.nodes[sentence_node]['text'].lower()
                entity_lower = entity.lower()
                
                # Kiểm tra xem entity có xuất hiện trong sentence không
                if entity_lower in sentence_text:
                    self.connect_entity_to_sentence(entity_node, sentence_node)
                    print(f"Đã kết nối entity '{entity}' với sentence {sent_idx}")
        
        print(f"Đã thêm {len(entity_nodes_added)} entity nodes vào graph.")
        return entity_nodes_added
    
    def extract_and_add_entities(self, context_text, context_sentences):
        """Phương thức chính để trích xuất và thêm entities vào graph"""
        print("Đang trích xuất entities từ OpenAI...")
        entities = self.extract_entities_with_openai(context_text)
        
        if entities:
            print("Đang thêm entities vào graph...")
            entity_nodes = self.add_entities_to_graph(entities, context_sentences)
            print(f"Hoàn thành! Đã thêm {len(entity_nodes)} entities vào graph.")
            return entity_nodes
        else:
            print("Không có entities nào được trích xuất.")
            return []
    
    def _init_phobert_model(self):
        """Khởi tạo PhoBERT model"""
        try:
            self.phobert_tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base")
            self.phobert_model = AutoModel.from_pretrained("vinai/phobert-base")
            print("PhoBERT model đã được khởi tạo thành công.")
        except Exception as e:
            print(f"Lỗi khi khởi tạo PhoBERT model: {e}")
    
    def get_word_embeddings(self, words):
        """Lấy embeddings của các từ"""
        if not self.phobert_tokenizer or not self.phobert_model:
            print("PhoBERT model chưa được khởi tạo. Không thể lấy embeddings.")
            return None
        
        embeddings = []
        for word in words:
            if word not in self.word_embeddings:
                inputs = self.phobert_tokenizer(word, return_tensors="pt")
                with torch.no_grad():
                    outputs = self.phobert_model(**inputs)
                embeddings.append(outputs.last_hidden_state.mean(dim=1).squeeze().numpy())
                self.word_embeddings[word] = embeddings[-1]
            else:
                embeddings.append(self.word_embeddings[word])
        
        return np.array(embeddings)
    
    def get_similarity(self, word1, word2):
        """Tính độ tương đồng giữa hai từ"""
        if word1 not in self.word_embeddings or word2 not in self.word_embeddings:
            print(f"Từ '{word1}' hoặc '{word2}' không có trong word_embeddings.")
            return 0.0
        
        embedding1 = self.word_embeddings[word1]
        embedding2 = self.word_embeddings[word2]
        return cosine_similarity([embedding1], [embedding2])[0][0]
    
    def get_similar_words(self, word, top_k=5):
        """Tìm các từ có độ tương đồng cao với từ đã cho"""
        if word not in self.word_embeddings:
            return []
        
        similarities = []
        for other_word in self.word_embeddings.keys():
            if other_word != word:
                similarity = self.get_similarity(word, other_word)
                similarities.append((other_word, similarity))
        
        similarities.sort(key=lambda x: x[1], reverse=True)
        return [word for word, similarity in similarities[:top_k]]
    
    def get_sentence_embeddings(self, sentences):
        """Lấy embeddings của các câu"""
        if not self.phobert_tokenizer or not self.phobert_model:
            print("PhoBERT model chưa được khởi tạo. Không thể lấy embeddings.")
            return None
        
        embeddings = []
        for sentence in sentences:
            inputs = self.phobert_tokenizer(sentence, return_tensors="pt", truncation=True, max_length=256)
            with torch.no_grad():
                outputs = self.phobert_model(**inputs)
            embeddings.append(outputs.last_hidden_state.mean(dim=1).squeeze().numpy())
        
        return np.array(embeddings)
    
    def get_sentence_similarity(self, sentence1, sentence2):
        """Tính độ tương đồng giữa hai câu"""
        # Lấy embeddings cho cả 2 câu
        embeddings = self.get_sentence_embeddings([sentence1, sentence2])
        if embeddings is None or len(embeddings) < 2:
            return 0.0
        
        return cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
    

    
    def build_semantic_similarity_edges(self, use_pca=True, use_faiss=True):
        """Xây dựng các cạnh semantic similarity giữa các từ"""
        print("Đang bắt đầu xây dựng semantic similarity edges...")
        
        # Lấy tất cả word nodes
        word_nodes = [node_id for node_id in self.graph.nodes() 
                     if self.graph.nodes[node_id]['type'] == 'word']
        
        if len(word_nodes) < 2:
            print("Cần ít nhất 2 word nodes để xây dựng semantic edges.")
            return
        
        # Lấy danh sách từ và POS tags
        words = []
        pos_tags = []
        word_node_mapping = {}
        
        for node_id in word_nodes:
            word = self.graph.nodes[node_id]['text']
            pos = self.graph.nodes[node_id].get('pos', '')
            words.append(word)
            pos_tags.append(pos)
            word_node_mapping[word] = node_id
        
        print(f"Đang lấy embeddings cho {len(words)} từ...")
        
        # Lấy embeddings
        embeddings = self.get_word_embeddings(words)
        if embeddings is None:
            print("Không thể lấy embeddings.")
            return
        
        print(f"Đã lấy embeddings với shape: {embeddings.shape}")
        
        # Áp dụng PCA để giảm chiều (optional)
        if use_pca and embeddings.shape[1] > self.reduced_dim:
            print(f"Đang áp dụng PCA để giảm từ {embeddings.shape[1]} xuống {self.reduced_dim} chiều...")
            self.pca_model = PCA(n_components=self.reduced_dim)
            embeddings_reduced = self.pca_model.fit_transform(embeddings)
            print(f"PCA hoàn thành. Shape mới: {embeddings_reduced.shape}")
        else:
            embeddings_reduced = embeddings
        
        # Xây dựng Faiss index (optional)
        if use_faiss:
            print("Đang xây dựng Faiss index...")
            dimension = embeddings_reduced.shape[1]
            self.faiss_index = faiss.IndexFlatIP(dimension)  # Inner Product (for cosine similarity)
            
            # Normalize vectors for cosine similarity
            embeddings_normalized = embeddings_reduced / np.linalg.norm(embeddings_reduced, axis=1, keepdims=True)
            self.faiss_index.add(embeddings_normalized.astype(np.float32))
            
            # Create mappings
            self.word_to_index = {word: i for i, word in enumerate(words)}
            self.index_to_word = {i: word for i, word in enumerate(words)}
            print("Faiss index đã được xây dựng.")
        
        # Tìm similar words và tạo edges
        edges_added = 0
        print(f"Đang tìm từ tương đồng với threshold={self.similarity_threshold}, top_k={self.top_k_similar}...")
        
        for i, word1 in enumerate(words):
            pos1 = pos_tags[i]
            node1 = word_node_mapping[word1]
            
            if use_faiss and self.faiss_index is not None:
                # Sử dụng Faiss để tìm similar words
                query_vector = embeddings_normalized[i:i+1].astype(np.float32)
                similarities, indices = self.faiss_index.search(query_vector, self.top_k_similar + 1)  # +1 vì sẽ bao gồm chính nó
                
                for j, (similarity, idx) in enumerate(zip(similarities[0], indices[0])):
                    if idx == i:  # Skip chính nó
                        continue
                    
                    if similarity < self.similarity_threshold:
                        continue
                    
                    word2 = self.index_to_word[idx]
                    pos2 = pos_tags[idx]
                    node2 = word_node_mapping[word2]
                    
                    # Chỉ kết nối từ cùng loại POS (optional)
                    if pos1 and pos2 and pos1 == pos2:
                        if not self.graph.has_edge(node1, node2):
                            self.graph.add_edge(node1, node2, 
                                              relation="semantic_similar", 
                                              edge_type="semantic",
                                              similarity=float(similarity))
                            edges_added += 1
            else:
                # Sử dụng brute force comparison
                for j, word2 in enumerate(words):
                    if i >= j:  # Tránh duplicate và self-comparison
                        continue
                    
                    pos2 = pos_tags[j]
                    
                    # Chỉ so sánh từ cùng loại POS
                    if pos1 and pos2 and pos1 != pos2:
                        continue
                    
                    # Tính cosine similarity
                    similarity = cosine_similarity([embeddings_reduced[i]], [embeddings_reduced[j]])[0][0]
                    
                    if similarity >= self.similarity_threshold:
                        node2 = word_node_mapping[word2]
                        if not self.graph.has_edge(node1, node2):
                            self.graph.add_edge(node1, node2, 
                                              relation="semantic_similar", 
                                              edge_type="semantic",
                                              similarity=float(similarity))
                            edges_added += 1
        
        print(f"Đã thêm {edges_added} semantic similarity edges.")
        return edges_added
    
    def get_semantic_statistics(self):
        """Thống kê về semantic edges"""
        semantic_edges = [
            (u, v, data) for u, v, data in self.graph.edges(data=True) 
            if data.get('edge_type') == 'semantic'
        ]
        
        if not semantic_edges:
            return {
                "total_semantic_edges": 0,
                "average_similarity": 0.0,
                "similarity_distribution": {}
            }
        
        similarities = [data.get('similarity', 0.0) for u, v, data in semantic_edges]
        
        return {
            "total_semantic_edges": len(semantic_edges),
            "average_similarity": np.mean(similarities),
            "max_similarity": np.max(similarities),
            "min_similarity": np.min(similarities),
            "similarity_distribution": {
                "0.85-0.90": len([s for s in similarities if 0.85 <= s < 0.90]),
                "0.90-0.95": len([s for s in similarities if 0.90 <= s < 0.95]),
                "0.95-1.00": len([s for s in similarities if 0.95 <= s <= 1.00])
            }
        } 