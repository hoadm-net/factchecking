# FactChecking với MINT TextGraph Library

Một hệ thống fact-checking tiếng Việt sử dụng **MINT TextGraph Library** để xây dựng và phân tích đồ thị văn bản với **Dependency Parsing**.

![TextGraph Structure](https://img.shields.io/badge/Graph-TextGraph-blue) ![Dependency](https://img.shields.io/badge/Parsing-Dependency-green) ![Vietnamese](https://img.shields.io/badge/Language-Vietnamese-red)

## 🚀 Tổng quan

Project này xây dựng một thư viện phân tích văn bản tiếng Việt dựa trên **đồ thị (Graph)** để hỗ trợ fact-checking. Hệ thống sử dụng **py_vncorenlp** để xử lý ngôn ngữ tự nhiên và tạo ra đồ thị phức tạp với các mối quan hệ dependency giữa các từ.

### ✨ Tính năng chính

- 🔗 **TextGraph với Dependency Parsing**: Xây dựng đồ thị với các mối quan hệ ngữ pháp
- 📊 **Phân tích thống kê đa chiều**: Word frequency, shared words, dependency relationships
- 🎨 **Visualization nâng cao**: Phân biệt structural và dependency edges
- 💾 **Export đa định dạng**: GEXF, JSON với đầy đủ metadata
- ⚡ **Fact-checking ready**: Tối ưu cho việc so sánh claim vs context

## 🏗️ Cấu trúc TextGraph

### 📋 Loại Nodes

| Node Type | Mô tả | Thuộc tính |
|-----------|-------|------------|
| **Word Node** | Từng từ trong văn bản | `text`, `pos`, `lemma` |
| **Sentence Node** | Câu trong context | `text` |
| **Claim Node** | Nội dung claim cần kiểm tra | `text` |

### 🔗 Loại Edges

| Edge Type | Mô tả | Relation |
|-----------|-------|----------|
| **Structural** | Kết nối word ↔ sentence/claim | `belongs_to` |
| **Dependency** | Mối quan hệ ngữ pháp word ↔ word | `nmod`, `vmod`, `sub`, `dob`, etc. |

### 📊 Dependency Relationships

Hệ thống nhận diện **18 loại dependency** phổ biến trong tiếng Việt:

```
nmod  (44 edges) - Noun modifier
vmod  (34 edges) - Verb modifier  
punct (25 edges) - Punctuation
pob   (15 edges) - Prepositional object
dob   (13 edges) - Direct object
det   (12 edges) - Determiner
sub   (9 edges)  - Subject
prp   (5 edges)  - Preposition
...
```

## 📁 Cấu trúc Project

```
FactChecking/
├── README.md                 # 📖 Documentation chính
├── main.py                   # 🎯 Script chính demo TextGraph
├── demo_dependency.py        # 🔬 Demo dependency parsing
├── requirements.txt          # 📦 Dependencies
├── text_graph.gexf          # 💾 Exported graph data
├── mint/                    # 📚 MINT Library
│   ├── __init__.py          
│   ├── text_graph.py        # 🧠 Core TextGraph class
│   └── README.md            # 📋 Library documentation
└── vncorenlp/               # 🤖 py_vncorenlp models
```

## 🔧 Cài đặt

### Prerequisites

```bash
# Java 8+ (cho py_vncorenlp)
sudo apt-get install openjdk-8-jdk

# Python dependencies
pip install -r requirements.txt
```

### Setup

```bash
git clone <repository>
cd FactChecking

# Tạo virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# hoặc venv\Scripts\activate  # Windows

# Cài đặt dependencies
pip install -r requirements.txt

# Download VnCoreNLP models (tự động khi chạy lần đầu)
python main.py
```

## 🎮 Sử dụng

### Basic Usage

```python
from mint import TextGraph
import py_vncorenlp

# Khởi tạo
model = py_vncorenlp.VnCoreNLP(save_dir="vncorenlp")
text_graph = TextGraph()

# Dữ liệu
context = "SAWACO thông báo tạm ngưng cung cấp nước..."
claim = "SAWACO ngưng cung cấp nước..."

# Xử lý
context_sentences = model.annotate_text(context)
claim_sentences = model.annotate_text(claim)

# Tạo đồ thị
text_graph.build_from_vncorenlp_output(
    context_sentences, claim, claim_sentences
)

# Thống kê
stats = text_graph.get_detailed_statistics()
print(f"Nodes: {stats['total_nodes']}")
print(f"Dependency edges: {stats['dependency_edges']}")
print(f"Shared words: {stats['shared_words_count']}")
```

### Advanced Analysis

```python
# Phân tích dependency của từ cụ thể
deps = text_graph.get_word_dependencies("SAWACO")
print("Heads:", [h['word'] for h in deps['heads']])
print("Dependents:", [d['word'] for d in deps['dependents']])

# Thống kê dependency
dep_stats = text_graph.get_dependency_statistics()
print("Most common dependencies:")
for dep_type, count in dep_stats['most_common_dependencies']:
    print(f"  {dep_type}: {count}")
```

### Visualization

```python
# Vẽ đồ thị đầy đủ
text_graph.visualize(show_dependencies=True)

# Chỉ vẽ dependency graph
text_graph.visualize_dependencies_only()
```

### Export & Import

```python
# Save/Load
text_graph.save_graph("my_graph.gexf")
text_graph.load_graph("my_graph.gexf")

# Export JSON
json_data = text_graph.export_to_json()
```

## 📊 Kết quả mẫu

### Với context và claim về SAWACO:

```
=== THỐNG KÊ CHI TIẾT ĐỒ THỊ TEXT ===
Tổng số nodes: 132
Tổng số edges: 382
  - Structural edges: 204
  - Dependency edges: 178
Word nodes: 124
Sentence nodes: 7
Claim nodes: 1
Số từ duy nhất: 124
Số từ chung giữa context và claim: 30
Trung bình từ mỗi câu: 17.7

=== DEPENDENCY RELATIONSHIPS PHỔ BIẾN NHẤT ===
'nmod': 44 lần    # Noun modifier
'vmod': 34 lần    # Verb modifier
'punct': 25 lần   # Punctuation
'pob': 15 lần     # Prepositional object
'dob': 13 lần     # Direct object
```

### Phân tích từ khóa "SAWACO":

```
Heads (từ mà SAWACO phụ thuộc vào):
  -> Tổng_Công_ty (relation: nmod)
  -> cho (relation: sub)
  -> thông_báo (relation: sub)
  -> có (relation: sub)
  -> chủ_động (relation: sub)
```

## 🎯 Ứng dụng Fact-checking

### 1. Semantic Similarity
So sánh độ tương đồng ngữ nghĩa dựa trên:
- Shared words ratio
- Dependency structure similarity
- POS tag distribution

### 2. Evidence Detection
Tìm evidence supporting/contradicting:
- Phân tích dependency paths
- Identify key semantic roles
- Cross-reference entities và relations

### 3. Contradiction Analysis
Phát hiện mâu thuẫn qua:
- Structural differences trong dependency
- Semantic role conflicts
- Temporal/numerical inconsistencies

### 4. Feature Extraction
Trích xuất features cho ML models:
- Graph centrality measures
- Dependency pattern vectors
- Semantic similarity scores

## 🚀 Demo Scripts

### 1. main.py
Demo cơ bản với full pipeline:
```bash
python main.py
```

### 2. demo_dependency.py
Demo chi tiết về dependency parsing:
```bash
python demo_dependency.py
```

### Output Demo:
```
🚀 MINT TextGraph - Dependency Parsing Demo
============================================================
🔍 DEMO VỚI CÂU ĐÚN GIẢN
📊 Thống kê cơ bản:
  - Nodes: 9
  - Edges: 20 (Structural: 12, Dependency: 8)
  - Shared words: 5
```

## 📈 Performance

### Benchmark với văn bản SAWACO:

| Metric | Value |
|--------|--------|
| **Processing time** | ~30s (first run với model loading) |
| **Graph construction** | ~2s |
| **Memory usage** | ~200MB |
| **Export time** | ~1s |

### Scalability:

- ✅ **Small texts** (1-5 sentences): < 5s
- ✅ **Medium texts** (5-20 sentences): 10-30s  
- ⚠️ **Large texts** (20+ sentences): 30s+

## 🛠️ API Reference

### TextGraph Class

#### Core Methods
- `build_from_vncorenlp_output(context, claim, claim_sentences)`
- `add_word_node(word, pos_tag, lemma)`
- `add_sentence_node(sentence_id, text)`
- `connect_dependency(dependent, head, dep_label)`

#### Analysis Methods
- `get_detailed_statistics()` → `dict`
- `get_dependency_statistics()` → `dict`
- `get_word_dependencies(word)` → `dict`
- `get_shared_words()` → `list`

#### Visualization
- `visualize(show_dependencies=True, figsize=(15,10))`
- `visualize_dependencies_only(figsize=(12,8))`

#### I/O Methods
- `save_graph(filepath)` → Save to GEXF
- `load_graph(filepath)` → Load from GEXF
- `export_to_json()` → JSON string

## 🤝 Đóng góp

### Roadmap tính năng:

- [ ] **Named Entity Linking**: Kết nối entities giữa claim và context
- [ ] **Coreference Resolution**: Xử lý đại từ và tham chiếu
- [ ] **Similarity Scoring**: Algorithms tính similarity score
- [ ] **ML Integration**: Features cho machine learning models
- [ ] **Multi-document**: Hỗ trợ multiple contexts
- [ ] **Real-time API**: REST API cho fact-checking service

### Contribute:

1. Fork repository
2. Tạo feature branch
3. Implement changes
4. Add tests và documentation
5. Submit pull request

## 📄 License

MIT License - Xem chi tiết trong file LICENSE

## 📞 Contact

- **Author**: Hòa Đinh
- **Project**: FactChecking with MINT TextGraph
- **Version**: 1.0.0

---

**⭐ Nếu project hữu ích, hãy star repository!** 