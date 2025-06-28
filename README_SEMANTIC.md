# Tính năng Semantic Similarity với PhoBERT (No PCA)

## Mô tả

TextGraph đã được tối ưu với tính năng **Semantic Similarity** sử dụng PhoBERT-base-v2. Đây là phiên bản tối ưu, cho phép:

- **Lấy embedding vectors** của tất cả các từ sử dụng PhoBERT-base-v2 (768 dimensions)
- **Sử dụng full embeddings** để có độ chính xác cao nhất cho cosine similarity
- **Xây dựng Faiss index** để tìm kiếm vector nhanh chóng
- **Tìm top-k từ tương đồng** (cùng POS tag) với cosine similarity > threshold
- **Tạo semantic edges** giữa các từ có ngữ nghĩa tương đồng

> **⚠️ PCA đã được loại bỏ** vì có thể làm sai lệch cosine similarity relationships

## Yêu cầu hệ thống

### Dependencies
```bash
pip install -r requirements.txt
```

Các thư viện chính:
- `transformers` - Để sử dụng PhoBERT
- `torch` - Backend cho PhoBERT  
- `faiss-cpu` - Tìm kiếm vector nhanh
- `scikit-learn` - PCA để giảm chiều

### Khuyến nghị phần cứng
- **GPU**: Khuyến nghị có GPU để xử lý PhoBERT nhanh hơn
- **RAM**: Ít nhất 8GB RAM cho văn bản vừa
- **CPU**: Có thể chạy trên CPU nhưng sẽ chậm hơn

## Cấu hình

### Parameters mặc định (No PCA)
```python
similarity_threshold = 0.85    # Ngưỡng cosine similarity
top_k_similar = 5             # Số từ tương đồng tối đa
embedding_dim = 768           # Chiều PhoBERT (full dimensions)
```

### Parameters cho server không GPU
```python
similarity_threshold = 0.9    # Tăng threshold để giảm số edge
top_k_similar = 3            # Giảm top-k để tiết kiệm memory
use_faiss = False            # Tắt FAISS, dùng numpy dot product
```

## Sử dụng

### Demo nhanh (server không GPU)
```bash
python demo_semantic.py
```

### Sử dụng trong code

```python
from mint import TextGraph

# Khởi tạo
text_graph = TextGraph()

# Xây dựng graph cơ bản trước
text_graph.build_from_vncorenlp_output(context_sentences, claim, claim_sentences)

# Tùy chỉnh parameters (optional)
text_graph.similarity_threshold = 0.85  # Giữ nguyên hoặc tăng cho CPU
text_graph.top_k_similar = 5            # Giảm cho CPU nếu cần

# Build semantic similarity edges (no PCA)
edges_added = text_graph.build_semantic_similarity_edges(
    use_faiss=True     # Sử dụng Faiss để tìm kiếm nhanh
)

print(f"Đã thêm {edges_added} semantic edges")
```

### Phân tích kết quả

```python
# Lấy thống kê semantic
stats = text_graph.get_detailed_statistics()
semantic_stats = stats['semantic_statistics']

print(f"Tổng semantic edges: {semantic_stats['total_semantic_edges']}")
print(f"Similarity trung bình: {semantic_stats['average_similarity']:.3f}")
print(f"Similarity cao nhất: {semantic_stats['max_similarity']:.3f}")

# Phân bố similarity
for range_key, count in semantic_stats['similarity_distribution'].items():
    if count > 0:
        print(f"  {range_key}: {count} edges")
```

### Visualization

```python
# Hiển thị đồ thị với semantic edges (màu tím, nét chấm)
text_graph.visualize(
    figsize=(15, 10), 
    show_dependencies=True, 
    show_semantic=True
)
```

## Cách hoạt động (Optimized - No PCA)

### 1. **Embedding Extraction**
- Sử dụng PhoBERT-base-v2 để lấy embeddings 768 chiều cho mỗi từ
- Cache embeddings để tránh tính toán lại
- **Giữ nguyên full 768 dimensions** để có độ chính xác cao nhất

### 2. **Vector Normalization**  
- Normalize vectors để tính cosine similarity hiệu quả
- Sử dụng L2 normalization: `v_norm = v / ||v||`
- Sau normalize: cosine(a,b) = dot(a_norm, b_norm)

### 3. **Faiss Indexing (Optional)**
- Xây dựng `IndexFlatIP` (Inner Product) index với full embeddings
- Normalize vectors trước khi add vào index
- Mapping word ↔ index trong Faiss

### 4. **Similarity Search**
- **FAISS mode**: Sử dụng Faiss để tìm top-k nhanh chóng
- **Brute force mode**: Sử dụng numpy dot product (nhanh hơn sklearn)
- Chỉ kết nối từ cùng POS tag (optional)
- Chỉ tạo edge nếu similarity ≥ threshold

### 5. **Edge Creation**
- Tạo edges với `edge_type="semantic"`
- Lưu `similarity` score trong edge attributes
- Tránh duplicate edges

## Kết quả

### Edge Types trong Graph
- **Structural edges** (xám): word → sentence/claim
- **Dependency edges** (đỏ, nét đứt): word → word (syntax)
- **Entity edges** (cam): entity → sentence  
- **Semantic edges** (tím, nét chấm): word → word (semantic)

### Thống kê mới
- Tổng số semantic edges
- Similarity trung bình/min/max
- Phân bố similarity theo range
- Edge count theo loại

## Performance Tips

### Tối ưu cho server không GPU
```python
# Tăng threshold để giảm số edges
text_graph.similarity_threshold = 0.9

# Giảm top-k để tiết kiệm memory
text_graph.top_k_similar = 3

# Chỉ test trên text ngắn
context = "Câu ngắn để test tính năng"
```

### Tối ưu cho server có GPU
```python
# Sử dụng GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
text_graph.phobert_model.to(device)

# Tăng batch size khi xử lý nhiều từ
# (cần customize trong get_word_embeddings)
```

## Troubleshooting

### Lỗi "No module named 'transformers'"
```bash
pip install transformers torch
```

### Lỗi "No module named 'faiss'"  
```bash
pip install faiss-cpu  # hoặc faiss-gpu nếu có GPU
```

### Lỗi OutOfMemory
- Giảm `top_k_similar`
- Tăng `similarity_threshold` để có ít edges hơn
- Test trên text ngắn hơn
- Đóng các ứng dụng khác

### Không tìm thấy similar words
- Giảm `similarity_threshold`
- Tăng `top_k_similar`
- Kiểm tra POS tags có đúng không
- Thử với text dài hơn (nhiều từ đa dạng hơn)

### Chậm quá
- Sử dụng GPU thay vì CPU
- Tăng `similarity_threshold` để giảm computation
- Giảm `top_k_similar`
- Set `use_faiss=True`
- Cache embeddings được sử dụng

## Ví dụ kết quả

```
=== SEMANTIC SIMILARITY STATISTICS ===
Tổng semantic edges: 12
Similarity trung bình: 0.756
Similarity cao nhất: 0.892
Similarity thấp nhất: 0.701

Phân bố similarity:
  0.70-0.80: 8 edges
  0.80-0.90: 4 edges

=== MỘT SỐ EDGES SEMANTIC ===
  'nước' <-> 'cấp' (similarity: 0.834)
  'thông báo' <-> 'báo cáo' (similarity: 0.782)
  'bảo trì' <-> 'sửa chữa' (similarity: 0.756)
```

## Tích hợp với tính năng khác

Semantic similarity có thể kết hợp với:
- **Entity extraction**: Tìm entities có ngữ nghĩa tương đồng
- **Dependency parsing**: So sánh với syntax similarity
- **Fact checking**: Tìm từ có ý nghĩa tương tự trong claim vs context 