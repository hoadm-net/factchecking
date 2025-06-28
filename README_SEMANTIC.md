# Tính năng Semantic Similarity với PhoBERT

## Mô tả

TextGraph đã được cập nhật với tính năng **Semantic Similarity** sử dụng PhoBERT-base-v2. Đây là tính năng nặng nhất, cho phép:

- **Lấy embedding vectors** của tất cả các từ sử dụng PhoBERT-base-v2
- **Giảm chiều vector** bằng PCA để tối ưu performance
- **Xây dựng Faiss index** để tìm kiếm vector nhanh chóng
- **Tìm top-k từ tương đồng** (cùng POS tag) với cosine similarity > threshold
- **Tạo semantic edges** giữa các từ có ngữ nghĩa tương đồng

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

### Parameters mặc định
```python
similarity_threshold = 0.85    # Ngưỡng cosine similarity
top_k_similar = 5             # Số từ tương đồng tối đa
reduced_dim = 128             # Số chiều sau PCA
embedding_dim = 768           # Chiều gốc của PhoBERT
```

### Parameters cho server không GPU
```python
similarity_threshold = 0.7    # Giảm threshold
top_k_similar = 3            # Giảm top-k  
reduced_dim = 64             # Giảm PCA dimensions
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
text_graph.similarity_threshold = 0.7
text_graph.top_k_similar = 3
text_graph.reduced_dim = 64

# Build semantic similarity edges
edges_added = text_graph.build_semantic_similarity_edges(
    use_pca=True,      # Sử dụng PCA để giảm chiều
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

## Cách hoạt động

### 1. **Embedding Extraction**
- Sử dụng PhoBERT-base-v2 để lấy embeddings 768 chiều cho mỗi từ
- Cache embeddings để tránh tính toán lại

### 2. **Dimensionality Reduction**  
- Áp dụng PCA để giảm từ 768 xuống `reduced_dim` chiều
- Giúp giảm memory và tăng tốc độ tìm kiếm

### 3. **Faiss Indexing**
- Xây dựng `IndexFlatIP` (Inner Product) index
- Normalize vectors để tính cosine similarity
- Mapping word ↔ index trong Faiss

### 4. **Similarity Search**
- Với mỗi từ, tìm top-k từ có similarity cao nhất
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
# Giảm threshold để dễ tìm thấy similar words
text_graph.similarity_threshold = 0.6

# Giảm dimensions
text_graph.reduced_dim = 32

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
- Giảm `reduced_dim`
- Giảm `top_k_similar`  
- Test trên text ngắn hơn
- Đóng các ứng dụng khác

### Không tìm thấy similar words
- Giảm `similarity_threshold`
- Tăng `top_k_similar`
- Kiểm tra POS tags có đúng không
- Thử với text dài hơn (nhiều từ đa dạng hơn)

### Chậm quá
- Sử dụng GPU thay vì CPU
- Giảm `reduced_dim` 
- Set `use_pca=True, use_faiss=True`
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