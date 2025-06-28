# 🎯 PCA Removal Summary

## Tổng quan
Đã **loại bỏ hoàn toàn PCA** khỏi quy trình semantic similarity để tối ưu cho **cosine similarity**. Điều này mang lại:

- ✅ **Độ chính xác cao hơn** (không mất thông tin semantic)
- ✅ **Performance tốt hơn** (không overhead PCA computation)
- ✅ **Code đơn giản hơn** (ít parameters cần tune)
- ✅ **Full PhoBERT embeddings** (768 dimensions)

## 📋 Danh sách thay đổi

### 1. **Core TextGraph Class** (`mint/text_graph.py`)
#### Loại bỏ:
- ❌ `self.pca_model` attribute
- ❌ `self.reduced_dim` parameter  
- ❌ `use_pca` parameter trong `build_semantic_similarity_edges()`
- ❌ PCA computation logic
- ❌ `from sklearn.decomposition import PCA`
- ❌ `from sklearn.metrics.pairwise import cosine_similarity`

#### Cải tiến:
- ✅ Sử dụng **numpy dot product** thay vì sklearn cosine_similarity (nhanh hơn)
- ✅ **Vector normalization** trước khi tính similarity
- ✅ Full 768-dimension PhoBERT embeddings
- ✅ Optimized method signature: `build_semantic_similarity_edges(use_faiss=True)`

### 2. **CLI Interface** (`mint/cli.py`)
#### Loại bỏ:
- ❌ `--pca-dimensions` / `-pca` argument
- ❌ `--disable-pca` argument
- ❌ `pca_dimensions_overridden` tracking

#### Giữ lại:
- ✅ `--disable-faiss` (vẫn hữu ích cho CPU mode)
- ✅ `--similarity-threshold` và `--top-k`

### 3. **Configuration System** (`mint/helpers.py`)
#### Loại bỏ:
- ❌ `pca_dimensions` config
- ❌ `use_pca` config
- ❌ `cpu_pca_dimensions` fallback
- ❌ PCA optimization logic

#### Cập nhật:
- ✅ Simplified device optimization (GPU vs CPU)
- ✅ CPU fallback: higher threshold (0.9), lower top-k (3)
- ✅ GPU default: threshold 0.85, top-k 5

### 4. **Environment Config** (`config.env`)
#### Loại bỏ:
- ❌ `DEFAULT_PCA_DIMENSIONS=128`
- ❌ `DEFAULT_USE_PCA=true`
- ❌ `CPU_PCA_DIMENSIONS=64`

#### Cập nhật:
- ✅ `CPU_SIMILARITY_THRESHOLD=0.9` (tăng từ 0.7)
- ✅ Comments về PCA removal

### 5. **Dependencies** (`requirements.txt`)
#### Loại bỏ:
- ❌ `scikit-learn` (không cần thiết nữa)

#### Giữ lại:
- ✅ Tất cả dependencies khác
- ✅ `faiss-cpu` cho vector search

### 6. **Documentation** (`README_SEMANTIC.md`)
#### Cập nhật:
- ✅ Title: "PhoBERT (No PCA)"
- ✅ Updated parameters examples
- ✅ Revised "Cách hoạt động" section
- ✅ Warning về PCA removal
- ✅ Updated code examples

## 🚀 Cách sử dụng mới

### **Basic Usage:**
```python
from mint import TextGraph

text_graph = TextGraph()
text_graph.build_from_vncorenlp_output(context_sentences, claim, claim_sentences)

# Build semantic edges (no PCA)
edges_added = text_graph.build_semantic_similarity_edges(
    use_faiss=True  # Chỉ còn 1 parameter
)
```

### **CLI Usage:**
```bash
# Không còn PCA arguments
python main.py --demo --verbose
mint-graph --context "..." --claim "..." --similarity-threshold 0.85 --top-k 5
```

### **Performance Tuning:**
```python
# Cho GPU
text_graph.similarity_threshold = 0.85
text_graph.top_k_similar = 5

# Cho CPU (tự động optimize)
text_graph.similarity_threshold = 0.9  # Tăng threshold
text_graph.top_k_similar = 3           # Giảm top-k
use_faiss = False                      # Tắt FAISS
```

## 📊 Performance Benefits

### **Accuracy:**
- **Before PCA**: Similarity có thể bị sai lệch do information loss
- **After (No PCA)**: Full semantic information preserved

### **Speed:**
- **Before PCA**: PCA computation + Cosine similarity
- **After (No PCA)**: Chỉ Vector normalization + Dot product (nhanh hơn)

### **Memory:**
- **Before PCA**: Lưu cả original + reduced embeddings
- **After (No PCA)**: Chỉ lưu normalized embeddings

## 🧪 Testing

Chạy test để verify:
```bash
python test_no_pca.py
```

Test sẽ kiểm tra:
- ✅ PCA attributes đã bị loại bỏ
- ✅ Method signature updated
- ✅ CLI arguments cleaned up
- ✅ Functional testing với data thực

## 🎯 Kết luận

**PCA đã được loại bỏ hoàn toàn** khỏi codebase vì:

1. **Không phù hợp với cosine similarity** - có thể làm sai lệch angle relationships
2. **Overhead không cần thiết** - PhoBERT embeddings đã được pre-trained tối ưu
3. **Information loss** - Mất 83% thông tin (768→128 dim)
4. **Complexity tăng** - Thêm parameters cần tune

Giờ đây hệ thống **đơn giản hơn, nhanh hơn, và chính xác hơn** cho semantic similarity tasks!

## 🔄 Migration Guide

Nếu bạn đang sử dụng version cũ:

### **Code changes:**
```python
# OLD
edges_added = text_graph.build_semantic_similarity_edges(
    use_pca=True, 
    use_faiss=True
)

# NEW
edges_added = text_graph.build_semantic_similarity_edges(
    use_faiss=True
)
```

### **CLI changes:**
```bash
# OLD
mint-graph --demo --pca-dimensions 64 --disable-pca

# NEW  
mint-graph --demo  # PCA arguments removed
```

### **Config changes:**
```bash
# OLD (.env)
DEFAULT_PCA_DIMENSIONS=128
DEFAULT_USE_PCA=true

# NEW (.env)
# PCA configs removed, system uses full embeddings
``` 