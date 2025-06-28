# MINT TextGraph CLI - Hướng dẫn sử dụng

## 🚀 Giới thiệu

MINT TextGraph CLI là công cụ dòng lệnh mạnh mẽ để phân tích đồ thị văn bản tiếng Việt với các tính năng:
- ✅ Phân tích dependency parsing
- ✅ Trích xuất entities từ OpenAI GPT
- ✅ Semantic similarity với PhoBERT + FAISS
- ✅ Visualization và export đa định dạng
- ✅ Config thông qua .env file
- ✅ Demo data linh hoạt
- ✅ **Auto GPU/CPU detection và optimization**

## 📦 Cài đặt

```bash
# Cài đặt package
pip install -e .

# Kiểm tra commands có sẵn
mint-graph --help
textgraph --help
```

## 🔧 Cấu hình

### 1. Tạo file .env
```bash
# Copy từ template
cp config.env .env

# Chỉnh sửa values
nano .env
```

### 2. Cấu trúc .env file
```bash
# OpenAI API Configuration  
OPENAI_KEY=your-openai-api-key-here

# Default parameters (GPU baseline - auto-optimize for CPU)
DEFAULT_SIMILARITY_THRESHOLD=0.85
DEFAULT_TOP_K=5
DEFAULT_PCA_DIMENSIONS=128
DEFAULT_OPENAI_MODEL=gpt-4o-mini
DEFAULT_FIGURE_WIDTH=15
DEFAULT_FIGURE_HEIGHT=10

# Demo Data
DEMO_DATA_PATH=data/demo.json

# Auto CPU Fallback (used when GPU not detected)
CPU_SIMILARITY_THRESHOLD=0.7
CPU_TOP_K=3
CPU_PCA_DIMENSIONS=64
CPU_USE_FAISS=false
```

### 3. Demo Data
Đặt demo data tại `data/demo.json`:
```json
{
  "context": "Văn bản context đầy đủ...",
  "claim": "Câu claim cần verify...",
  "evidence": "Evidence nếu có..."
}
```

## 📋 Commands chính

### 1. Demo nhanh
```bash
# Demo cơ bản (tự động detect GPU/CPU)
mint-graph --demo

# Demo với verbose (hiển thị device info)
mint-graph --demo --verbose

# Demo với custom parameters (override auto-optimization)
mint-graph --demo --similarity-threshold 0.9 --top-k 10
```

### 2. Phân tích text tùy chỉnh
```bash
# Từ command line
mint-graph --context "Văn bản context..." --claim "Văn bản claim..." 

# Từ file JSON
mint-graph --input-file your_data.json

# Với export ảnh
mint-graph --demo --export-image graph.png --figure-size 20,15
```

### 3. GPU/CPU Optimization
```bash
# Tự động detect và optimize
mint-graph --demo --verbose
# Output: 🖥️ Device detected: GPU (NVIDIA GeForce RTX 3080)
#         🔧 GPU optimizations applied: threshold=0.85, top_k=5...

# Override auto-optimization với custom values
mint-graph --demo --similarity-threshold 0.9 --top-k 10 --pca-dimensions 256

# Disable specific features nếu có vấn đề
mint-graph --demo --disable-faiss --disable-pca
```

## 🎛️ Auto-Optimization System

### 🤖 Automatic Device Detection:
- **PyTorch CUDA detection**: Tự động phát hiện GPU availability
- **Hardware profiling**: Memory, cores, capabilities
- **Zero configuration**: Không cần manual flags

### ⚡ Optimization Profiles:

**GPU Profile (auto-enabled when CUDA available):**
- Similarity threshold: 0.85 (higher precision)
- Top-K: 5 (more connections)
- PCA dimensions: 128 (full dimensionality)
- FAISS enabled: ✅ (fast similarity search)

**CPU Profile (auto-enabled when GPU not available):**
- Similarity threshold: 0.7 (lower precision, faster)
- Top-K: 3 (fewer connections)
- PCA dimensions: 64 (reduced dimensionality)
- FAISS disabled: ❌ (avoid compatibility issues)

### 🎯 Override System:
- **User-specified values** always take priority
- **Auto-optimization** only applies to default values
- **Verbose mode** shows applied optimizations

## 🎯 Ví dụ sử dụng

### 1. Setup lần đầu
```bash
# Copy config template
cp config.env .env

# Edit với API key thật
sed -i 's/your-openai-api-key-here/sk-your-real-key/' .env

# Test với auto-detection
mint-graph --demo --verbose
```

### 2. Server production
```bash
# Auto-optimized cho production
mint-graph --input-file data.json \
  --disable-visualization \
  --export-json results.json \
  --quiet
```

### 3. Research với GPU server
```bash
# GPU sẽ tự động sử dụng high-performance settings
mint-graph --input-file research_data.json \
  --similarity-threshold 0.95 \
  --top-k 15 \
  --pca-dimensions 512 \
  --export-image ultra_high_quality.png \
  --verbose
```

### 4. Batch processing (CPU/GPU agnostic)
```bash
# Script tự động optimize cho từng server
for file in data/*.json; do
    mint-graph --input-file "$file" \
      --disable-visualization \
      --export-json "results/$(basename "$file" .json)_result.json" \
      --quiet
done
```

### 5. Debug device detection
```bash
# Kiểm tra device được detect
mint-graph --demo --verbose
# Output sẽ hiển thị:
# 🖥️ Device detected: GPU (NVIDIA GeForce RTX 3080)
#   Memory: 10.0GB
# 🔧 GPU optimizations applied:
#   Similarity threshold: 0.85
#   Top-K: 5
#   PCA dimensions: 128
#   Use FAISS: True
```

## 🔍 Troubleshooting

### Device Detection Issues
```bash
# Kiểm tra PyTorch CUDA
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"

# Force check device detection
mint-graph --demo --verbose | grep "Device detected"
```

### Performance Issues
```bash
# GPU detected nhưng vẫn chậm - override với CPU values
mint-graph --demo --similarity-threshold 0.7 --top-k 3 --disable-faiss

# CPU quá chậm - giảm thêm parameters
mint-graph --demo --similarity-threshold 0.6 --top-k 2 --pca-dimensions 32
```

### Mixed Environments
```bash
# Development trên GPU, production trên CPU
# Không cần thay đổi gì - tự động optimize

# Test cả hai profiles
mint-graph --demo --verbose  # Hiển thị current optimization
```

## 📊 Config Reference

### Auto-Detection Config
- System tự động detect và áp dụng optimal values
- User-specified values luôn override auto-optimization
- Verbose mode hiển thị applied optimizations

### Base Configuration (.env)
- `DEFAULT_*`: GPU baseline values
- `CPU_*`: Fallback values cho CPU detection
- No manual device selection needed

### Override Behavior
1. **Auto-detect device** (GPU/CPU)
2. **Load base config** from .env
3. **Apply device optimization** if no user override
4. **Execute with optimized parameters**

## 🎯 Best Practices

### Hardware Optimization
- ✅ Để system tự động detect và optimize
- ✅ Sử dụng `--verbose` để xem applied optimizations
- ✅ Override specific parameters nếu cần tuning
- ❌ Không cần guess hardware capabilities

### Development Workflow
- **Local development**: Auto-optimize cho dev machine
- **Testing**: Verbose mode để verify optimizations
- **Production**: Quiet mode với auto-optimization
- **Debugging**: Verbose để troubleshoot performance

### Config Management
- **Base config**: Set reasonable GPU defaults trong .env
- **CPU fallback**: Configure conservative CPU values
- **Override**: Chỉ specify parameters khi cần customize
- **Portability**: Same CLI commands work trên mọi hardware

### Performance Tuning
- **Start with auto**: Để system optimize tự động
- **Profile**: Sử dụng verbose để see current settings
- **Iterate**: Override specific parameters nếu cần
- **Benchmark**: Test performance với different overrides

---

**Tip**: Chạy `mint-graph --demo --verbose` để xem device capabilities và optimizations! 