# New Features - Tính năng mới

## POS Filtering (Lọc từ loại) - MẶC ĐỊNH ĐÃ BẬT

Tính năng **POS Filtering** cho phép lọc các từ trong quá trình tạo word nodes dựa trên từ loại (Part-of-Speech tags). Điều này giúp:

- ✅ **Giảm nhiễu** trong đồ thị
- ✅ **Tăng hiệu suất** xử lý
- ✅ **Cải thiện chất lượng** semantic analysis
- ✅ **Tập trung vào từ có ý nghĩa** cho fact-checking

## Auto-Save Graph - MẶC ĐỊNH ĐÃ BẬT

Tính năng **Auto-Save** tự động lưu đồ thị sau khi build xong:

- ✅ **Tự động lưu** mỗi khi tạo đồ thị
- ✅ **Timestamp unique** tránh ghi đè
- ✅ **Tạo thư mục tự động** nếu chưa có
- ✅ **Dễ dàng tùy chỉnh** đường dẫn lưu

## Beam Search Path Finding - TÙY CHỌN

Tính năng **Beam Search** tìm đường đi từ claim đến sentence nodes:

- 🎯 **Tìm đường đi tốt nhất** từ claim đến sentences
- 📊 **Đánh giá paths** dựa trên word overlap và entities
- 💾 **Export kết quả** ra JSON và text summary
- ⚙️ **Tùy chỉnh parameters**: beam width, max depth, scoring weights

## Cách sử dụng

### 1. Qua CLI

```bash
# POS filtering đã BẬT MẶC ĐỊNH - chạy bình thường
mint-graph --demo --verbose

# Tắt POS filtering nếu cần (giữ tất cả từ)
mint-graph --demo --disable-pos-filtering --verbose

# Tùy chỉnh từ loại muốn giữ lại (vẫn bật filtering)
mint-graph --demo --pos-tags "N,Np,V,A" --verbose

# Kết hợp với các tính năng khác
mint-graph --context "..." --claim "..." --export-image graph.png

# Tùy chỉnh auto-save path
mint-graph --demo --auto-save-path "my_graphs/graph_{timestamp}.gexf" --verbose

# Beam Search để tìm paths từ claim đến sentences
mint-graph --demo --beam-search --beam-width 12 --beam-max-depth 7 --verbose

# Beam Search với tùy chỉnh parameters
mint-graph --demo --beam-search --beam-width 15 --beam-max-paths 25 --beam-export-dir "my_paths" --verbose
```

### 2. Qua Python API

```python
from mint.text_graph import TextGraph

# Tạo đồ thị với POS filtering
graph = TextGraph()

# Bật với cấu hình mặc định
graph.set_pos_filtering(enable=True)

# Hoặc tùy chỉnh từ loại
graph.set_pos_filtering(enable=True, custom_pos_tags={'N', 'Np', 'V', 'A'})

# Xây dựng đồ thị
graph.build_from_vncorenlp_output(context_sentences, claim, claim_sentences)

# Beam Search để tìm paths từ claim đến sentences
paths = graph.beam_search_paths(
    beam_width=10,
    max_depth=6, 
    max_paths=20
)

# Phân tích chất lượng paths
stats = graph.analyze_paths_quality(paths)
print(f"Found {stats['total_paths']} paths")
print(f"Average score: {stats['avg_score']:.3f}")
print(f"Paths to sentences: {stats['paths_to_sentences']}")

# Export kết quả
json_file, summary_file = graph.export_beam_search_results(
    paths, 
    output_dir="output",
    file_prefix="my_beam_search"
)
```

## Cấu hình từ loại

### Từ loại mặc định (được giữ lại):

- **N**: Danh từ thường
- **Np**: Danh từ riêng 
- **V**: Động từ
- **A**: Tính từ
- **Nc**: Danh từ chỉ người
- **M**: Số từ
- **R**: Trạng từ
- **P**: Đại từ

### Từ loại thường bị lọc bỏ:

- **E**: Giới từ (tại, trong, của, ...)
- **C**: Liên từ (và, hoặc, nhưng, ...)
- **CH**: Dấu câu (. , ! ? ...)
- **L**: Định từ (các, những, ...)
- **T**: Thán từ

## Demo script

Chạy script demo để xem hiệu quả:

```bash
python demo_pos_filtering.py
```

Script này sẽ:
- So sánh đồ thị có và không có POS filtering
- Hiển thị tỷ lệ giảm nodes/edges
- Phân tích chi tiết các từ được giữ lại và lọc bỏ

## Kết quả thực nghiệm

Với dữ liệu demo, POS filtering thường mang lại:

- 🔸 **Giảm 30-50%** tổng số nodes
- 🔸 **Giảm 35-60%** word nodes  
- 🔸 **Giảm 20-40%** tổng số edges
- 🔸 **Tăng tốc** xử lý semantic similarity
- 🔸 **Cải thiện** chất lượng phân tích

## Lưu ý kỹ thuật

1. **Dependency parsing**: Chỉ tạo dependency edges giữa các từ được giữ lại
2. **Semantic similarity**: Chỉ tính similarity cho các từ quan trọng
3. **Entity extraction**: Không bị ảnh hưởng, vẫn hoạt động bình thường
4. **Tương thích ngược**: Mặc định tắt, không ảnh hưởng code cũ

## Khuyến nghị sử dụng

### ✅ Nên bật POS filtering khi:
- Văn bản dài, phức tạp
- Cần giảm thời gian xử lý
- Tập trung vào content chính
- Xây dựng hệ thống production

### ❌ Có thể tắt khi:
- Phân tích ngôn ngữ học chi tiết
- Cần giữ nguyên cấu trúc câu
- Nghiên cứu về từ chức năng
- Debugging và phát triển

## Tùy chỉnh nâng cao

```python
# Cấu hình riêng cho domain cụ thể
medical_pos_tags = {'N', 'Np', 'V', 'A', 'M'}  # Y tế: tập trung vào thuật ngữ
legal_pos_tags = {'N', 'Np', 'V', 'A', 'R'}    # Pháp lý: bao gồm trạng từ

graph.set_pos_filtering(enable=True, custom_pos_tags=medical_pos_tags)
``` 