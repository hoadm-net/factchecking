# Sentence Extraction from Beam Search Results

Tính năng trích xuất và xếp hạng các câu từ kết quả Beam Search

## 🎯 Mục đích

Sau khi Beam Search tìm được các đường đi từ claim đến sentences, bạn có thể:
- **Trích xuất danh sách sentences** không trùng lặp
- **Xếp hạng theo tần suất** xuất hiện trong paths
- **Phân tích theo nhiều tiêu chí** khác nhau
- **Export kết quả** để khảo sát thêm

## 🚀 Cách sử dụng

### 1. Quick Extract (Đơn giản nhất)

```bash
# Tự động tìm kết quả beam search mới nhất và extract sentences
python3 extract_sentences.py
```

**Output:**
```
🏆 TOP 10 SENTENCES (by frequency):
1. [freq:2] Mỗi ngày hai vợ chồng ông Chính làm khoảng 5-7 kg gạo...
2. [freq:2] Gạo tẻ ngon được xay nhuyễn bằng cối đá từ xưa...
3. [freq:2] So với sự tỉ mỉ trong từng công đoạn để làm ra một đĩa bánh...
```

### 2. Advanced Analysis (Chi tiết hơn)

```bash
# Phân tích với file cụ thể
python3 analyze_beam_sentences.py --input "output/beam_search_20250629_015137.json" --ranking frequency --simple

# Xếp hạng theo average score
python3 analyze_beam_sentences.py --auto --ranking avg_score

# Xếp hạng theo combined score (frequency × avg_score)
python3 analyze_beam_sentences.py --auto --ranking combined --simple
```

### 3. Các phương pháp xếp hạng

| Method | Mô tả | Khi nào dùng |
|--------|-------|--------------|
| `frequency` | Số lần xuất hiện trong paths | **Mặc định** - tìm sentences phổ biến nhất |
| `avg_score` | Điểm trung bình của paths đi qua | Tìm sentences với paths chất lượng cao |
| `max_score` | Điểm cao nhất của paths đi qua | Tìm sentences có potential cao nhất |
| `total_score` | Tổng điểm tất cả paths | Kết hợp frequency và quality |
| `combined` | Frequency × Average Score | **Cân bằng** giữa phổ biến và chất lượng |

## 📊 Output Files

### Simple List (`simple_sentences_*.txt`)
```
# Sentences ranked by frequency
# Total: 12 unique sentences

 1. [  2.0] Mỗi ngày hai vợ chồng ông Chính làm khoảng 5-7 kg gạo...
 2. [  2.0] Gạo tẻ ngon được xay nhuyễn bằng cối đá từ xưa...
 3. [  1.0] Ông Chính dùng que tre dẹt cuốn lấy vỏ bánh...
```

### Detailed Analysis (`ranked_sentences_*.txt`)
```
📊 BEAM SEARCH - RANKED SENTENCES ANALYSIS

RANK # 1 │ Frequency:  2 │ Paths:  2
Text: Mỗi ngày hai vợ chồng ông Chính làm khoảng 5-7 kg gạo...
Stats: freq=2, avg_score=5.200, score_range=(5.200-5.200)
ID: sentence_40
```

## 🔍 Workflow hoàn chỉnh

### Bước 1: Chạy Beam Search
```bash
python3 main.py --demo --beam-search --beam-width 10 --beam-max-depth 6 --verbose
```

### Bước 2: Extract Sentences
```bash
python3 extract_sentences.py
```

### Bước 3: Phân tích chi tiết (tùy chọn)
```bash
python3 analyze_beam_sentences.py --auto --ranking combined --simple
```

## 📈 Ví dụ thực tế

**Input:** 15 paths từ beam search về "Phần vỏ bánh được làm rất công phu..."

**Output:** 12 unique sentences được ranked:

```
🏆 TOP 3 SENTENCES:
1. [freq:2] Mỗi ngày hai vợ chồng ông Chính làm khoảng 5-7 kg gạo , cho ra hơn 200 đĩa bánh .
2. [freq:2] Gạo tẻ ngon được xay nhuyễn bằng cối đá từ xưa , sau đó loại bỏ tạp chất...
3. [freq:2] So với sự tỉ mỉ trong từng công đoạn để làm ra một đĩa bánh thì giá bán...
```

**Phân tích:**
- **Sentence #1**: Về **quy mô sản xuất** - 2 paths dẫn đến
- **Sentence #2**: Về **quy trình xay gạo** - 2 paths dẫn đến  
- **Sentence #3**: Về **giá cả và công phu** - 2 paths dẫn đến

→ Những câu này có **mức độ liên quan cao** với claim về "làm vỏ bánh công phu"

## 🛠️ Tùy chỉnh nâng cao

### Custom ranking weights
```python
from analyze_beam_sentences import BeamSearchSentenceAnalyzer

analyzer = BeamSearchSentenceAnalyzer()
# Load your custom logic here
```

### Multiple ranking comparison
```bash
# So sánh nhiều phương pháp
for method in frequency avg_score combined; do
    python3 analyze_beam_sentences.py --auto --ranking $method --simple
done
```

## 🎯 Use Cases

### 1. **Fact-checking Research**
- Tìm sentences liên quan nhất đến claim
- Prioritize verification targets
- Identify key evidence sentences

### 2. **Content Analysis**  
- Phân tích mức độ coverage của claim
- Tìm gaps trong knowledge base
- Extract key information points

### 3. **Question Answering**
- Rank candidate answer sentences
- Find most relevant context
- Build evidence chains

## 💡 Tips & Best Practices

1. **Luôn dùng `frequency` làm baseline** trước khi thử methods khác
2. **`combined` ranking** thường cho kết quả cân bằng nhất
3. **Check detailed analysis** để hiểu tại sao sentences được ranked như vậy  
4. **So sánh multiple rankings** để có góc nhìn toàn diện
5. **Export both simple và detailed** để có đầy đủ thông tin

## 🔧 Troubleshooting

**❌ "No beam search files found"**
```bash
# Chạy beam search trước
python3 main.py --demo --beam-search --verbose
```

**❌ "Found 0 unique sentences"**
- Kiểm tra beam search có tìm được paths không
- Tăng beam_width và max_depth
- Verify graph có sentence nodes không

**❌ File saved ở wrong location**
- Scripts tự động tìm đúng location
- Check `output/`, `vncorenlp/output/`, `/home/hoadm/FactChecking/output/`

---

🎉 **Sentence extraction hoàn chỉnh!** Giờ bạn có danh sách các câu được xếp hạng theo mức độ liên quan đến claim rồi! 