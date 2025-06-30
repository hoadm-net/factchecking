# 🎯 MINT TextGraph - Hướng dẫn sử dụng Beam Search Analysis

Hướng dẫn chi tiết về việc sử dụng các công cụ phân tích kết quả Beam Search trong MINT TextGraph để đánh giá hiệu quả fact-checking.

## 📋 Mục lục

1. [Tổng quan chức năng](#-tổng-quan-chức-năng)
2. [Chuẩn bị trước khi sử dụng](#-chuẩn-bị-trước-khi-sử-dụng)
3. [Hướng dẫn sử dụng từng công cụ](#-hướng-dẫn-sử-dụng-từng-công-cụ)
4. [Workflow khuyến nghị](#-workflow-khuyến-nghị)
5. [Hiểu kết quả phân tích](#-hiểu-kết-quả-phân-tích)
6. [Xử lý sự cố](#-xử-lý-sự-cố)

## 🔍 Tổng quan chức năng

### Hệ thống Beam Search Analysis gồm 2 công cụ chính:

#### 🔬 **1. analyze_beam_sentences.py** - Phân tích thống kê toàn diện
```
📊 Chức năng chính:
├── Phân tích performance tổng thể của beam search
├── Tạo biểu đồ trực quan về scores, lengths, success rates
├── Phân tích patterns trong đường đi (paths)
├── Tạo báo cáo comprehensive với insights
├── Xuất thống kê chi tiết dạng CSV
├── Xếp hạng câu quan trọng nhất cho từng sample
└── Tóm tắt top sentences với scoring advanced
```

**💡 Khi nào sử dụng:**
- Muốn đánh giá hiệu suất beam search trên toàn bộ dataset
- So sánh performance giữa các cấu hình khác nhau
- Tìm hiểu xu hướng và patterns trong kết quả
- Tạo báo cáo tổng quan cho nghiên cứu
- Xem câu nào quan trọng nhất cho fact-checking
- Cần kết quả cụ thể để kiểm tra manual

#### 🗂️ **2. process_beam_samples.py** - Tổ chức dữ liệu theo claim
```
🔧 Chức năng chính:
├── Nhóm sentences theo claim tương ứng
├── Loại bỏ duplicate và sắp xếp
├── Tạo cấu trúc dữ liệu claim -> sentences
├── Validation và backup tự động
└── Tạo báo cáo processing detailed
```

**💡 Khi nào sử dụng:**
- Cần dữ liệu có cấu trúc để tích hợp hệ thống
- Muốn tra cứu nhanh sentences cho một claim
- Chuẩn bị dữ liệu cho analysis tools khác
- Export dữ liệu cho external applications

## 🛠️ Chuẩn bị trước khi sử dụng

### Bước 1: Kiểm tra môi trường
```bash
# Kiểm tra Python version (yêu cầu >= 3.7)
python --version

# Cài đặt dependencies cần thiết
pip install pandas matplotlib seaborn numpy
```

### Bước 2: Tạo dữ liệu beam search
```bash
# Chạy beam search cho một sample test
python main.py --demo --beam-search --beam-width 10 --beam-max-depth 6 --verbose

# Chạy beam search cho nhiều samples (ví dụ 0-99)
for i in {0..99}; do
    echo "🔄 Processing sample $i..."
    python main.py --idx $i --beam-search --beam-width 10 --beam-max-depth 6 --disable-visualization --quiet
done
```

### Bước 3: Kiểm tra dữ liệu
```bash
# Kiểm tra output directory
ls output/beam_search_*.json | wc -l  # Đếm số file beam search

# Kiểm tra một file mẫu
head -20 output/beam_search_0.json
```

## 📚 Hướng dẫn sử dụng từng công cụ

### 🔬 **Tool 1: Phân tích thống kê toàn diện**

#### Cách chạy:
```bash
python analyze_beam_sentences.py
```

#### Quá trình thực hiện:
```
🔍 Analyzing beam search results...
📊 Successfully loaded 100 beam search samples
📊 Analyzing individual samples...
   Processed 10/100 samples...
   Processed 20/100 samples...
   ...
💾 Saved detailed statistics to output/beam_search_detailed_stats.csv
📊 Creating visualizations...
📊 Saved output/beam_search_scores_analysis.png
📊 Saved output/beam_search_lengths_analysis.png
📊 Saved output/beam_search_rates_analysis.png
📊 Saved output/beam_search_coverage_analysis.png
📊 Saved output/beam_search_diversity_analysis.png
🔍 Analyzing path patterns...
💾 Saved path patterns to output/beam_search_patterns.json
📄 Saved comprehensive report to output/beam_search_comprehensive_report.txt
✅ Analysis completed successfully!
```

#### Files được tạo ra:
```
output/
├── beam_search_detailed_stats.csv           # 📊 Dữ liệu thống kê chi tiết
├── beam_search_scores_analysis.png          # 📈 Biểu đồ điểm số
├── beam_search_lengths_analysis.png         # 📏 Biểu đồ độ dài path
├── beam_search_rates_analysis.png           # 📊 Biểu đồ tỷ lệ thành công
├── beam_search_coverage_analysis.png        # 🎯 Biểu đồ độ phủ từ
├── beam_search_diversity_analysis.png       # 🌐 Biểu đồ đa dạng nodes
├── beam_search_patterns.json               # 🔍 Patterns phổ biến
├── beam_search_comprehensive_report.txt    # 📋 Báo cáo tổng hợp
└── beam_search_top_sentences.txt           # 📝 Top sentences từ tất cả samples
```

#### Ví dụ kết quả thống kê:
```
📈 SUMMARY STATISTICS:
                avg_score  avg_length  sentence_reach_rate  entity_visit_rate
count           100.000000  100.000000          100.000000         100.000000
mean              8.234567    4.567890            0.756000           0.234000
std               2.123456    1.234567            0.198765           0.156789
min               3.456789    2.000000            0.200000           0.000000
max              14.567890    8.000000            1.000000           0.600000

🏆 TOP 5 PERFORMING SAMPLES:
Sample 23: Score 14.568, Success Rate 100.0%
Sample 45: Score 13.234, Success Rate 95.0%
Sample 67: Score 12.890, Success Rate 90.0%
```

### 🗂️ **Tool 2: Tổ chức dữ liệu theo claim**

#### Cách chạy:
```bash
python process_beam_samples.py
```

#### Quá trình thực hiện:
```
🔄 Starting Beam Search Samples Processing...
📁 Found 100 beam search files to process
   Processed 10/100 files...
   Processed 20/100 files...
   ...
✅ Processing completed:
   Successfully processed: 100 files
   Errors encountered: 0 files
   Unique claims found: 98

📊 PROCESSING SUMMARY:
========================================
📋 Total unique claims: 98
📝 Claims with sentences: 95
📄 Total sentences across all claims: 2,156
📊 Average sentences per claim: 22.0
📈 Maximum sentences in a single claim: 45

🔍 SAMPLE PREVIEW:
----------------------------------------
Sample 1:
Claim: Phó Thủ tướng Trần Hồng Hà thay mặt Chính phủ, Thủ tướng Chính phủ chúc mừng...
Sentences (7):
  1. Thay_mặt Chính_phủ , Thủ_tướng Chính_phủ , Phó Thủ_tướng Trần_Hồng_Hà chúc_mừng...
  2. Sau thời_gian gián_đoạn do đại_dịch COVID-19 , Liên_hoan truyền_hình toàn_quốc...
  ... and 5 more sentences

📁 Created backup: beam_samples_organized_backup.json
✅ Saved organized samples to beam_samples_organized.json
📊 Saved processing report to beam_samples_organized_report.txt
✅ Processing completed successfully!
```

#### Files được tạo ra:
```
├── beam_samples_organized.json             # 🗂️ Dữ liệu tổ chức chính
├── beam_samples_organized_report.txt       # 📊 Báo cáo validation  
└── beam_samples_organized_backup.json      # 💾 Backup tự động
```

#### Cấu trúc dữ liệu output:
```json
{
  "Claim text đầy đủ ở đây...": [
    "Câu 1 liên quan đến claim này...",
    "Câu 2 liên quan đến claim này...",
    "Câu 3 liên quan đến claim này..."
  ],
  "Claim khác ở đây...": [
    "Câu A liên quan đến claim khác...",
    "Câu B liên quan đến claim khác..."
  ]
}
```

## 🔄 Workflow khuyến nghị

### **Workflow cơ bản (cho người mới bắt đầu):**

```bash
# Bước 1: Chạy beam search cho vài samples test
python main.py --idx 0 --beam-search --verbose
python main.py --idx 1 --beam-search --verbose
python main.py --idx 2 --beam-search --verbose

# Bước 2: Kiểm tra kết quả có hợp lý không
python analyze_beam_sentences.py

# Bước 3: Nếu OK, chạy batch lớn hơn
for i in {0..19}; do
    python main.py --idx $i --beam-search --disable-visualization --quiet
done

# Bước 4: Phân tích toàn diện và tổ chức dữ liệu
python analyze_beam_sentences.py
python process_beam_samples.py
```

### **Workflow nâng cao (cho nghiên cứu):**

```bash
# Bước 1: Chạy beam search với nhiều cấu hình
mkdir -p experiments/config1 experiments/config2

# Config 1: Beam width 10
for i in {0..99}; do
    python main.py --idx $i --beam-search --beam-width 10 --beam-max-depth 6 --disable-visualization --quiet
done
mv output/* experiments/config1/

# Config 2: Beam width 20  
for i in {0..99}; do
    python main.py --idx $i --beam-search --beam-width 20 --beam-max-depth 6 --disable-visualization --quiet
done
mv output/* experiments/config2/

# Bước 2: So sánh kết quả
cd experiments/config1 && python ../../analyze_beam_sentences.py
cd ../config2 && python ../../analyze_beam_sentences.py

# Bước 3: Tạo báo cáo so sánh
python compare_experiments.py config1 config2
```

### **Workflow cho production:**

```bash
# Bước 1: Chuẩn bị environment
mkdir -p production_run/$(date +%Y%m%d_%H%M%S)
cd production_run/$(date +%Y%m%d_%H%M%S)

# Bước 2: Chạy full dataset
python batch_process.py --samples 0-999 --beam-width 15 --beam-max-depth 8

# Bước 3: Quality check
python analyze_beam_sentences.py
python validate_results.py

# Bước 4: Export final results
python process_beam_samples.py
python export_for_annotation.py
```

## 📊 Hiểu kết quả phân tích

### **Các metrics quan trọng:**

#### 1. **Score (Điểm số path)**
```
🎯 Ý nghĩa: Độ liên quan của path với claim
📊 Giá trị tốt: > 8.0
⚠️ Cần chú ý: < 5.0
💡 Cách cải thiện: Tăng similarity threshold, tune scoring weights
```

#### 2. **Sentence Reach Rate (Tỷ lệ đến sentence)**
```
🎯 Ý nghĩa: % paths thành công tìm được evidence sentences
📊 Giá trị tốt: > 80%
⚠️ Cần chú ý: < 60%  
💡 Cách cải thiện: Tăng beam width, max depth
```

#### 3. **Entity Visit Rate (Tỷ lệ đi qua entity)**
```
🎯 Ý nghĩa: % paths đi qua các entity quan trọng
📊 Giá trị tốt: > 40%
⚠️ Cần chú ý: < 20%
💡 Cách cải thiện: Cải thiện entity extraction, entity linking
```

#### 4. **Word Coverage (Độ phủ từ)**
```
🎯 Ý nghĩa: % từ trong claim được match trong paths
📊 Giá trị tốt: > 70%
⚠️ Cần chú ý: < 50%
💡 Cách cải thiện: Cải thiện semantic similarity, POS filtering
```

### **Đọc hiểu patterns:**

#### Pattern phổ biến:
```
C->W->S (SUCCESS): Claim -> Word -> Sentence = Tốt nhất
C->W->E->S (SUCCESS): Qua entity = Rất tốt  
C->W->W->S (SUCCESS): Nhiều word hops = OK
C->W->W (PARTIAL): Không đến sentence = Cần cải thiện
```

#### Ví dụ giải thích:
```
Pattern: C->W->E->W->S (SUCCESS) +ENTITY
Ý nghĩa: 
- Bắt đầu từ claim (C)
- Đi qua word node (W) 
- Đi qua entity quan trọng (E)
- Đi qua word khác (W)
- Đến được sentence evidence (S)
- Đây là path lý tưởng!
```

### **Phân tích câu được rank:**

#### Cách đọc sentence ranking:
```
1. sentence_2 (visited 8x)
   Score: 20.1000
   Text: Câu evidence chất lượng cao...

Giải thích:
- sentence_2: ID của câu trong graph
- visited 8x: Xuất hiện trong 8 paths khác nhau
- Score 20.1: Điểm tổng hợp (cao = tốt)
- Text: Nội dung câu để human review
```

#### Ngưỡng đánh giá:
```
🔥 Score > 15: Evidence rất mạnh, gần như chắc chắn liên quan
✅ Score 10-15: Evidence tốt, có khả năng cao liên quan  
🤔 Score 5-10: Evidence trung bình, cần kiểm tra manual
⚠️ Score < 5: Evidence yếu, có thể không liên quan
```

## 🚨 Xử lý sự cố

### **Lỗi thường gặp và cách khắc phục:**

#### 1. **Không tìm thấy beam search files**
```bash
⚠️ No beam search results found. Make sure you have run beam search first.
```
**Nguyên nhân:** Chưa chạy beam search hoặc files bị xóa
**Giải pháp:**
```bash
# Kiểm tra thư mục output
ls -la output/beam_search_*.json

# Nếu không có file, chạy lại beam search
python main.py --demo --beam-search --verbose

# Kiểm tra lại
ls -la output/beam_search_*.json
```

#### 2. **Files bị corrupt hoặc không đọc được**
```bash
❌ Error loading beam_search_5.json: Invalid JSON format
```
**Nguyên nhân:** File bị hỏng do gián đoạn process hoặc đầy đĩa
**Giải pháp:**
```bash
# Kiểm tra dung lượng đĩa
df -h

# Kiểm tra file bị hỏng
python -m json.tool output/beam_search_5.json

# Xóa file hỏng và chạy lại
rm output/beam_search_5.json
python main.py --idx 5 --beam-search --verbose
```

#### 3. **Thiếu dependencies**
```bash
ModuleNotFoundError: No module named 'matplotlib'
```
**Giải pháp:**
```bash
# Cài đặt đầy đủ dependencies
pip install -r requirements.txt

# Hoặc cài từng package
pip install pandas matplotlib seaborn numpy

# Kiểm tra cài đặt
python -c "import pandas, matplotlib, seaborn, numpy; print('All dependencies OK')"
```

#### 4. **Memory error với dataset lớn**
```bash
MemoryError: Unable to allocate array
```
**Giải pháp:**
```bash
# Giảm batch size, xử lý từng phần
python analyze_beam_sentences.py --batch-size 50

# Hoặc filter chỉ files gần đây
python analyze_beam_sentences.py --recent-only 100

# Tăng virtual memory (Linux)
sudo sysctl vm.overcommit_memory=1
```

#### 5. **Permission errors**
```bash
❌ Error saving file: Permission denied
```
**Giải pháp:**
```bash
# Kiểm tra quyền thư mục
ls -la output/

# Tạo lại thư mục với quyền đúng
rm -rf output/
mkdir output
chmod 755 output/

# Hoặc thay đổi owner
sudo chown -R $(whoami):$(whoami) output/
```

### **Debug và troubleshooting:**

#### Kiểm tra chất lượng dữ liệu:
```bash
# Đếm số files
find output/ -name "beam_search_*.json" | wc -l

# Kiểm tra size files
du -sh output/beam_search_*.json | sort -h

# Kiểm tra files trống
find output/ -name "beam_search_*.json" -size 0

# Validate JSON format
for file in output/beam_search_*.json; do
    echo "Checking $file..."
    python -m json.tool "$file" > /dev/null || echo "❌ $file is invalid"
done
```

#### Chạy trong debug mode:
```bash
# Chạy với verbose output
python analyze_beam_sentences.py --verbose

# Chạy với logging
python analyze_beam_sentences.py --log-level DEBUG

# Test với sample nhỏ
python analyze_beam_sentences.py --test-mode --samples 5
```

### **Performance tuning:**

#### Cho datasets lớn:
```bash
# Sử dụng multiprocessing
python analyze_beam_sentences.py --workers 4

# Batch processing
python analyze_beam_sentences.py --batch-size 100

# Chỉ tạo summary, bỏ qua visualization
python analyze_beam_sentences.py --no-plots
```

#### Tối ưu memory:
```bash
# Xóa files tạm
rm -rf output/temp_*

# Compress old files
gzip output/beam_search_*.json

# Cleanup sau khi chạy
python cleanup_analysis.py
```

## 💡 Tips và best practices

### **Tối ưu hiệu suất:**
1. **Chạy beam search theo batch nhỏ** thay vì một lúc toàn bộ
2. **Backup thường xuyên** thư mục output
3. **Monitor disk space** khi chạy datasets lớn
4. **Sử dụng screen/tmux** cho long-running jobs

### **Đảm bảo chất lượng:**
1. **Luôn validate** kết quả với sample nhỏ trước
2. **So sánh kết quả** giữa các lần chạy
3. **Manual review** top sentences từ summarize tool
4. **Document parameters** đã sử dụng cho reproducibility

### **Integration với tools khác:**
```bash
# Export sang Excel
python export_to_excel.py output/beam_search_detailed_stats.csv

# Tích hợp với Jupyter Notebook
jupyter notebook analysis_dashboard.ipynb

# API endpoint
python flask_api.py --port 5000
```

---

🎉 **Chúc bạn phân tích hiệu quả!** Nếu có thắc mắc, hãy kiểm tra log output và error messages để debug. 