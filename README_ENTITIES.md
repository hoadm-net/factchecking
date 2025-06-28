# Tính năng Entity Extraction với OpenAI

## Mô tả

TextGraph đã được cập nhật với tính năng trích xuất thực thể (Entity Extraction) sử dụng OpenAI GPT-4o-mini. Tính năng này cho phép:

- Trích xuất tự động các thực thể quan trọng từ context
- Thêm Entity nodes vào đồ thị
- Kết nối Entity nodes với Sentence nodes nếu thực thể được nhắc đến trong câu
- Hiển thị thống kê chi tiết về entities

## Cài đặt

### 1. Cài đặt dependencies mới

```bash
pip install -r requirements.txt
```

### 2. Cấu hình OpenAI API Key

1. Tạo file `.env` từ template:
```bash
cp .env.example .env
```

2. Sửa file `.env` và thay thế `your_openai_api_key_here` bằng API key thực của bạn:
```
OPENAI_API_KEY=sk-your-actual-openai-api-key-here
```

3. Đảm bảo file `.env` không được commit vào git (đã có trong .gitignore)

## Sử dụng

### Phương thức mới trong TextGraph

```python
# Khởi tạo TextGraph (sẽ tự động load .env)
text_graph = TextGraph()

# Xây dựng đồ thị cơ bản
text_graph.build_from_vncorenlp_output(context_sentences, claim, claim_sentences)

# Trích xuất và thêm entities
entity_nodes = text_graph.extract_and_add_entities(context, context_sentences)
```

### Các phương thức hỗ trợ

```python
# Chỉ trích xuất entities (không thêm vào graph)
entities = text_graph.extract_entities_with_openai(context_text)

# Thêm entity node riêng lẻ
entity_node = text_graph.add_entity_node("SAWACO", "ORGANIZATION")

# Kết nối entity với sentence
text_graph.connect_entity_to_sentence(entity_node, sentence_node)
```

## Loại thực thể được trích xuất

- **Tên người**: Các cá nhân được nhắc đến
- **Tên tổ chức/công ty**: SAWACO, PLO, v.v.
- **Địa điểm**: Quận 6, TP.HCM, Nhà máy nước Tân Hiệp, v.v.
- **Ngày tháng/thời gian**: 25-3, 22 giờ, v.v.
- **Số liệu quan trọng**: Thời gian, số lượng
- **Sản phẩm/dịch vụ**: Dịch vụ cấp nước
- **Sự kiện**: Bảo trì, cúp nước, v.v.

## Visualization

Đồ thị được cập nhật với:
- **Entity nodes**: Màu vàng (gold), kích thước trung bình
- **Entity edges**: Màu cam, kết nối Entity với Sentence
- Legend được cập nhật để hiển thị đầy đủ các loại nodes và edges

## Thống kê mới

```python
stats = text_graph.get_detailed_statistics()

# Thông tin về entities
print(f"Entity nodes: {stats['entity_nodes']}")
print(f"Entity edges: {stats['entity_structural_edges']}")
print(f"Unique entities: {stats['unique_entities']}")

# Danh sách chi tiết entities
for entity in stats['entities']:
    print(f"'{entity['name']}' - Kết nối với {entity['connected_sentences']} câu")
```

## Troubleshooting

### Lỗi "OpenAI client chưa được khởi tạo"

- Kiểm tra file `.env` đã được tạo và có đúng format
- Đảm bảo OPENAI_API_KEY hợp lệ
- Kiểm tra tài khoản OpenAI có đủ credit

### Lỗi API

- Kiểm tra kết nối internet
- Xem log lỗi để biết chi tiết
- Đảm bảo sử dụng model `gpt-4o-mini` (model có sẵn)

### Không trích xuất được entities

- Kiểm tra context có chứa thông tin thực thể không
- Thử chạy lại với text khác
- Xem response raw từ OpenAI trong log

## Demo

Chạy file `main.py` để xem demo tính năng:

```bash
python main.py
```

Output sẽ bao gồm:
- Trích xuất entities từ OpenAI
- Thống kê về entities
- Visualization với entity nodes
- Lưu graph bao gồm entities 