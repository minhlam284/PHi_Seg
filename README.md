# PHi_Seg

Đây là cách chạy Phi_Seg chuẩn nhất thế giới, nếu bạn đau đầu với đống thư mục trong unet_zoo để dùng phiseg, thì repo này là dành cho bạn 

**PHi_Seg** là một mô hình phân đoạn ảnh y tế dựa trên kiến trúc học sâu, được phát triển bằng PyTorch. Dự án này nhằm mục đích cải thiện độ chính xác và độ tin cậy trong phân đoạn ảnh y tế, với khả năng xử lý các dữ liệu phức tạp và không chắc chắn.

## 🚀 Tính năng nổi bật

- **Kiến trúc mô hình hiện đại**: Sử dụng các lớp học sâu tiên tiến để xử lý và phân đoạn ảnh y tế.
- **Hỗ trợ huấn luyện và đánh giá**: Cung cấp các script để huấn luyện mô hình (`train.py`) và đánh giá hiệu suất (`eval_mmis.py`).
- **Cấu hình linh hoạt**: Dễ dàng tùy chỉnh các tham số huấn luyện và cấu hình mô hình thông qua thư mục `config/`.
- **Tiện ích hỗ trợ**: Bao gồm các tiện ích trong `utils.py` để hỗ trợ xử lý dữ liệu và các tác vụ phụ trợ khác.

## 📁 Cấu trúc thư mục

```
PHi_Seg/
├── config/             # Cấu hình mô hình và huấn luyện
│   └── *.yaml          # Các tệp cấu hình YAML
├── data/               # Dữ liệu đầu vào và xử lý dữ liệu
│   ├── raw/            # Dữ liệu gốc chưa xử lý
│   └── processed/      # Dữ liệu đã xử lý
├── log/                # Lưu trữ log huấn luyện và kết quả
├── model/              # Định nghĩa kiến trúc mô hình
│   ├── __init__.py     # Khởi tạo module
│   └── *.py            # Các tệp định nghĩa mô hình
├── utils/              # Các hàm tiện ích
│   └── utils.py        # Tệp chứa các hàm hỗ trợ
├── train.py            # Script huấn luyện chính
├── eval_mmis.py        # Script đánh giá mô hình
├── requirements.txt    # Danh sách thư viện phụ thuộc
└── README.md           # Tệp hướng dẫn sử dụng
```

## 🛠️ Cài đặt

1. **Clone repository**:

   ```bash
   git clone https://github.com/minhlam284/PHi_Seg.git
   cd PHi_Seg
   pip install -r requirements.txt

## 🧪 Cách dùng
1. **Train**:
    ```bash
    python train.py path/to/your/experiment.py
2. **Eval**:
    ```bash
    python eval_mmis.py
