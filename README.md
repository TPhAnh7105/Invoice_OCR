# Invoice OCR

Hệ thống đọc hóa đơn tiếng Việt, sử dụng PAN để phát hiện vùng chữ, VietOCR để nhận dạng chữ và PhoBERT để phân loại thông tin.

Demo hỗ trợ tải ảnh hóa đơn, trích xuất tên cửa hàng, địa chỉ, thời gian và tổng tiền; chỉnh sửa kết quả và xuất Excel.

## Cấu trúc thư mục

- `data/`: dữ liệu ảnh, nhãn, CSV và từ điển.
- `invoice_demo/`: backend FastAPI, frontend và mã mô hình.
- `notebooks/`: notebook thử nghiệm và đánh giá.
- `configs/`: cấu hình mô hình.
- `weights/`: trọng số mô hình, lưu bằng Git LFS.
- `metrics/`: kết quả đánh giá.
- `tools/`: công cụ hỗ trợ.
- `tests/`: kiểm thử ứng dụng.

## Cài đặt và chạy

Cần Python 3.12 và Git LFS. Chạy các lệnh sau trong PowerShell:

```powershell
git lfs install
git clone https://github.com/TPhAnh7105/Invoice_OCR.git
cd Invoice_OCR
git lfs pull
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements-demo.txt
$env:OCR_ALLOW_DOWNLOAD = "1"
.venv\Scripts\python.exe run_demo.py
```

Mở **http://127.0.0.1:8000** để sử dụng. Lần đầu cần Internet để tải cấu hình và tokenizer PhoBERT. Hệ thống tự dùng GPU CUDA nếu có, hoặc CPU.

Các lần sau có thể chạy `start_demo.cmd`. Kết quả và lịch sử xử lý được lưu trong `output/fastapi_demo/`.

## Notebook

Mở `notebooks/invoice-ocr-with-metrics.ipynb` bằng Jupyter hoặc VS Code và chọn môi trường Python đã cài các thư viện cần thiết. Notebook đọc dữ liệu từ `data/`.
