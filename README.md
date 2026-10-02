---
title: Face Swap Video
sdk: gradio
sdk_version: 4.44.0
app_file: app.py
---
# Face Swap Video (Matech UA)

Input: video + ảnh mặt → Output: video với mặt từ ảnh, giữ biểu cảm, góc mặt, lip-sync, audio gốc.

**Pipeline:** InsightFace `buffalo_l` (detect + embedding) → `inswapper_128` (swap từng frame) → làm mượt landmark theo thời gian + bám mặt bằng IoU → GFPGAN (tuỳ chọn) → ffmpeg ghép audio.

## Dùng online (không cần tải về)
1. Push repo lên GitHub.
2. Tạo Hugging Face Space (SDK Gradio) → chọn phần cứng GPU → Space sẽ là link công khai để team UA dùng. Có thể đồng bộ GitHub → Space bằng GitHub Action (`huggingface-cli upload` / git push mirror).

## Chạy local
```
pip install -r requirements.txt   # cần ffmpeg
python app.py
```
Biến môi trường: `MAX_SECONDS` (mặc định 60), `MODELS_DIR`.

## Lưu ý
- **License:** weights `inswapper_128` của InsightFace chỉ cho nghiên cứu/phi thương mại. Dùng cho quảng cáo thương mại cần xin license hoặc thay model.
- Chỉ dùng với khuôn mặt bạn có quyền sử dụng; gắn nhãn nội dung AI khi chạy ads theo chính sách nền tảng.
