import os, tempfile
import cv2
import gradio as gr
from swapper import VideoFaceSwapper

MAX_SECONDS = int(os.environ.get("MAX_SECONDS", "60"))
swapper = VideoFaceSwapper()


def process(video, image, restore, restore_weight, smooth, face_idx, consent, progress=gr.Progress()):
    if not consent:
        raise gr.Error("Vui lòng xác nhận bạn có quyền sử dụng khuôn mặt trong ảnh/video.")
    if video is None or image is None:
        raise gr.Error("Cần upload cả video và ảnh mặt.")
    out = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False).name
    try:
        return swapper.run(video, image, out, restore=restore, restore_weight=restore_weight,
                           smooth=smooth, target_index=int(face_idx), max_seconds=MAX_SECONDS,
                           progress=lambda p, d="": progress(p, desc=d))
    except ValueError as e:
        raise gr.Error(str(e))


with gr.Blocks(title="Face Swap Video - Matech UA") as demo:
    gr.Markdown(f"# Face Swap Video\nUpload video + ảnh mặt → video với mặt mới, giữ nguyên biểu cảm/chuyển động gốc. (Tối đa {MAX_SECONDS}s)")
    with gr.Row():
        vid = gr.Video(label="Video gốc")
        img = gr.Image(type="filepath", label="Ảnh mặt (rõ, nhìn thẳng, đủ sáng)")
    with gr.Accordion("Tuỳ chọn", open=False):
        restore = gr.Checkbox(True, label="Làm nét mặt (GFPGAN)")
        rw = gr.Slider(0, 1, 0.5, step=0.05, label="Cường độ làm nét (thấp = giống ảnh gốc hơn)")
        sm = gr.Slider(0, 0.9, 0.6, step=0.05, label="Làm mượt chống rung")
        idx = gr.Number(0, precision=0, label="Mặt thứ mấy trong video (0 = trái nhất)")
    consent = gr.Checkbox(False, label="Tôi có quyền/đồng ý của người trong ảnh & video, không dùng để lừa đảo/mạo danh.")
    btn = gr.Button("Swap", variant="primary")
    out = gr.Video(label="Kết quả")
    btn.click(process, [vid, img, restore, rw, sm, idx, consent], out)

if __name__ == "__main__":
    demo.queue(max_size=5).launch()
