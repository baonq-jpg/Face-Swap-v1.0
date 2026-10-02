"""Video face swap pipeline: InsightFace (detect+embed) + inswapper_128 (swap) + GFPGAN (optional restore)."""
import os, shutil, subprocess, tempfile
import cv2
import numpy as np
import onnxruntime as ort
import insightface
from insightface.app import FaceAnalysis
from huggingface_hub import hf_hub_download

MODELS_DIR = os.environ.get("MODELS_DIR", "models")
GFPGAN_URL = "https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.4.pth"


def _bbox_iou(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


class VideoFaceSwapper:
    def __init__(self):
        avail = ort.get_available_providers()
        providers = [p for p in ("CUDAExecutionProvider", "CPUExecutionProvider") if p in avail]
        self.analyser = FaceAnalysis(name="buffalo_l", root=MODELS_DIR, providers=providers)
        self.analyser.prepare(ctx_id=0, det_size=(640, 640))
        model_path = hf_hub_download("ezioruan/inswapper_128.onnx", "inswapper_128.onnx", cache_dir=MODELS_DIR)
        self.swapper = insightface.model_zoo.get_model(model_path, providers=providers)
        self._restorer = None

    # ---------- helpers ----------
    def _get_restorer(self):
        if self._restorer is None:
            from gfpgan import GFPGANer
            self._restorer = GFPGANer(model_path=GFPGAN_URL, upscale=1, arch="clean",
                                      channel_multiplier=2, bg_upsampler=None)
        return self._restorer

    def source_face(self, image_bgr):
        faces = self.analyser.get(image_bgr)
        if not faces:
            raise ValueError("Không tìm thấy mặt trong ảnh input.")
        return max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))

    # ---------- main ----------
    def run(self, video_path, image_path, out_path, restore=True, restore_weight=0.5,
            smooth=0.6, target_index=0, max_seconds=None, progress=lambda p, d="": None):
        src_img = cv2.imread(image_path)
        if src_img is None:
            raise ValueError("Không đọc được ảnh input.")
        src_face = self.source_face(src_img)

        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 25
        w, h = int(cap.get(3)), int(cap.get(4))
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
        if max_seconds:
            total = min(total, int(max_seconds * fps))

        tmpdir = tempfile.mkdtemp()
        raw = os.path.join(tmpdir, "raw.mp4")
        writer = cv2.VideoWriter(raw, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

        track_bbox, prev_kps, miss = None, None, 0
        for i in range(total):
            ok, frame = cap.read()
            if not ok:
                break
            faces = self.analyser.get(frame)
            target = None
            if faces:
                if track_bbox is None:   # frame đầu: chọn mặt theo thứ tự từ trái sang phải
                    faces = sorted(faces, key=lambda f: f.bbox[0])
                    target = faces[min(target_index, len(faces) - 1)]
                else:                    # các frame sau: bám mặt theo IoU
                    best = max(faces, key=lambda f: _bbox_iou(f.bbox, track_bbox))
                    if _bbox_iou(best.bbox, track_bbox) > 0.1:
                        target = best
            if target is not None:
                kps = target.kps.copy()
                if prev_kps is not None:  # làm mượt landmark -> chống rung/nhấp nháy
                    kps = smooth * prev_kps + (1 - smooth) * kps
                target.kps = kps
                prev_kps, track_bbox, miss = kps, target.bbox, 0
            elif prev_kps is not None and miss < 5:   # mất detect ngắn (nghiêng/che mặt): dùng lại landmark cũ
                from insightface.app.common import Face
                target = Face(bbox=track_bbox, kps=prev_kps, det_score=0.5)
                miss += 1

            if target is not None:
                out = self.swapper.get(frame, target, src_face, paste_back=True)
                if restore:
                    _, _, restored = self._get_restorer().enhance(out, has_aligned=False,
                                                                  only_center_face=False, paste_back=True)
                    out = cv2.addWeighted(restored, restore_weight, out, 1 - restore_weight, 0)
            else:
                out = frame
            writer.write(out)
            if i % 5 == 0:
                progress((i + 1) / total, f"Frame {i + 1}/{total}")
        cap.release(); writer.release()

        # ghép audio gốc + encode H.264
        cmd = ["ffmpeg", "-y", "-i", raw, "-i", video_path, "-map", "0:v:0", "-map", "1:a:0?",
               "-c:v", "libx264", "-crf", "17", "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest", out_path]
        subprocess.run(cmd, check=True, capture_output=True)
        shutil.rmtree(tmpdir, ignore_errors=True)
        return out_path
