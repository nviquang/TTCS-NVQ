"""
demo.py — Vietnamese Image Captioning Demo
==========================================
Sử dụng BLIP fine-tuned (saved_models_v3) để sinh caption tiếng Việt từ webcam.
VieNeu-TTS (pip install vieneu) đọc caption bằng giọng nói.

Điều khiển:
  Space  — chụp frame hiện tại, sinh caption + đọc to
  C      — chuyển sang webcam kế tiếp (vòng tròn)
  V      — chuyển qua/lại giữa webcam và video sample_video.mp4
  Q      — thoát
"""

import os
import sys
import threading
import time

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from transformers import BlipForConditionalGeneration, BlipProcessor

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(SCRIPT_DIR, "saved_models_v3")
FONT_PATH = os.path.join("C:\\Windows\\Fonts", "arial.ttf")
VIDEO_PATH = os.path.join(SCRIPT_DIR, "sample_video.mp4")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MAX_CAM_INDEX = 10
CAPTION_FONT_SIZE = 22
STATUS_FONT_SIZE = 18
CAPTION_MAX_BEAMS = 5
CAPTION_MAX_LENGTH = 64
CAPTION_NO_REPEAT_NGRAM = 3
CAPTION_REPETITION_PENALTY = 1.5

WINDOW_MAIN = "BLIP Vietnamese Captioning  [Space=chup | C=doi cam | V=video/cam | Q=thoat]"

STATUS_BAR_H = 40
PANEL_PADDING = 16


def _load_font(size: int) -> ImageFont.FreeTypeFont:
    """Tải font TrueType hỗ trợ tiếng Việt, fallback về font mặc định nếu không tìm thấy."""
    for candidate in [FONT_PATH, "C:\\Windows\\Fonts\\arial.ttf", "C:\\Windows\\Fonts\\tahoma.ttf"]:
        try:
            return ImageFont.truetype(candidate, size)
        except (OSError, IOError):
            continue
    return ImageFont.load_default()


def put_viet_text(
    img_bgr: np.ndarray,
    text: str,
    pos: tuple[int, int],
    font: ImageFont.FreeTypeFont,
    color: tuple[int, int, int] = (255, 255, 255),
    max_width: int | None = None,
) -> np.ndarray:
    """Vẽ text Unicode (tiếng Việt) lên ảnh BGR dùng PIL, trả về ảnh BGR mới."""
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img_rgb)
    draw = ImageDraw.Draw(pil_img)

    if max_width is None:
        max_width = pil_img.width - PANEL_PADDING * 2

    words = text.split()
    lines: list[str] = []
    while words:
        line = words.pop(0)
        while words:
            test = line + " " + words[0]
            bbox = draw.textbbox((0, 0), test, font=font)
            if bbox[2] - bbox[0] <= max_width:
                line = test
                words.pop(0)
            else:
                break
        lines.append(line)

    x, y = pos
    for line in lines:
        bbox = draw.textbbox((0, 0), line, font=font)
        line_h = bbox[3] - bbox[1]
        draw.text(
            (x, y),
            line,
            font=font,
            fill=color,
            stroke_width=2,
            stroke_fill=(0, 0, 0),
        )
        y += line_h + 6

    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)


def draw_status_bar(
    frame: np.ndarray,
    text: str,
    font: ImageFont.FreeTypeFont,
    color: tuple[int, int, int] = (0, 220, 0),
) -> np.ndarray:
    """Vẽ thanh trạng thái PIL (hỗ trợ tiếng Việt) ở dưới cùng khung hình."""
    h, w = frame.shape[:2]
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, h - STATUS_BAR_H), (w, h), (30, 30, 30), -1)
    cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)
    return put_viet_text(frame, text, (10, h - STATUS_BAR_H + 8), font, color=color, max_width=w - 20)


def make_blank_panel(h: int, w: int) -> np.ndarray:
    """Tạo panel trắng với hướng dẫn ban đầu."""
    panel = np.full((h, w, 3), 245, dtype=np.uint8)
    return panel


def render_result_panel(
    snapshot: np.ndarray,
    caption: str,
    panel_w: int,
    font_caption: ImageFont.FreeTypeFont,
    font_label: ImageFont.FreeTypeFont,
) -> np.ndarray:
    """Render panel bên phải: ảnh đã chụp (thu nhỏ vừa panel) + caption bên dưới."""
    panel_h = snapshot.shape[0]

    img_area_h = panel_h - 100
    img_aspect = snapshot.shape[1] / snapshot.shape[0]
    img_w = min(panel_w - PANEL_PADDING * 2, int(img_area_h * img_aspect))
    img_h = int(img_w / img_aspect)
    resized = cv2.resize(snapshot, (img_w, img_h))

    panel = np.full((panel_h, panel_w, 3), 245, dtype=np.uint8)

    x_off = (panel_w - img_w) // 2
    panel[PANEL_PADDING: PANEL_PADDING + img_h, x_off: x_off + img_w] = resized

    caption_y = PANEL_PADDING + img_h + 10
    if caption:
        panel = put_viet_text(
            panel,
            caption,
            (PANEL_PADDING, caption_y),
            font_caption,
            color=(30, 30, 30),
            max_width=panel_w - PANEL_PADDING * 2,
        )

    return panel


def find_available_cameras(max_index: int = MAX_CAM_INDEX) -> list[int]:
    print("Đang quét webcam...", end=" ", flush=True)
    available: list[int] = []
    for i in range(max_index):
        cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
        if cap.isOpened():
            ret, _ = cap.read()
            if ret:
                available.append(i)
        cap.release()
    print(f"Tìm thấy: {available if available else 'Không có'}")
    return available


def load_blip_model(model_dir: str, device: str):
    print(f"Đang tải model từ '{model_dir}' trên {device.upper()}...")
    processor = BlipProcessor.from_pretrained(model_dir)
    model = BlipForConditionalGeneration.from_pretrained(model_dir).to(device)
    model.eval()
    print("Model đã sẵn sàng.")
    return processor, model


def predict_caption(
    image_rgb: np.ndarray,
    processor: BlipProcessor,
    model: BlipForConditionalGeneration,
    device: str,
) -> str:
    inputs = processor(images=image_rgb, return_tensors="pt").to(device)
    with torch.no_grad():
        generated_ids = model.generate(
            pixel_values=inputs["pixel_values"],
            max_length=CAPTION_MAX_LENGTH,
            num_beams=CAPTION_MAX_BEAMS,
            early_stopping=True,
            no_repeat_ngram_size=CAPTION_NO_REPEAT_NGRAM,
            repetition_penalty=CAPTION_REPETITION_PENALTY,
        )
    caption = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return caption.strip()


_tts_engine = None
_tts_lock = threading.Lock()


def get_tts_engine():
    global _tts_engine
    if _tts_engine is not None:
        return _tts_engine
    with _tts_lock:
        if _tts_engine is None:
            try:
                from vieneu import Vieneu
                import pygame
                print("Đang tải VieNeu-TTS...")
                _tts_engine = Vieneu(emotion="natural")
                print("VieNeu-TTS đã sẵn sàng.")
                pygame.mixer.init()
            except ImportError:
                print(
                    "[WARN] Không tìm thấy vieneu. Cài đặt bằng: pip install vieneu\n"
                    "       Tính năng đọc caption sẽ bị tắt."
                )
                _tts_engine = None
    return _tts_engine


def speak_caption(caption: str) -> None:
    """Tổng hợp giọng nói và phát ngầm trên một thread riêng."""

    def _run() -> None:
        tts = get_tts_engine()
        if tts is None:
            return
        try:
            import tempfile
            import pygame
            audio = tts.infer(text=caption)
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                tmp_path = f.name
            tts.save(audio, tmp_path)
            try:
                pygame.mixer.music.load(tmp_path)
                pygame.mixer.music.play()
                while pygame.mixer.music.get_busy():
                    time.sleep(0.05)
                pygame.mixer.music.unload()
            finally:
                if os.path.exists(tmp_path):
                    os.unlink(tmp_path)
        except Exception as exc:
            print(f"[TTS ERROR] {exc}")

    threading.Thread(target=_run, daemon=True).start()


def main() -> None:
    cam_indices = find_available_cameras()
    if not cam_indices:
        print("Không tìm thấy webcam nào. Thoát.")
        sys.exit(1)

    if not os.path.isdir(MODEL_DIR):
        print(f"[ERROR] Không tìm thấy model tại '{MODEL_DIR}'.")
        sys.exit(1)

    processor, model = load_blip_model(MODEL_DIR, DEVICE)

    font_caption = _load_font(CAPTION_FONT_SIZE)
    font_status = _load_font(STATUS_FONT_SIZE)

    threading.Thread(target=get_tts_engine, daemon=True).start()

    cam_idx_ptr = 0
    cap = cv2.VideoCapture(cam_indices[cam_idx_ptr], cv2.CAP_DSHOW)
    use_video: bool = False

    # Đọc frame đầu tiên để lấy kích thước thực tế
    ret0, frame0 = cap.read()
    current_h, current_w = (frame0.shape[:2] if ret0 else (480, 640))

    current_caption: str = ""
    is_processing: bool = False
    result_panel: np.ndarray = make_blank_panel(current_h, current_w)

    print(
        f"\nSử dụng webcam #{cam_indices[cam_idx_ptr]}  "
        f"(tổng {len(cam_indices)} cam: {cam_indices})\n"
        "  Space = chụp & sinh caption\n"
        "  C     = đổi cam\n"
        "  V     = chuyển qua/lại video sample_video.mp4\n"
        "  Q     = thoát\n"
    )

    def on_capture(snapshot: np.ndarray) -> None:
        nonlocal current_caption, is_processing, result_panel
        try:
            rgb = cv2.cvtColor(snapshot, cv2.COLOR_BGR2RGB)
            caption = predict_caption(rgb, processor, model, DEVICE)
            current_caption = caption
            print(f"\n[Caption] {caption}")
            result_panel = render_result_panel(snapshot, caption, snapshot.shape[1], font_caption, font_status)
            speak_caption(caption)
        finally:
            is_processing = False

    def trigger_capture(frame: np.ndarray) -> None:
        nonlocal is_processing
        if is_processing:
            print("[INFO] Đang xử lý, vui lòng chờ...")
            return
        is_processing = True
        snapshot = frame.copy()
        threading.Thread(target=on_capture, args=(snapshot,), daemon=True).start()

    while True:
        ret, frame = cap.read()
        if not ret:
            if use_video:
                # Video hết → loop lại từ đầu
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
            print("[WARN] Không đọc được frame. Thử lại...")
            time.sleep(0.05)
            continue

        if use_video:
            # Resize frame video về đúng kích thước cam, giữ tỉ lệ (letterbox)
            fh, fw = frame.shape[:2]
            scale = min(current_w / fw, current_h / fh)
            new_fw, new_fh = int(fw * scale), int(fh * scale)
            resized = cv2.resize(frame, (new_fw, new_fh), interpolation=cv2.INTER_AREA)
            frame = np.zeros((current_h, current_w, 3), dtype=np.uint8)
            x_off = (current_w - new_fw) // 2
            y_off = (current_h - new_fh) // 2
            frame[y_off:y_off + new_fh, x_off:x_off + new_fw] = resized
        else:
            # Lấy kích thước thực tế của frame cam
            h, w = frame.shape[:2]
            # Nếu kích thước thay đổi (đổi cam), cập nhật và tạo lại result_panel
            if h != current_h or w != current_w:
                print(f"[INFO] Kích thước frame thay đổi: ({current_w}x{current_h}) → ({w}x{h})")
                current_h, current_w = h, w
                result_panel = make_blank_panel(current_h, current_w)

        cam_frame = frame.copy()

        if use_video:
            source_label = "Video: sample_video.mp4"
        else:
            source_label = f"Cam #{cam_indices[cam_idx_ptr]}  ({cam_idx_ptr + 1}/{len(cam_indices)})"

        if is_processing:
            status_text = f"{source_label}  |  Đang sinh caption..."
            status_color = (0, 200, 255)
        else:
            status_text = f"{source_label}  |  Space=chụp  C=đổi cam  V=video/cam  Q=thoát"
            status_color = (0, 220, 0)

        cam_frame = draw_status_bar(cam_frame, status_text, font_status, color=status_color)

        panel_display = cv2.resize(result_panel, (current_w, current_h))
        composite = np.hstack([cam_frame, panel_display])

        cv2.imshow(WINDOW_MAIN, composite)

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q") or key == 27:
            break
        elif key == ord(" "):
            trigger_capture(frame)
        elif key == ord("c") or key == ord("C"):
            if use_video:
                print("[INFO] Đang ở chế độ video. Nhấn V để quay về cam trước khi đổi cam.")
            else:
                cap.release()
                prev_cam_idx_ptr = cam_idx_ptr
                opened = False
                for attempt in range(1, len(cam_indices)):
                    cam_idx_ptr = (prev_cam_idx_ptr + attempt) % len(cam_indices)
                    new_idx = cam_indices[cam_idx_ptr]
                    print(f"[INFO] Chuyển sang webcam #{new_idx}")
                    cap = cv2.VideoCapture(new_idx, cv2.CAP_DSHOW)
                    if cap.isOpened():
                        opened = True
                        break
                    print(f"[WARN] Không mở được cam #{new_idx}, thử cam tiếp theo...")
                    cap.release()
                if not opened:
                    cam_idx_ptr = prev_cam_idx_ptr
                    old_idx = cam_indices[cam_idx_ptr]
                    print(f"[ERROR] Tất cả cam đều thất bại. Quay lại webcam #{old_idx}.")
                    cap = cv2.VideoCapture(old_idx, cv2.CAP_DSHOW)
                current_caption = ""
                result_panel = make_blank_panel(current_h, current_w)
        elif key == ord("v") or key == ord("V"):
            if not use_video:
                try:
                    if not os.path.isfile(VIDEO_PATH):
                        raise FileNotFoundError(f"Không tìm thấy file video tại '{VIDEO_PATH}'")
                    new_cap = cv2.VideoCapture(VIDEO_PATH)
                    if not new_cap.isOpened():
                        new_cap.release()
                        raise IOError(f"OpenCV không mở được video '{VIDEO_PATH}'")
                    cap.release()
                    cap = new_cap
                    use_video = True
                    print(f"[INFO] Chuyển sang video: {VIDEO_PATH}")
                    current_caption = ""
                    result_panel = make_blank_panel(current_h, current_w)
                except Exception as exc:
                    print(f"[ERROR] Không thể mở video: {exc}")
                    print(f"[INFO] Giữ nguyên webcam #{cam_indices[cam_idx_ptr]}.")
            else:
                try:
                    new_cap = cv2.VideoCapture(cam_indices[cam_idx_ptr], cv2.CAP_DSHOW)
                    if not new_cap.isOpened():
                        new_cap.release()
                        raise IOError(f"Không mở được webcam #{cam_indices[cam_idx_ptr]}")
                    cap.release()
                    cap = new_cap
                    use_video = False
                    print(f"[INFO] Quay lại webcam #{cam_indices[cam_idx_ptr]}")
                    current_caption = ""
                    result_panel = make_blank_panel(current_h, current_w)
                except Exception as exc:
                    print(f"[ERROR] Không thể quay lại webcam: {exc}")
                    print("[INFO] Giữ nguyên chế độ video.")

    global _tts_engine
    if _tts_engine is not None:
        try:
            _tts_engine.close()
        except Exception as exc:
            print(f"[WARN] TTS cleanup: {exc}")
        finally:
            _tts_engine = None
        try:
            import pygame
            pygame.mixer.quit()
        except Exception as exc:
            print(f"[WARN] pygame.mixer.quit: {exc}")

    cap.release()
    cv2.destroyAllWindows()
    print("Đã thoát.")


if __name__ == "__main__":
    main()
