import cv2
import numpy as np
import random
import sys
import time
import os

# --- 💡 設定項目 ---
TOLERANCE = 50
CHANGE_INTERVAL_FRAMES = 30
TRANSITION_FRAMES = 15
BLUR_KERNEL = max(3, int(TOLERANCE // 8) * 2 + 1)
INPUT_VIDEO_FG = 'video1.mp4'
INPUT_VIDEO_BG = 'video2.mp4'
OUTPUT_VIDEO = 'final_video.mp4'

def rand_bgr():
    return np.array([random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)], dtype=np.uint8)

def loop_reset(cap):
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

def process_video():
    random.seed(time.time() + os.getpid())

    cap_fg = cv2.VideoCapture(INPUT_VIDEO_FG)
    cap_bg = cv2.VideoCapture(INPUT_VIDEO_BG)

    if not cap_fg.isOpened():
        print(f"エラー: {INPUT_VIDEO_FG} を開けませんでした。")
        return
    if not cap_bg.isOpened():
        print(f"エラー: {INPUT_VIDEO_BG} を開けませんでした。")
        return

    width = int(cap_fg.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap_fg.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap_fg.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap_fg.get(cv2.CAP_PROP_FRAME_COUNT))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(OUTPUT_VIDEO, fourcc, fps, (width, height))

    print("動画処理を開始します...")
    
    frame_count = 0
    current_color = rand_bgr().astype(np.float32)
    target_color = rand_bgr().astype(np.float32)
    transition_progress = 1.0

    while True:
        ret_fg, frame_fg = cap_fg.read()
        ret_bg, frame_bg = cap_bg.read()

        if not ret_fg:
            break
        if not ret_bg:
            loop_reset(cap_bg)
            ret_bg, frame_bg = cap_bg.read()
            if not ret_bg:
                print("背景ビデオの読み込みに失敗しました。")
                break

        frame_count += 1
        sys.stdout.write(f"\rフレーム処理中: {frame_count} / {total_frames}")
        sys.stdout.flush()

        frame_bg_resized = cv2.resize(frame_bg, (width, height))

        if frame_count % CHANGE_INTERVAL_FRAMES == 0:
            target_color = rand_bgr().astype(np.float32)
            transition_progress = 0.0

        if transition_progress < 1.0:
            transition_progress += 1.0 / max(1, TRANSITION_FRAMES)
            t = min(1.0, transition_progress)
            current_color = (1 - t) * current_color + t * target_color

        f = frame_fg.astype(np.int16)
        c = current_color.astype(np.int16)
        diff = f - c
        dist = np.linalg.norm(diff, axis=2).astype(np.float32)
        alpha = np.clip(1.0 - (dist / float(max(1, TOLERANCE))), 0.0, 1.0)
        alpha_blur = cv2.GaussianBlur(alpha, (BLUR_KERNEL, BLUR_KERNEL), 0)
        alpha_3 = alpha_blur[..., np.newaxis]
        fg_f = frame_fg.astype(np.float32)
        bg_f = frame_bg_resized.astype(np.float32)
        final_f = fg_f * (1.0 - alpha_3) + bg_f * alpha_3
        final_frame = np.clip(final_f, 0, 255).astype(np.uint8)

        out.write(final_frame)

    print(f"\n動画処理が完了しました。'{OUTPUT_VIDEO}' として保存されました。")

    cap_fg.release()
    cap_bg.release()
    out.release()
    try:
        cv2.destroyAllWindows()
    except Exception:
        pass

if __name__ == '__main__':
    process_video()
