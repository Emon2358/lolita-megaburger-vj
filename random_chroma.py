# random_chroma.py
# Simple color-key compositor:
# - Pick a random foreground frame each output frame
# - Make pixels close to a random target color transparent (replace by background)
# - No glitches / no extra noise — pure color transparency
# - Resize background to match foreground size automatically
#
# Usage: put video1.mp4 (foreground) and video2.mp4 (background) next to this script and run.

import cv2
import numpy as np
import random
import time
import os

# ---------- 設定 ----------
INPUT_VIDEO_FG = 'video1.mp4'
INPUT_VIDEO_BG = 'video2.mp4'
OUTPUT_VIDEO_RAW = 'final_video_raw.mp4'

# 色透過（チョーク）設定
TOLERANCE = 80                  # 色距離の閾値。小さくすると厳密に色が合う部分のみ透過
CHANGE_INTERVAL_FRAMES = 1      # ターゲット色を何フレームごとに切り替えるか（1 = 毎出力フレーム）
TRANSITION_FRAMES = 0           # 色の補間フレーム数（0 = 瞬変）
BLUR_KERNEL = 7                 # マスクをぼかすカーネル（奇数推奨。1だとぼかしなし）

# 出力制限（処理フレーム数を減らしたい場合に設定）
MAX_OUTPUT_SECONDS = 0          # 0 = 無制限。例: 10 で出力を10秒に制限
OUTPUT_FPS = 0                  # 0 = 元動画の fps を使用。例: 12 にすると出力fpsを下げる
MAX_OUTPUT_FRAMES = 0           # 0 = 無効。これが >0 の場合、優先してそのフレーム数で終了

# 前景フレームサンプリング（オンザフライ時のI/Oを減らす）
FG_SAMPLE_POOL_SIZE = 0         # 0 = 無制限（全フレームからランダム）。小さくするとI/O負荷低下（例: 200）

# プリロードしきい値（小さければ全部メモリに読み込む）
PRELOAD_FRAME_LIMIT = 700
MAX_PRELOAD_BYTES = 800 * 1024 * 1024  # 800MB

FOURCC = cv2.VideoWriter_fourcc(*'mp4v')
# ----------------------------

def rand_bgr():
    return np.array([random.randint(0,255), random.randint(0,255), random.randint(0,255)], dtype=np.uint8)

def loop_reset(cap):
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

def safe_read_frame_at(cap, idx):
    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
    return cap.read()

def ensure_odd(n):
    n = int(n)
    if n <= 1:
        return 1
    return n if n % 2 == 1 else n+1

def process_video():
    random.seed(time.time() + os.getpid())

    cap_fg = cv2.VideoCapture(INPUT_VIDEO_FG)
    cap_bg = cv2.VideoCapture(INPUT_VIDEO_BG)

    if not cap_fg.isOpened():
        print(f"ERROR: cannot open foreground '{INPUT_VIDEO_FG}'")
        return
    if not cap_bg.isOpened():
        print(f"ERROR: cannot open background '{INPUT_VIDEO_BG}'")
        return

    width = int(cap_fg.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap_fg.get(cv2.CAP_PROP_FRAME_HEIGHT))
    src_fps = cap_fg.get(cv2.CAP_PROP_FPS) or 30.0
    total_fg_frames = int(cap_fg.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    total_bg_frames = int(cap_bg.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    # 出力 fps 決定
    out_fps = float(OUTPUT_FPS) if OUTPUT_FPS and OUTPUT_FPS > 0 else float(src_fps)

    # MAX_OUTPUT_FRAMES の決定
    if MAX_OUTPUT_FRAMES and MAX_OUTPUT_FRAMES > 0:
        max_frames = int(MAX_OUTPUT_FRAMES)
    elif MAX_OUTPUT_SECONDS and MAX_OUTPUT_SECONDS > 0:
        max_frames = int(max(1, round(MAX_OUTPUT_SECONDS * out_fps)))
    else:
        max_frames = None  # 無制限

    print(f"FG frames: {total_fg_frames}, BG frames: {total_bg_frames}, {width}x{height} @ {src_fps}fps")
    if max_frames:
        print(f"Output limited to {max_frames} frames ({out_fps} fps -> ~{max_frames/out_fps:.2f}s)")

    out = cv2.VideoWriter(OUTPUT_VIDEO_RAW, FOURCC, out_fps, (width, height))

    # プリロード判定（小さい前景なら全部読み込む）
    preload = False
    estimated_bytes = width * height * 3 * max(1, total_fg_frames)
    if total_fg_frames > 0 and total_fg_frames <= PRELOAD_FRAME_LIMIT and estimated_bytes <= MAX_PRELOAD_BYTES:
        preload = True

    fg_frames = []
    fg_sample_indices = None
    if preload:
        print("プリロード: 前景フレームをメモリに読み込みます...")
        cap_fg.set(cv2.CAP_PROP_POS_FRAMES, 0)
        cnt = 0
        while True:
            ret, f = cap_fg.read()
            if not ret:
                break
            if (f.shape[1], f.shape[0]) != (width, height):
                f = cv2.resize(f, (width, height))
            fg_frames.append(f)
            cnt += 1
        print(f"プリロード完了: {cnt} frames")
        loop_reset(cap_bg)
    else:
        # サンプルプールを作る（オンザフライで全フレーム走査しないため）
        if FG_SAMPLE_POOL_SIZE and FG_SAMPLE_POOL_SIZE > 0 and total_fg_frames > 0:
            pool_size = min(total_fg_frames, FG_SAMPLE_POOL_SIZE)
            fg_sample_indices = random.sample(range(total_fg_frames), pool_size)
            print(f"FG sample pool: {pool_size} indices prepared")
        else:
            fg_sample_indices = None
            print("FG sample pool: none (random from full range)")

    # 初期カラーターゲット
    current_color = rand_bgr().astype(np.float32)
    target_color = current_color.copy()
    transition_progress = 1.0

    frame_count = 0
    while True:
        # 終了条件
        if max_frames is not None and frame_count >= max_frames:
            print(f"\nReached max output frames ({max_frames}). Stopping.")
            break

        ret_bg, frame_bg = cap_bg.read()
        if not ret_bg:
            loop_reset(cap_bg)
            ret_bg, frame_bg = cap_bg.read()
            if not ret_bg:
                print("背景ビデオの読み込みに失敗しました。")
                break

        # 背景を出力サイズにリサイズ（必ず合わせる）
        if (frame_bg.shape[1], frame_bg.shape[0]) != (width, height):
            frame_bg = cv2.resize(frame_bg, (width, height))

        frame_count += 1
        sys.stdout.write(f"\rProcessing frame {frame_count}")
        sys.stdout.flush()

        # 前景の選択
        if preload:
            if len(fg_frames) == 0:
                fg_frame = frame_bg.copy()
            else:
                fg_frame = fg_frames[random.randrange(len(fg_frames))]
        else:
            if total_fg_frames <= 0:
                fg_frame = frame_bg.copy()
            else:
                if fg_sample_indices:
                    idx = random.choice(fg_sample_indices)
                else:
                    idx = random.randrange(total_fg_frames)
                ret_fg, fg_frame = safe_read_frame_at(cap_fg, idx)
                if not ret_fg:
                    # フォールバック: シーケンシャル read
                    ret_fg2, fg_frame = cap_fg.read()
                    if not ret_fg2:
                        fg_frame = frame_bg.copy()
            if (fg_frame.shape[1], fg_frame.shape[0]) != (width, height):
                fg_frame = cv2.resize(fg_frame, (width, height))

        # ターゲットカラーの切り替え（瞬変 or 補間）
        if frame_count % max(1, CHANGE_INTERVAL_FRAMES) == 0:
            target_color = rand_bgr().astype(np.float32)
            transition_progress = 0.0

        if transition_progress < 1.0 and TRANSITION_FRAMES > 0:
            transition_progress += 1.0 / max(1, TRANSITION_FRAMES)
            t = min(1.0, transition_progress)
            current_color = (1 - t) * current_color + t * target_color
        else:
            if TRANSITION_FRAMES == 0:
                current_color = target_color.copy()

        # マスク作成：色距離に基づくソフトマスク（ターゲット色に近いほど 1 -> 背景を採用）
        f = fg_frame.astype(np.float32)
        c = current_color.reshape((1,1,3)).astype(np.float32)
        diff = f - c
        dist = np.linalg.norm(diff, axis=2).astype(np.float32)
        tol = max(1.0, float(TOLERANCE))
        mask = 1.0 - (dist / tol)
        mask = np.clip(mask, 0.0, 1.0)

        # マスクのぼかし（エッジを滑らかに）
        bk = ensure_odd(BLUR_KERNEL)
        if bk > 1:
            mask = cv2.GaussianBlur(mask, (bk, bk), 0)

        alpha_3 = mask[..., np.newaxis]  # shape (h,w,1), 1 => background

        bg_f = frame_bg.astype(np.float32)

        # 合成（mask=1なら背景、0なら前景）
        final = f * (1.0 - alpha_3) + bg_f * alpha_3
        final = np.clip(final, 0, 255).astype(np.uint8)

        out.write(final)

    print(f"\nFinished. Saved '{OUTPUT_VIDEO_RAW}'")

    cap_fg.release()
    cap_bg.release()
    out.release()
    try:
        cv2.destroyAllWindows()
    except Exception:
        pass

if __name__ == '__main__':
    process_video()
