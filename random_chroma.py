# random_chroma.py
# Simple color-key compositor — reduced work, safe for CI
# - Limits output duration / frames
# - Lowers output FPS
# - Samples a small FG pool to avoid massive random seeks
# - Reads BG frames with stepping to match output FPS (reduces I/O)
# - Pure color-key only (no glitches)

import cv2
import numpy as np
import random
import time
import os
import sys

# ----------------- ユーザー設定（ここを調整） -----------------
INPUT_VIDEO_FG = 'video1.mp4'
INPUT_VIDEO_BG = 'video2.mp4'
OUTPUT_VIDEO_RAW = 'final_video_raw.mp4'

# Color key (chroma) settings
TOLERANCE = 80                  # 色距離の閾値（小さいほどピンポイント）
CHANGE_INTERVAL_FRAMES = 1      # ターゲット色を何出力フレームごとに切り替えるか
TRANSITION_FRAMES = 0           # 色の補間フレーム数（0 = 瞬変）
BLUR_KERNEL = 7                 # マスクぼかし（奇数推奨）

# --- 出力量を厳しく制限して処理フレームを最小にする設定（必ずどれか有効にして下さい） ---
MAX_OUTPUT_SECONDS = 0        # 出力を N 秒に制限（0 = 無制限）
OUTPUT_FPS = 8                 # 出力 FPS（0 = 元動画FPSを使用）
MAX_OUTPUT_FRAMES = 0          # 直接フレーム数上限。0 = 無効（存在するなら優先）

# 前景サンプル（オンザフライ時のI/O低減）
FG_SAMPLE_POOL_SIZE = 50       # 0 = 無制限（全フレーム）。小さくするとI/O激減（例: 50）

# プリロードしきい値（総フレームが少なければ全部プリロード）
PRELOAD_FRAME_LIMIT = 250
MAX_PRELOAD_BYTES = 400 * 1024 * 1024  # 400MB 上限目安

FOURCC = cv2.VideoWriter_fourcc(*'mp4v')
# --------------------------------------------------------------

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

    src_width = int(cap_fg.get(cv2.CAP_PROP_FRAME_WIDTH))
    src_height = int(cap_fg.get(cv2.CAP_PROP_FRAME_HEIGHT))
    src_fps = cap_fg.get(cv2.CAP_PROP_FPS) or 30.0
    total_fg_frames = int(cap_fg.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    total_bg_frames = int(cap_bg.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    # 出力 fps を決定（低くして処理を減らす）
    out_fps = float(OUTPUT_FPS) if OUTPUT_FPS and OUTPUT_FPS > 0 else float(src_fps)

    # max_frames（優先度: MAX_OUTPUT_FRAMES > MAX_OUTPUT_SECONDS > 無制限）
    if MAX_OUTPUT_FRAMES and MAX_OUTPUT_FRAMES > 0:
        max_frames = int(MAX_OUTPUT_FRAMES)
    elif MAX_OUTPUT_SECONDS and MAX_OUTPUT_SECONDS > 0:
        max_frames = int(max(1, round(MAX_OUTPUT_SECONDS * out_fps)))
    else:
        max_frames = None

    print(f"FG frames: {total_fg_frames}, BG frames: {total_bg_frames}, src {src_width}x{src_height} @ {src_fps}fps")
    print(f"Output: {out_fps} fps, max_frames: {max_frames if max_frames else 'none'}")

    out = cv2.VideoWriter(OUTPUT_VIDEO_RAW, FOURCC, out_fps, (src_width, src_height))

    # プリロード判定（小さければ全プリロード）
    preload = False
    est_bytes = src_width * src_height * 3 * max(1, total_fg_frames)
    if total_fg_frames > 0 and total_fg_frames <= PRELOAD_FRAME_LIMIT and est_bytes <= MAX_PRELOAD_BYTES:
        preload = True

    fg_frames = []
    fg_sample_indices = None
    if preload:
        print("プリロード: 前景をメモリに読み込みます...")
        cap_fg.set(cv2.CAP_PROP_POS_FRAMES, 0)
        cnt = 0
        while True:
            ret, f = cap_fg.read()
            if not ret:
                break
            if (f.shape[1], f.shape[0]) != (src_width, src_height):
                f = cv2.resize(f, (src_width, src_height))
            fg_frames.append(f)
            cnt += 1
        print(f"プリロード完了: {cnt} frames")
        loop_reset(cap_bg)
    else:
        # FG_SAMPLE_POOL_SIZE が指定されていれば index pool を作成してランダムシーク回数を低減
        if FG_SAMPLE_POOL_SIZE and FG_SAMPLE_POOL_SIZE > 0 and total_fg_frames > 0:
            pool_size = min(total_fg_frames, FG_SAMPLE_POOL_SIZE)
            fg_sample_indices = random.sample(range(total_fg_frames), pool_size)
            print(f"FG sample pool: {pool_size} indices (reduces random seeks)")
        else:
            fg_sample_indices = None
            print("FG sample pool: none (random from full range)")

    # 背景読み出しのステップを決める（src_fps -> out_fps に合わせて読み出し頻度を下げる）
    bg_step = max(1, int(round(src_fps / out_fps)))
    bg_frame_idx = 0  # 現在読み出す背景フレーム位置（int）
    # もし bg_frame_idx 超過したらループするように cap_bg.set を使う

    # 初期色
    current_color = rand_bgr().astype(np.float32)
    target_color = current_color.copy()
    transition_progress = 1.0

    frame_count = 0
    while True:
        # 終了条件
        if max_frames is not None and frame_count >= max_frames:
            print(f"\nReached max output frames ({max_frames}). Stopping.")
            break

        # 背景は bg_frame_idx を指定して読み出す（これで読み出し回数を減らす）
        if total_bg_frames > 0:
            cap_bg.set(cv2.CAP_PROP_POS_FRAMES, bg_frame_idx % max(1, total_bg_frames))
            ret_bg, frame_bg = cap_bg.read()
            bg_frame_idx += bg_step
            if not ret_bg:
                # ループ失敗時は再リセットして1フレーム読んでみる
                loop_reset(cap_bg)
                ret_bg, frame_bg = cap_bg.read()
                if not ret_bg:
                    print("背景ビデオの読み込みに失敗しました。")
                    break
        else:
            # 背景フレームが無ければ前景をそのまま返す後続処理でカバー
            ret_bg = False
            frame_bg = None

        # サイズ合わせ（必ず出力サイズにする）
        if ret_bg and (frame_bg.shape[1], frame_bg.shape[0]) != (src_width, src_height):
            frame_bg = cv2.resize(frame_bg, (src_width, src_height))

        frame_count += 1
        sys.stdout.write(f"\rProcessing output frame {frame_count}")
        sys.stdout.flush()

        # 前景選択
        if preload:
            if len(fg_frames) == 0:
                fg_frame = frame_bg.copy() if ret_bg else np.zeros((src_height, src_width, 3), dtype=np.uint8)
            else:
                fg_frame = fg_frames[random.randrange(len(fg_frames))]
        else:
            if total_fg_frames <= 0:
                fg_frame = frame_bg.copy() if ret_bg else np.zeros((src_height, src_width, 3), dtype=np.uint8)
            else:
                if fg_sample_indices:
                    idx = random.choice(fg_sample_indices)
                else:
                    idx = random.randrange(total_fg_frames)
                ret_fg, fg_frame = safe_read_frame_at(cap_fg, idx)
                if not ret_fg:
                    # フォールバック
                    ret_fg2, fg_frame = cap_fg.read()
                    if not ret_fg2:
                        fg_frame = frame_bg.copy() if ret_bg else np.zeros((src_height, src_width, 3), dtype=np.uint8)
            if (fg_frame.shape[1], fg_frame.shape[0]) != (src_width, src_height):
                fg_frame = cv2.resize(fg_frame, (src_width, src_height))

        # ターゲットカラー更新
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

        # マスク作成（ターゲット色に近いほど背景優先）
        f = fg_frame.astype(np.float32)
        c = current_color.reshape((1,1,3)).astype(np.float32)
        diff = f - c
        dist = np.linalg.norm(diff, axis=2).astype(np.float32)
        tol = max(1.0, float(TOLERANCE))
        mask = 1.0 - (dist / tol)
        mask = np.clip(mask, 0.0, 1.0)

        # マスクぼかし
        bk = ensure_odd(BLUR_KERNEL)
        if bk > 1:
            mask = cv2.GaussianBlur(mask, (bk, bk), 0)

        alpha_3 = mask[..., np.newaxis]  # 1 => background

        # 背景がない場合は黒を使う
        if not ret_bg:
            bg_f = np.zeros_like(f)
        else:
            bg_f = frame_bg.astype(np.float32)

        # 合成
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
