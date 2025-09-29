# random_chroma.py
# Aggressive chroma/glitch processor (safe for headless env)
import cv2
import numpy as np
import random
import sys
import time
import os

# --- パラメータ（目的に応じて調整） ---
TOLERANCE = 80                 # 色距離の閾値（大きいほど大胆に置換）
CHANGE_INTERVAL_FRAMES = 3     # 色を切り替える頻度（小さいほど速い）
TRANSITION_FRAMES = 6          # 色の滑らかな移行フレーム数
BLUR_KERNEL = 5                # マスクぼかし（奇数推奨）
AGGRESSIVENESS = 0.9           # 0..1。1 に近いほど激烈
JPEG_QUALITY = 20              # 1..100（低いほど荒いブロックノイズ）
SLICE_COUNT = 20               # ランダムスライス数（多いほど破綻）
SALT_PEPPER_PROB = 0.003       # ノイズ確率

# ファイル名（ワークフロー側で再エンコード）
INPUT_VIDEO_FG = 'video1.mp4'
INPUT_VIDEO_BG = 'video2.mp4'
OUTPUT_VIDEO_RAW = 'final_video_raw.mp4'  # まずは生出力
OUTPUT_VIDEO_FINAL = 'final_video_100mb.mp4'  # (ワークフローで生成)

# --- ユーティリティ関数 ---
def rand_bgr():
    return np.array([random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)], dtype=np.uint8)

def loop_reset(cap):
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

def apply_glitch(frame, intensity=1.0):
    """激しいグリッチ系エフェクトを符号化（チャンネル分離・スライス・JPEGノイズ等）"""
    h, w = frame.shape[:2]
    f = frame.copy()

    # チャンネル分離（左右ずらす）
    max_shift = int(16 * intensity)
    for i in range(3):  # B,G,R
        dx = random.randint(-max_shift, max_shift)
        M = np.float32([[1, 0, dx], [0, 1, 0]])
        ch = f[:, :, i]
        f[:, :, i] = cv2.warpAffine(ch, M, (w, h), borderMode=cv2.BORDER_REFLECT)

    # ランダム水平スライスの横シフト
    slices = min(SLICE_COUNT, h)
    # pick slice boundaries
    ys = sorted(random.sample(range(0, h), slices)) if slices > 0 else []
    prev_y = 0
    out = np.zeros_like(f)
    for y in ys + [h]:
        slice_h = y - prev_y
        if slice_h <= 0:
            prev_y = y
            continue
        dx = int((random.random() - 0.5) * 2 * max_shift * intensity)
        slice_img = f[prev_y:y, :, :]
        if dx != 0:
            slice_img = np.roll(slice_img, dx, axis=1)
        out[prev_y:y, :, :] = slice_img
        prev_y = y
    f = out

    # JPEG 低品質でエンコード→デコード（ブロックノイズ）
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), max(1, int(JPEG_QUALITY * (1.0 - 0.5 * (1 - intensity))))]
    _, enc = cv2.imencode('.jpg', f, encode_param)
    f = cv2.imdecode(enc, cv2.IMREAD_COLOR)

    # ポスタリゼーション（色数削減）
    levels = max(2, int(16 * (1.0 - intensity * 0.5)))
    f = np.floor(f / (256 / levels)) * (256 // levels)
    f = f.astype(np.uint8)

    # ソルト＆ペッパー
    if SALT_PEPPER_PROB > 0:
        noise = np.random.rand(h, w)
        sp_mask_salt = (noise < (SALT_PEPPER_PROB * intensity))
        sp_mask_pepper = (noise > (1 - (SALT_PEPPER_PROB * intensity)))
        f[sp_mask_salt] = 255
        f[sp_mask_pepper] = 0

    # 色相シフト（軽度）
    hsv = cv2.cvtColor(f, cv2.COLOR_BGR2HSV).astype(np.int32)
    shift = int(30 * (random.random() - 0.5) * intensity)
    hsv[:, :, 0] = (hsv[:, :, 0] + shift) % 180
    f = cv2.cvtColor(np.clip(hsv, 0, 255).astype(np.uint8), cv2.COLOR_HSV2BGR)

    return f

# --- メイン処理 ---
def process_video():
    # ランダムシード（毎回変える）
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
    out = cv2.VideoWriter(OUTPUT_VIDEO_RAW, fourcc, fps, (width, height))

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

        # 色のターゲットを周期的に変更
        if frame_count % CHANGE_INTERVAL_FRAMES == 0:
            target_color = rand_bgr().astype(np.float32)
            transition_progress = 0.0

        # 補間（滑らかに current_color -> target_color）
        if transition_progress < 1.0:
            transition_progress += 1.0 / max(1, TRANSITION_FRAMES)
            t = min(1.0, transition_progress)
            current_color = (1 - t) * current_color + t * target_color

        # 色距離でソフトマスクを作る
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

        # 激しさエフェクトを確率的に適用
        if random.random() < 0.7 * AGGRESSIVENESS:
            final_frame = apply_glitch(final_frame, intensity=AGGRESSIVENESS)

        out.write(final_frame)

    print(f"\n動画処理が完了しました。'{OUTPUT_VIDEO_RAW}' として保存されました。")

    cap_fg.release()
    cap_bg.release()
    out.release()

    # headless 環境では destroyAllWindows が未実装なことがあるため安全に呼ぶ
    try:
        cv2.destroyAllWindows()
    except Exception:
        pass

if __name__ == '__main__':
    process_video()
