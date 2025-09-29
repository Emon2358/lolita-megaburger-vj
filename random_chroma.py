# random_chroma.py
# Foreground color-key (make pixels close to a random color transparent) compositor
import cv2
import numpy as np
import random
import sys
import time
import os

# ========== パラメータ ==========
INPUT_VIDEO_FG = 'video1.mp4'
INPUT_VIDEO_BG = 'video2.mp4'
OUTPUT_VIDEO_RAW = 'final_video_raw.mp4'

# カラー透過（チョーク）関連
TOLERANCE = 10             # 色距離の閾値（小さいほど厳密に色を選ぶ）
CHANGE_INTERVAL_FRAMES = 1       # カラーターゲットを何フレームごとに切り替えるか（1で毎フレーム）
TRANSITION_FRAMES = 0            # カラーの補間フレーム数（0で瞬変、>0で滑らかに変化）
BLUR_KERNEL = 1          # マスクのぼかしカーネル（奇数推奨。1にするとぼかし無し）

# 前景フレーム取り出しのプリロード制御
PRELOAD_FRAME_LIMIT = 700
MAX_PRELOAD_BYTES = 800 * 1024 * 1024  # 800MB 上限目安

# 軽いエフェクト（必要なら調整）
AGGRESSIVENESS = 0.0             # 0でグリッチ無効、0..1 で軽いチャンネルオフセットが入る
SALT_PEPPER_PROB = 0.0

FOURCC = cv2.VideoWriter_fourcc(*'mp4v')

# ========== ユーティリティ ==========
def rand_bgr():
    return np.array([random.randint(0,255), random.randint(0,255), random.randint(0,255)], dtype=np.uint8)

def loop_reset(cap):
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

def safe_read_frame_at(cap, idx):
    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
    return cap.read()

def maybe_apply_light_glitch(img, intensity=0.3):
    if intensity <= 0: return img
    out = img.copy()
    h, w = out.shape[:2]
    max_shift = int(6 * intensity)
    for ch in range(3):
        dx = random.randint(-max_shift, max_shift)
        M = np.float32([[1,0,dx],[0,1,0]])
        out[:,:,ch] = cv2.warpAffine(out[:,:,ch], M, (w,h), borderMode=cv2.BORDER_REFLECT)
    if SALT_PEPPER_PROB > 0:
        noise = np.random.rand(h, w)
        out[noise < (SALT_PEPPER_PROB * intensity)] = 255
        out[noise > (1 - (SALT_PEPPER_PROB * intensity))] = 0
    return out

# ========== メイン処理 ==========
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
    fps = cap_fg.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap_fg.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    bg_total_frames = int(cap_bg.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    print(f"FG frames: {total_frames}, BG frames: {bg_total_frames}, {width}x{height} @ {fps}fps")

    out = cv2.VideoWriter(OUTPUT_VIDEO_RAW, FOURCC, fps, (width, height))

    # プリロード判定
    preload = False
    estimated_bytes = width * height * 3 * max(1, total_frames)
    if total_frames > 0 and total_frames <= PRELOAD_FRAME_LIMIT and estimated_bytes <= MAX_PRELOAD_BYTES:
        preload = True

    fg_frames = []
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
        print("プリロードなし (オンザフライでランダムアクセスします)")

    # カラー透過用の初期色
    current_color = rand_bgr().astype(np.float32)
    target_color = current_color.copy()
    transition_progress = 1.0

    frame_count = 0
    while True:
        ret_bg, frame_bg = cap_bg.read()
        if not ret_bg:
            loop_reset(cap_bg)
            ret_bg, frame_bg = cap_bg.read()
            if not ret_bg:
                print("背景ビデオの読み込みに失敗しました。")
                break

        frame_count += 1
        sys.stdout.write(f"\rProcessing frame {frame_count}")
        sys.stdout.flush()

        # ランダムに前景フレームを選択
        if preload:
            if len(fg_frames) == 0:
                fg_frame = frame_bg.copy()
            else:
                fg_frame = fg_frames[random.randrange(len(fg_frames))]
        else:
            if total_frames <= 0:
                fg_frame = frame_bg.copy()
            else:
                idx = random.randrange(total_frames)
                ret_fg, fg_frame = safe_read_frame_at(cap_fg, idx)
                if not ret_fg:
                    # fallback
                    ret_fg2, fg_frame = cap_fg.read()
                    if not ret_fg2:
                        fg_frame = frame_bg.copy()
            if (fg_frame.shape[1], fg_frame.shape[0]) != (width, height):
                fg_frame = cv2.resize(fg_frame, (width, height))

        # ターゲットカラーの切り替え（瞬変または補間）
        if frame_count % max(1, CHANGE_INTERVAL_FRAMES) == 0:
            target_color = rand_bgr().astype(np.float32)
            transition_progress = 0.0

        if transition_progress < 1.0 and TRANSITION_FRAMES > 0:
            transition_progress += 1.0 / max(1, TRANSITION_FRAMES)
            t = min(1.0, transition_progress)
            current_color = (1 - t) * current_color + t * target_color
        else:
            # 瞬変時はここで current_color を target に揃える
            if TRANSITION_FRAMES == 0:
                current_color = target_color.copy()

        # 色距離に基づくソフトマスク作成（現在のターゲット色に近いほど「透過」＝背景を採用）
        f = fg_frame.astype(np.float32)
        c = current_color.reshape((1,1,3)).astype(np.float32)
        diff = f - c
        # 色距離（Euclid）
        dist = np.linalg.norm(diff, axis=2).astype(np.float32)
        tol = max(1.0, float(TOLERANCE))
        # mask: 1 -> 背景（透過）、0 -> 前景（残す）
        mask = 1.0 - (dist / tol)
        mask = np.clip(mask, 0.0, 1.0)

        # ぼかしてソフトにする（BLUR_KERNELは奇数が望ましい）
        bk = int(BLUR_KERNEL)
        if bk % 2 == 0:
            bk = bk+1 if bk>1 else 1
        if bk > 1:
            mask = cv2.GaussianBlur(mask, (bk, bk), 0)

        alpha_3 = mask[..., np.newaxis]  # shape (h,w,1) : 1 => background

        # 軽いエフェクト（任意）
        if AGGRESSIVENESS > 0 and random.random() < AGGRESSIVENESS:
            f = maybe_apply_light_glitch(f.astype(np.uint8), AGGRESSIVENESS).astype(np.float32)

        bg_f = frame_bg.astype(np.float32)
        # 合成: mask が 1 のとき背景優先、0 のとき前景優先
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
