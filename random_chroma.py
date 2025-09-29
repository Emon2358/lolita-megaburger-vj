import cv2
import numpy as np
import random
import sys

# --- 💡 設定項目 ---
# これらの値を変更して効果を調整してください

# 1. 範囲 (TOLERANCE)
#    透過させる色の類似度。値を大きくするほど、より広い範囲の色が一気に透過します。
#    推奨値: 30 (控えめ) ~ 100 (大胆)
TOLERANCE = 50

# 2. 頻度・速度 (CHANGE_INTERVAL_FRAMES)
#    何フレームごとに透過する色をランダムに変えるか。
#    - 1 を設定すると、毎フレーム色が変わるため「最速」になります。
#    - 10 を設定すると、10フレーム（約0.3秒）は同じ色が使われるため「ゆっくり」になります。
#    - 小さいほど速く、大きいほど遅くなります。
CHANGE_INTERVAL_FRAMES = 1

# --- 基本設定 ---
INPUT_VIDEO_FG = 'video1.mp4'
INPUT_VIDEO_BG = 'video2.mp4'
OUTPUT_VIDEO = 'final_video.mp4' # GitHub Actionsのファイル名と合わせています
# --- 設定はここまで ---

def process_video():
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
    fps = cap_fg.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap_fg.get(cv2.CAP_PROP_FRAME_COUNT))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(OUTPUT_VIDEO, fourcc, fps, (width, height))

    print("動画処理を開始します...")
    
    frame_count = 0
    # 最初にランダムな色を生成
    random_color_bgr = np.array([random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)])

    while True:
        ret_fg, frame_fg = cap_fg.read()
        ret_bg, frame_bg = cap_bg.read()

        if not ret_fg or not ret_bg:
            break
        
        frame_count += 1
        sys.stdout.write(f"\rフレーム処理中: {frame_count} / {total_frames}")
        sys.stdout.flush()

        # ★変更点: 指定したフレーム間隔(CHANGE_INTERVAL_FRAMES)で色を更新する
        if frame_count % CHANGE_INTERVAL_FRAMES == 0:
            random_color_bgr = np.array([random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)])

        frame_bg_resized = cv2.resize(frame_bg, (width, height))
        
        lower_bound = np.clip(random_color_bgr - TOLERANCE, 0, 255)
        upper_bound = np.clip(random_color_bgr + TOLERANCE, 255, 255)

        mask = cv2.inRange(frame_fg, lower_bound, upper_bound)
        mask_inv = cv2.bitwise_not(mask)

        fg_masked = cv2.bitwise_and(frame_fg, frame_fg, mask=mask_inv)
        bg_masked = cv2.bitwise_and(frame_bg_resized, frame_bg_resized, mask=mask)
        final_frame = cv2.add(fg_masked, bg_masked)

        out.write(final_frame)

    print(f"\n動画処理が完了しました。'{OUTPUT_VIDEO}' として保存されました。")

    cap_fg.release()
    cap_bg.release()
    out.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    process_video()
