# random_chroma.py
import cv2
import numpy as np
import random
import sys

# --- 設定項目 ---
INPUT_VIDEO_FG = 'video1.mp4'
INPUT_VIDEO_BG = 'video2.mp4'
# ★変更点: 出力ファイル名を最終成果物の名前に変更
OUTPUT_VIDEO = 'final_video.mp4' 
TOLERANCE = 30 
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
    while True:
        ret_fg, frame_fg = cap_fg.read()
        ret_bg, frame_bg = cap_bg.read()

        if not ret_fg or not ret_bg:
            break
        
        frame_count += 1
        sys.stdout.write(f"\rフレーム処理中: {frame_count} / {total_frames}")
        sys.stdout.flush()

        frame_bg_resized = cv2.resize(frame_bg, (width, height))
        random_color_bgr = np.array([random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)])
        
        lower_bound = np.clip(random_color_bgr - TOLERANCE, 0, 255)
        upper_bound = np.clip(random_color_bgr + TOLERANCE, 0, 255)

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
