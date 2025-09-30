import cv2
import numpy as np
import random
import sys
import argparse # コマンドライン引数を扱うために追加

# --- 設定項目 ---
INPUT_VIDEO_FG = 'video1.mp4'
INPUT_VIDEO_BG = 'video2.mp4'
OUTPUT_VIDEO = 'final_video.mp4'
# --- 設定はここまで ---

def process_video(args):
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
    
    # ★変更点: ポスタリゼーション時間エフェクト用の変数を初期化
    held_frame = None
    frame_counter = 0
    
    while True:
        ret_fg, frame_fg = cap_fg.read()
        ret_bg, frame_bg = cap_bg.read()

        if not ret_fg or not ret_bg:
            break
        
        frame_counter += 1
        sys.stdout.write(f"\rフレーム処理中: {frame_counter} / {total_frames}")
        sys.stdout.flush()

        # ★変更点: 指定した間隔でフレームを更新するロジック
        # (カウンター % 間隔 == 1) のタイミングで新しいフレームを処理
        if (frame_counter % args.posterize_time_duration == 1) or held_frame is None:
            frame_bg_resized = cv2.resize(frame_bg, (width, height))
            random_color_bgr = np.array([random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)])
            
            # クロマキー処理
            lower_bound = np.clip(random_color_bgr - args.tolerance, 0, 255)
            upper_bound = np.clip(random_color_bgr + args.tolerance, 0, 255)
            mask = cv2.inRange(frame_fg, lower_bound, upper_bound)
            mask_inv = cv2.bitwise_not(mask)
            fg_masked = cv2.bitwise_and(frame_fg, frame_fg, mask=mask_inv)
            bg_masked = cv2.bitwise_and(frame_bg_resized, frame_bg_resized, mask=mask)
            
            # 処理したフレームを「保持」する
            held_frame = cv2.add(fg_masked, bg_masked)

        # ★変更点: 次の更新タイミングまで、保持したフレームを書き出し続ける
        out.write(held_frame)

    print(f"\n動画処理が完了しました。'{OUTPUT_VIDEO}' として保存されました。")

    cap_fg.release()
    cap_bg.release()
    out.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    # ★変更点: 引数をポスタリゼーション時間エフェクト用に変更
    parser = argparse.ArgumentParser(description='動画にクロマキーとポスタリゼーション時間エフェクトを適用します。')
    parser.add_argument('--tolerance', type=int, default=50, help='クロマキーの色の許容度 (0-255)')
    parser.add_argument('--posterize-time-duration', type=int, default=3, help='1フレームを保持する長さ（大きいほどカクカクになる）')
    
    args = parser.parse_args()
    # 期間は1以上である必要があるためバリデーション
    if args.posterize_time_duration < 1:
        args.posterize_time_duration = 1
        
    process_video(args)
