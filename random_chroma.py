import cv2
import numpy as np
import random
import sys
import argparse # コマンドライン引数を扱うために追加

# --- 設定項目 ---
INPUT_VIDEO_FG = 'video1.mp4'
INPUT_VIDEO_BG = 'video2.mp4'
# ★変更点: Pythonスクリプト側では中間ファイルとしてmp4を出力
OUTPUT_VIDEO = 'final_video.mp4'
# --- 設定はここまで ---

# ★追加: 16進数カラーコードをBGRタプルに変換する関数
def hex_to_bgr(hex_color):
    """ #RRGGBB 形式の16進数カラーコードを (B, G, R) タプルに変換する """
    hex_color = hex_color.lstrip('#')
    return tuple(int(hex_color[i:i+2], 16) for i in (4, 2, 0))

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
    
    # ★変更点: 残像効果のための前のフレームを保持する変数を初期化
    prev_frame = None
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
        
        # クロマキー処理
        lower_bound = np.clip(random_color_bgr - args.tolerance, 0, 255)
        upper_bound = np.clip(random_color_bgr + args.tolerance, 0, 255)
        mask = cv2.inRange(frame_fg, lower_bound, upper_bound)
        mask_inv = cv2.bitwise_not(mask)
        fg_masked = cv2.bitwise_and(frame_fg, frame_fg, mask=mask_inv)
        bg_masked = cv2.bitwise_and(frame_bg_resized, frame_bg_resized, mask=mask)
        chroma_frame = cv2.add(fg_masked, bg_masked)

        # ★変更点: 残像効果のロジックを修正
        if prev_frame is None:
            # 最初のフレームはそのまま保持
            prev_frame = chroma_frame.copy()
        
        # 現在のフレームと前のフレームをブレンドして残像を作成
        afterimage_frame = cv2.addWeighted(chroma_frame, 1.0 - args.afterimage_strength, prev_frame, args.afterimage_strength, 0)
        # 次のループのために現在のフレームを保持
        prev_frame = afterimage_frame.copy()

        output_frame = afterimage_frame

        # ★変更点: 単色オーバーレイ機能
        if args.overlay_color != 'none':
            try:
                bgr_color = hex_to_bgr(args.overlay_color)
                # フレームと同じサイズの単色レイヤーを作成
                color_layer = np.full((height, width, 3), bgr_color, dtype=np.uint8)
                # 残像フレームと単色レイヤーをブレンド
                output_frame = cv2.addWeighted(output_frame, 1.0 - args.overlay_strength, color_layer, args.overlay_strength, 0)
            except Exception as e:
                print(f"\n無効なカラーコードです: {args.overlay_color} エラー: {e}")


        out.write(output_frame)

    print(f"\n動画処理が完了しました。'{OUTPUT_VIDEO}' として保存されました。")

    cap_fg.release()
    cap_bg.release()
    out.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    # ★変更点: コマンドライン引数の設定を全面的に更新
    parser = argparse.ArgumentParser(description='動画にクロマキー、残像、カラーオーバーレイ効果を適用します。')
    parser.add_argument('--tolerance', type=int, default=50, help='クロマキーの色の許容度 (0-255)')
    parser.add_argument('--afterimage-strength', type=float, default=0.3, help='残像の強さ (0.0 - 1.0)')
    parser.add_argument('--overlay-color', type=str, default='none', help='オーバーレイする単色 (#RRGGBB形式、または "none")')
    parser.add_argument('--overlay-strength', type=float, default=0.4, help='単色オーバーレイの強さ (0.0 - 1.0)')
    
    args = parser.parse_args()
    process_video(args)
