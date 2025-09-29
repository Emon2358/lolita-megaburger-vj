import cv2
import numpy as np
import random
import sys
import argparse

# --- 設定項目 ---
INPUT_VIDEO_FG = 'video1.mp4'
INPUT_VIDEO_BG = 'video2.mp4'
OUTPUT_VIDEO = 'final_video.mp4'
# --- 設定はここまで ---

def apply_color_effect(frame, effect):
    """
    指定されたカラーエフェクトをフレームに適用します。
    Args:
        frame (numpy.ndarray): 入力フレーム
        effect (str): 'none', 'bright', 'warm', 'cool' のいずれか
    Returns:
        numpy.ndarray: エフェクト適用後のフレーム
    """
    if effect == 'none':
        return frame

    # オーバーフローを防ぐためにfloat32に変換して計算
    frame = frame.astype(np.float32)

    if effect == 'bright':
        # 明るさを50増加させる
        frame = np.clip(frame + 50, 0, 255)
    elif effect == 'warm':
        # 暖色系: 赤チャンネルを強調し、青チャンネルを弱める
        frame[:, :, 2] = np.clip(frame[:, :, 2] * 1.2, 0, 255)  # BGRのRチャンネル
        frame[:, :, 0] = np.clip(frame[:, :, 0] * 0.8, 0, 255)  # BGRのBチャンネル
    elif effect == 'cool':
        # 寒色系: 青チャンネルを強調し、赤チャンネルを弱める
        frame[:, :, 0] = np.clip(frame[:, :, 0] * 1.2, 0, 255)  # BGRのBチャンネル
        frame[:, :, 2] = np.clip(frame[:, :, 2] * 0.8, 0, 255)  # BGRのRチャンネル

    return frame.astype(np.uint8)

def process_video(tolerance, color_effect, afterimage_strength):
    """
    動画を処理し、ランダムクロマキー、カラーエフェクト、残像効果を適用します。
    """
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
    print(f"設定値: Tolerance={tolerance}, Color Effect='{color_effect}', Afterimage Strength={afterimage_strength}")
    
    frame_count = 0
    prev_frame = None  # ★追加: 前のフレームを保存する変数

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
        
        lower_bound = np.clip(random_color_bgr - tolerance, 0, 255)
        upper_bound = np.clip(random_color_bgr + tolerance, 0, 255)

        mask = cv2.inRange(frame_fg, lower_bound, upper_bound)
        mask_inv = cv2.bitwise_not(mask)

        fg_masked = cv2.bitwise_and(frame_fg, frame_fg, mask=mask_inv)
        bg_masked = cv2.bitwise_and(frame_bg_resized, frame_bg_resized, mask=mask)
        final_frame = cv2.add(fg_masked, bg_masked)
        
        # ★変更点: 残像効果を適用
        if afterimage_strength > 0 and prev_frame is not None:
            # 現在のフレームと前のフレームをブレンド
            processed_frame = cv2.addWeighted(
                final_frame, 1.0 - afterimage_strength, prev_frame, afterimage_strength, 0
            )
        else:
            processed_frame = final_frame

        # 次のループのために、色補正前のフレームを保存
        prev_frame = final_frame.copy()

        # ★変更点: カラーエフェクトを適用
        output_frame = apply_color_effect(processed_frame, color_effect)

        out.write(output_frame)

    print(f"\n動画処理が完了しました。'{OUTPUT_VIDEO}' として保存されました。")

    cap_fg.release()
    cap_bg.release()
    out.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    # ★追加: コマンドライン引数を解析
    parser = argparse.ArgumentParser(description='Process video with random chroma key and effects.')
    parser.add_argument('--tolerance', type=int, default=50, help='Color tolerance for chroma keying.')
    parser.add_argument('--color-effect', type=str, default='none', choices=['none', 'bright', 'warm', 'cool'], help='Color effect to apply.')
    parser.add_argument('--afterimage-strength', type=float, default=0.0, help='Strength of the afterimage effect (0.0 to 1.0).')
    
    args = parser.parse_args()

    # afterimage-strengthの値を0.0から1.0の間に制限
    afterimage_strength_clamped = max(0.0, min(1.0, args.afterimage_strength))

    process_video(args.tolerance, args.color_effect, afterimage_strength_clamped)
