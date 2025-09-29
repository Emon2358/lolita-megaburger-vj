import sys
import os
import subprocess
import numpy as np
import cv2
import json

def run_command(command):
    """コマンドを実行し、エラーがあれば例外を発生させる"""
    process = subprocess.run(command, capture_output=True, text=True)
    if process.returncode != 0:
        print("Error executing command:", " ".join(command))
        print("STDOUT:", process.stdout)
        print("STDERR:", process.stderr)
        raise RuntimeError(f"FFmpeg command failed with exit code {process.returncode}")
    return process

def get_video_info(video_path):
    """ffprobeを使って動画の情報を取得する"""
    command = ['ffprobe', '-v', 'quiet', '-print_format', 'json', '-show_format', '-show_streams', video_path]
    result = run_command(command)
    return json.loads(result.stdout)

# 単色化の関数は削除 (または呼び出さない)

def apply_ultra_glitch(frame, persistence):
    """
    超強力なグリッチエフェクトを適用する
    """
    h, w, _ = frame.shape
    
    # フレームのコピーを作成し、直接操作する
    current_frame = frame.copy()

    # 1. 激しいRGBチャンネル操作と空間歪み
    if np.random.rand() < 0.8: # 高確率で発生
        b, g, r = cv2.split(current_frame)
        
        # 各チャンネルを個別に激しくずらす
        shift_r = np.random.randint(-50, 50)
        shift_g = np.random.randint(-50, 50)
        shift_b = np.random.randint(-50, 50)
        
        r_shifted = np.roll(r, shift_r, axis=1) # 水平シフト
        g_shifted = np.roll(g, shift_g, axis=0) # 垂直シフト
        b_shifted = np.roll(b, shift_b, axis=1) # 水平シフト
        
        current_frame = cv2.merge([b_shifted, g_shifted, r_shifted])

    # 2. ピクセル単位の破壊とランダムノイズ / ラインスクランブル強化
    if np.random.rand() < 0.9: # ほぼ常に発生
        # 全体を高周波ノイズで覆う
        noise = np.random.randint(0, 255, (h, w, 3), dtype=np.uint8)
        current_frame = cv2.addWeighted(current_frame, 0.7, noise, 0.3, 0)
        
        # ランダムなラインスクランブル (行をランダムにずらす)
        if np.random.rand() < 0.7:
            for _ in range(np.random.randint(1, h // 5)): # 多くの行を対象に
                row_idx = np.random.randint(0, h)
                shift_amount = np.random.randint(-w // 2, w // 2)
                current_frame[row_idx, :] = np.roll(current_frame[row_idx, :], shift_amount, axis=0)

    # 3. よりアグレッシブなデータモッシュ (大きなブロックと高頻度)
    prev_frame = persistence.get('prev_frame')
    if prev_frame is not None and np.random.rand() < 0.5: # 50%の確率で発生
        # ランダムな大きな矩形領域を、前のフレームの内容で上書きする
        x, y = np.random.randint(0, w // 4), np.random.randint(0, h // 4)
        rw, rh = np.random.randint(w // 2, w), np.random.randint(h // 2, h)
        current_frame[y:y+rh, x:x+rw] = prev_frame[y:y+rh, x:x+rw]
    persistence['prev_frame'] = frame.copy() # オリジナルのフレームを次回のデータモッシュ用に保存

    # 4. 完全にランダムなブロックの挿入
    if np.random.rand() < 0.6: # 60%の確率で発生
        block_w, block_h = np.random.randint(w // 10, w // 2), np.random.randint(h // 10, h // 2)
        block_x, block_y = np.random.randint(0, w - block_w), np.random.randint(0, h - block_h)
        random_color = (np.random.randint(0, 256), np.random.randint(0, 256), np.random.randint(0, 256))
        
        # ランダムな色のブロックで一部を上書き
        current_frame[block_y:block_y+block_h, block_x:block_x+block_w] = random_color

    # 5. よりアグレッシブなフィードバックループ
    feedback_frame = persistence.get('feedback_frame')
    if feedback_frame is not None and np.random.rand() < 0.9: # 90%の確率で発生
        # 以前のフィードバック結果をさらに変形させて重ねる
        scale = np.random.uniform(0.9, 0.99) # 拡大縮小をランダムに
        angle = np.random.uniform(-5, 5) # わずかな回転を加える
        
        M_rot = cv2.getRotationMatrix2D((w/2, h/2), angle, scale)
        feedback_transformed = cv2.warpAffine(feedback_frame, M_rot, (w, h))
        
        # 現在のフレームとフィードバック結果を強力にブレンド
        current_frame = cv2.addWeighted(current_frame, 0.6, feedback_transformed, 0.4, 0)
        
    persistence['feedback_frame'] = current_frame.copy() # 現在の結果を次回のフィードバックのために保存
    
    return current_frame

def main():
    if len(sys.argv) != 5:
        print("使い方: python process_videos.py <基準動画> <重ねる動画> <出力ファイル> <色(R,G,B)>")
        sys.exit(1)

    base_video_path = sys.argv[1]
    overlay_video_path = sys.argv[2]
    output_path = sys.argv[3]
    # target_color_str = sys.argv[4] # 単色化はしないため、引数は受け取るが使わない

    print("動画情報を取得中...")
    base_info = get_video_info(base_video_path)
    overlay_info = get_video_info(overlay_video_path)
    base_duration = float(base_info['format']['duration'])
    overlay_duration = float(overlay_info['format']['duration'])
    
    print("重ねる動画の速度を調整中...")
    temp_overlay_path = "temp_overlay.mp4"
    speed_factor = overlay_duration / base_duration
    command_respeed = [
        'ffmpeg', '-y', '-i', overlay_video_path, '-vf', f'setpts={1/speed_factor}*PTS',
        '-af', f'atempo={speed_factor}', temp_overlay_path
    ]
    run_command(command_respeed)

    print("フレームを1枚ずつ処理中...")
    cap_base = cv2.VideoCapture(base_video_path)
    cap_overlay = cv2.VideoCapture(temp_overlay_path)
    width = int(cap_base.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap_base.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap_base.get(cv2.CAP_PROP_FPS)

    command_encode = [
        'ffmpeg', '-y', '-f', 'rawvideo', '-vcodec', 'rawvideo', '-s', f'{width}x{height}',
        '-pix_fmt', 'bgr24', '-r', str(fps), '-i', '-', '-i', base_video_path,
        '-c:v', 'libx264', '-c:a', 'aac', '-map', '0:v:0', '-map', '1:a:0?',
        '-pix_fmt', 'yuv420p', output_path
    ]
    
    process = subprocess.Popen(command_encode, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    
    persistence1 = {} # 各動画ストリーム用に別々のpersistenceを保持
    persistence2 = {}

    while True:
        ret1, frame1 = cap_base.read()
        ret2, frame2 = cap_overlay.read()
        if not ret1 or not ret2:
            break
        
        # ★★★ 修正箇所: 単色化を削除し、直接強力なグリッチを適用 ★★★
        
        # 基準動画にグリッチを適用
        frame1_fx = apply_ultra_glitch(frame1, persistence1)
        
        # 重ねる動画はリサイズしてからグリッチを適用
        resized_frame2 = cv2.resize(frame2, (width, height))
        frame2_fx = apply_ultra_glitch(resized_frame2, persistence2)

        # 2つのグリッチ映像を合成 (ブレンド比率を調整してさらに混濁させる)
        final_frame = cv2.addWeighted(frame1_fx, 0.4, frame2_fx, 0.6, 0) # overlayを強めに

        # ★★★ 修正箇所ここまで ★★★
        
        try:
            process.stdin.write(final_frame.tobytes())
        except (BrokenPipeError, IOError):
            print("FFmpegプロセスが予期せず終了しました。")
            break

    print("後処理を実行中...")
    cap_base.release()
    cap_overlay.release()
    if process.stdin:
        process.stdin.close()
    process.wait()
    if os.path.exists(temp_overlay_path):
        os.remove(temp_overlay_path)

    print(f"処理が完了しました: {output_path}")

if __name__ == "__main__":
    main()
