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

def parse_color(color_str):
    """'R,G,B' 形式の文字列を OpenCV の BGR タプルに変換"""
    try:
        r, g, b = map(int, color_str.split(','))
        return (b, g, r)
    except ValueError:
        raise ValueError("色は 'R,G,B' 形式で指定してください (例: '255,0,0')")

def apply_advanced_glitch(frame, persistence):
    """
    電気耳/奇怪電波倶楽部風の高度なグリッチエフェクトを適用する
    """
    # 1. RGBシフト（色ずれ）
    if np.random.rand() < 0.2: # 20%の確率で発生
        b, g, r = cv2.split(frame)
        shift = np.random.randint(-15, 15)
        g_shifted = np.roll(g, shift, axis=1)
        b_shifted = np.roll(b, shift, axis=0)
        frame = cv2.merge([b_shifted, g_shifted, r])

    # 2. ブロックディスプレイスメント（既存のグリッチを強化）
    if np.random.rand() < 0.5: # 50%の確率で発生
        num_blocks = np.random.randint(15, 80)
        block_height = frame.shape[0] // num_blocks if num_blocks > 0 else frame.shape[0]
        for i in range(num_blocks):
            start = i * block_height
            end = (i + 1) * block_height
            shift = np.random.randint(-frame.shape[1] // 4, frame.shape[1] // 4)
            frame[start:end, :] = np.roll(frame[start:end, :], shift, axis=1)

    # 3. データモッシュ風エフェクト（フレームの持続）
    prev_frame = persistence.get('prev_frame')
    if prev_frame is not None and np.random.rand() < 0.3: # 30%の確率で発生
        h, w, _ = frame.shape
        x, y = np.random.randint(0, w//2), np.random.randint(0, h//2)
        rw, rh = np.random.randint(w//2, w), np.random.randint(h//2, h)
        frame[y:y+rh, x:x+rw] = prev_frame[y:y+rh, x:x+rw]
    persistence['prev_frame'] = frame.copy()

    # 4. フィードバックループ
    feedback_frame = persistence.get('feedback_frame')
    if feedback_frame is not None and np.random.rand() < 0.8: # 80%の確率で発生
        h, w, _ = frame.shape
        scale = 0.98
        M = np.float32([[scale, 0, w*(1-scale)/2], [0, scale, h*(1-scale)/2]])
        feedback_transformed = cv2.warpAffine(feedback_frame, M, (w, h))
        frame = cv2.addWeighted(frame, 0.8, feedback_transformed, 0.2, 0)
    persistence['feedback_frame'] = frame.copy()
    
    return frame

def apply_monochrome(frame, target_color_bgr):
    """フレームを指定された単色に変換する"""
    target_color_np = np.array(target_color_bgr, dtype=np.uint8)
    gray_intensity = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) / 255.0
    return np.einsum('ij,k->ijk', gray_intensity, target_color_np).astype(np.uint8)

def main():
    if len(sys.argv) != 5:
        print("使い方: python process_videos.py <基準動画> <重ねる動画> <出力ファイル> <色(R,G,B)>")
        sys.exit(1)

    base_video_path = sys.argv[1]
    overlay_video_path = sys.argv[2]
    output_path = sys.argv[3]
    color_str = sys.argv[4]

    target_color_bgr = parse_color(color_str)
    
    print("動画情報を取得中...")
    base_info = get_video_info(base_video_path)
    overlay_info = get_video_info(overlay_video_path)

    base_duration = float(base_info['format']['duration'])
    overlay_duration = float(overlay_info['format']['duration'])
    
    print("重ねる動画の速度を調整中...")
    temp_overlay_path = "temp_overlay.mp4"
    speed_factor = overlay_duration / base_duration
    command_respeed = [
        'ffmpeg', '-y', '-i', overlay_video_path,
        '-vf', f'setpts={1/speed_factor}*PTS',
        '-af', f'atempo={speed_factor}',
        temp_overlay_path
    ]
    run_command(command_respeed)

    print("フレームを1枚ずつ処理中...")
    cap_base = cv2.VideoCapture(base_video_path)
    cap_overlay = cv2.VideoCapture(temp_overlay_path)
    
    width = int(cap_base.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap_base.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap_base.get(cv2.CAP_PROP_FPS)

    command_encode = [
        'ffmpeg', '-y', '-f', 'rawvideo', '-vcodec', 'rawvideo',
        '-s', f'{width}x{height}', '-pix_fmt', 'bgr24', '-r', str(fps),
        '-i', '-', '-i', base_video_path, '-c:v', 'libx264', '-c:a', 'aac',
        '-map', '0:v:0', '-map', '1:a:0?', '-pix_fmt', 'yuv4p', output_path
    ]
    
    process = subprocess.Popen(command_encode, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    
    persistence1 = {}
    persistence2 = {}

    while True:
        ret1, frame1 = cap_base.read()
        ret2, frame2 = cap_overlay.read()
        if not ret1 or not ret2:
            break
        
        # ★★★ 修正箇所: ここからロジックを変更 ★★★
        
        # 1. 最初に各フレームを単色化する
        mono_frame1 = apply_monochrome(frame1, target_color_bgr)
        
        resized_frame2 = cv2.resize(frame2, (width, height))
        mono_frame2 = apply_monochrome(resized_frame2, target_color_bgr)

        # 2. 単色化したフレームに対して、高度なグリッチを適用する
        #    これにより、色ずれなどが新たな色として現れる
        frame1_fx = apply_advanced_glitch(mono_frame1, persistence1)
        frame2_fx = apply_advanced_glitch(mono_frame2, persistence2)

        # 3. エフェクト適用済みの2つのフレームを合成する
        final_frame = cv2.addWeighted(frame1_fx, 0.5, frame2_fx, 0.5, 0)
        
        # ★★★ 修正箇所: ここまで ★★★

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
