import sys
import os
import subprocess
import numpy as np
import cv2
import json
import shutil

def run_command(command):
    """指定されたコマンドを実行し、エラーがあれば例外を発生させる"""
    # shell=True を避け、コマンドをリストで渡すように変更
    process = subprocess.run(command, capture_output=True, text=True)
    if process.returncode != 0:
        print("Error executing command:", " ".join(command))
        print("STDOUT:", process.stdout)
        print("STDERR:", process.stderr)
        raise RuntimeError(f"FFmpeg command failed with exit code {process.returncode}")
    return process

def get_video_info(video_path):
    """ffprobeを使って動画の情報をJSONで取得する"""
    command = ['ffprobe', '-v', 'quiet', '-print_format', 'json', '-show_format', '-show_streams', video_path]
    result = run_command(command)
    return json.loads(result.stdout)

def parse_color(color_str):
    """'255,0,0' のような文字列を (255, 0, 0) のタプルに変換"""
    try:
        r, g, b = map(int, color_str.split(','))
        return (b, g, r) # OpenCVはBGR順
    except ValueError:
        raise ValueError("色は 'R,G,B' 形式で指定してください (例: '255,0,0')")

def glitch_and_monochrome(frame, target_color_bgr):
    """フレームを激しく乱し、指定された単色に変換する"""
    target_color_np = np.array(target_color_bgr, dtype=np.uint8)
    gray_intensity = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) / 255.0
    mono_frame = np.einsum('ij,k->ijk', gray_intensity, target_color_np).astype(np.uint8)
    
    num_blocks = np.random.randint(15, 50)
    block_height = frame.shape[0] // num_blocks if num_blocks > 0 else frame.shape[0]
    processed_frame = mono_frame.copy()
    for i in range(num_blocks):
        start = i * block_height
        end = (i + 1) * block_height
        shift = np.random.randint(-frame.shape[1] // 3, frame.shape[1] // 3)
        processed_frame[start:end, :] = np.roll(processed_frame[start:end, :], shift, axis=1)
    
    if np.random.rand() < 0.05:
        np.random.shuffle(processed_frame)
    return processed_frame

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

    # ★★★ 修正箇所 ★★★
    # 1. コーデック名を 'libx24' から 'libx264' に修正
    # 2. shell=True をやめ、コマンドをリスト形式でPopenに渡すように変更
    command_encode = [
        'ffmpeg', '-y', '-f', 'rawvideo', '-vcodec', 'rawvideo',
        '-s', f'{width}x{height}', '-pix_fmt', 'bgr24', '-r', str(fps),
        '-i', '-',
        '-i', base_video_path,
        '-c:v', 'libx264', # <-- 修正点１：タイプミスを修正
        '-c:a', 'aac',
        '-map', '0:v:0',
        '-map', '1:a:0?',
        '-pix_fmt', 'yuv420p',
        output_path
    ]
    
    # <-- 修正点２：より安全なコマンド実行方法に変更
    process = subprocess.Popen(command_encode, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    while True:
        ret1, frame1 = cap_base.read()
        ret2, frame2 = cap_overlay.read()
        if not ret1 or not ret2:
            break
        
        frame1_fx = glitch_and_monochrome(frame1, target_color_bgr)
        frame2_fx = glitch_and_monochrome(frame2, target_color_bgr)
        resized_frame2_fx = cv2.resize(frame2_fx, (width, height))
        final_frame = cv2.addWeighted(frame1_fx, 0.5, resized_frame2_fx, 0.5, 0)
        
        try:
            process.stdin.write(final_frame.tobytes())
        except BrokenPipeError:
            print("FFmpegプロセスが予期せず終了しました。コマンド引数を確認してください。")
            break # パイプが壊れたらループを抜ける
        except Exception as e:
            print(f"フレームの書き込み中にエラーが発生しました: {e}")
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
