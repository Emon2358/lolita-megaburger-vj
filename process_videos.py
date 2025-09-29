import sys
import numpy as np
from moviepy.editor import VideoFileClip, CompositeVideoClip
from moviepy.video.fx.all import speedx

def parse_color(color_str):
    """
    '255,0,0' のような文字列を (255, 0, 0) のタプルに変換します。
    """
    try:
        return tuple(map(int, color_str.split(',')))
    except ValueError:
        raise ValueError("色は 'R,G,B' 形式で指定してください (例: '255,0,0')")

def glitch_and_monochrome(clip, target_color):
    """
    動画フレームを激しく乱し、指定された単色に変換するエフェクト。
    """
    target_color_np = np.array(target_color, dtype=np.uint8)

    def effect_func(get_frame, t):
        frame = get_frame(t)
        
        # 1. 輝度を基準にした単色化
        # フレームをグレースケールに変換し、その明るさ(0.0-1.0)を取得
        gray_intensity = np.dot(frame[...,:3], [0.2989, 0.5870, 0.1140]) / 255.0
        # 明るさに応じて、指定された単色を適用
        mono_frame = np.einsum('ij,k->ijk', gray_intensity, target_color_np).astype(np.uint8)

        # 2. 視覚的に極限まで乱す (グリッチエフェクト)
        # フレームを複数の水平ブロックに分割
        num_blocks = np.random.randint(15, 50) # ブロック数はランダムに
        block_height = frame.shape[0] // num_blocks
        
        processed_frame = mono_frame.copy()
        for i in range(num_blocks):
            start = i * block_height
            end = (i + 1) * block_height
            # 各ブロックをランダムな量だけ水平にずらす
            shift = np.random.randint(-frame.shape[1] // 3, frame.shape[1] // 3)
            processed_frame[start:end, :] = np.roll(processed_frame[start:end, :], shift, axis=1)

        # 5%の確率でフレームの行を完全にシャッフルし、さらにカオスにする
        if np.random.rand() < 0.05:
            np.random.shuffle(processed_frame)
            
        return processed_frame

    return clip.fl(effect_func)

def main():
    if len(sys.argv) != 5:
        print("使い方: python process_videos.py <基準動画> <重ねる動画> <出力ファイル> <色(R,G,B)>")
        sys.exit(1)

    base_video_path = sys.argv[1]
    overlay_video_path = sys.argv[2]
    output_path = sys.argv[3]
    color_str = sys.argv[4]

    try:
        target_color = parse_color(color_str)
    except ValueError as e:
        print(f"エラー: {e}")
        sys.exit(1)

    print(f"基準動画: {base_video_path}")
    print(f"重ねる動画: {overlay_video_path}")

    # 動画クリップをロード
    with VideoFileClip(base_video_path) as clip1, VideoFileClip(overlay_video_path) as clip2:
        
        # clip2の再生速度を変更し、長さをclip1に合わせる
        print("動画の長さを調整中...")
        speed_factor = clip2.duration / clip1.duration
        clip2_resized = clip2.fx(speedx, factor=speed_factor)
        
        # 各クリップにエフェクトを適用
        print(f"エフェクトを適用中... (色: {target_color})")
        clip1_fx = glitch_and_monochrome(clip1.copy(), target_color)
        clip2_fx = glitch_and_monochrome(clip2_resized.copy(), target_color)
        
        # 2つの動画を重ねる (clip2を中央に配置)
        print("動画を合成中...")
        final_clip = CompositeVideoClip([clip1_fx, clip2_fx.set_position("center")])
        final_clip = final_clip.set_duration(clip1.duration)

        # 動画をファイルに書き出し
        print(f"動画を '{output_path}' に書き出し中...")
        final_clip.write_videofile(output_path, codec='libx264', audio_codec='aac', threads=4, logger='bar')

    print("処理が完了しました。")

if __name__ == "__main__":
    main()
