import subprocess
import os

def image_to_video(
    image_path: str,
    output_path: str,
    duration: int = 10,
    width: int = 1080,
    height: int = 1920,
    fps: int = 30,
    ffmpeg_path: str = "ffmpeg",
    music_path: str = None,
) -> bool:
    """
    Generate a video from a single image using FFmpeg subprocess.
    If music_path is provided and exists, mix it as audio (looped to video length,
    faded out at the end). Without music the video is generated silently.
    """
    if not os.path.exists(image_path):
        return False

    # Ensure output directory exists
    out_dir = os.path.dirname(output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    # FFmpeg command
    cmd = [
        ffmpeg_path, "-y",
        "-loop", "1",
        "-i", image_path,
    ]

    has_music = music_path is not None and os.path.exists(music_path)
    if has_music:
        cmd += ["-stream_loop", "-1", "-i", music_path]

    cmd += [
        "-c:v", "libx264",
        "-t", str(duration),
        "-pix_fmt", "yuv420p",
        "-vf", f"scale={width}:{height}:force_original_aspect_ratio=decrease,pad={width}:{height}:(ow-iw)/2:(oh-ih)/2,format=yuv420p",
        "-r", str(fps),
    ]

    if has_music:
        cmd += [
            "-map", "0:v:0",
            "-map", "1:a:0",
            "-c:a", "aac",
            "-b:a", "128k",
            "-shortest",
        ]
        # Fade the music out over the last second.
        fade_start = max(0, duration - 1)
        cmd += ["-af", f"afade=t=out:st={fade_start}:d=1"]

    cmd += [output_path]

    try:
        subprocess.run(cmd, check=True, capture_output=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"FFmpeg error: {e.stderr.decode()}")
        return False
    except FileNotFoundError:
        print(f"FFmpeg executable not found: {ffmpeg_path}")
        return False
