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
    audio_path: str = None,
) -> bool:
    """
    Generate a video from a single static image and TTS audio/music using FFmpeg subprocess.
    Video duration follows audio duration when audio_path or music_path is provided (via -shortest).
    """
    if not os.path.exists(image_path):
        return False

    effective_audio = audio_path or music_path
    if effective_audio and not os.path.exists(effective_audio):
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

    has_audio = effective_audio is not None and os.path.exists(effective_audio)
    if has_audio:
        cmd += ["-i", effective_audio]

    cmd += [
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-vf", f"scale={width}:{height}:force_original_aspect_ratio=decrease,pad={width}:{height}:(ow-iw)/2:(oh-ih)/2,format=yuv420p",
        "-r", str(fps),
    ]

    if has_audio:
        cmd += [
            "-map", "0:v:0",
            "-map", "1:a:0",
            "-c:a", "aac",
            "-b:a", "128k",
            "-shortest",
        ]
    else:
        cmd += ["-t", str(duration)]

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
