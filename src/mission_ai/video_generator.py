import subprocess
import os

def image_to_video(image_path: str, output_path: str, duration: int = 10, width: int = 1080, height: int = 1920, fps: int = 30) -> bool:
    """
    Generate a video from a single image using FFmpeg subprocess.
    """
    if not os.path.exists(image_path):
        return False
        
    # FFmpeg command
    cmd = [
        "ffmpeg", "-y",
        "-loop", "1",
        "-i", image_path,
        "-c:v", "libx264",
        "-t", str(duration),
        "-pix_fmt", "yuv420p",
        "-vf", f"scale={width}:{height}:force_original_aspect_ratio=decrease,pad={width}:{height}:(ow-iw)/2:(oh-ih)/2,format=yuv420p",
        "-r", str(fps),
        output_path
    ]
    
    try:
        subprocess.run(cmd, check=True, capture_output=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"FFmpeg error: {e.stderr.decode()}")
        return False
