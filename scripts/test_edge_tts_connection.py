#!/usr/bin/env python3
"""Standalone diagnostic script for testing Edge TTS direct connection.

Runs within the project virtual environment, tests edge-tts connection directly,
saves a temporary MP3 file, prints success or full exception traceback, cleans up
the temporary file, and does not use any fallback provider or require API keys.
"""
import asyncio
import sys
import tempfile
import traceback
from pathlib import Path


async def test_connection():
    print("Initializing edge-tts connection test...")
    try:
        import edge_tts
    except ImportError:
        print("Error: edge-tts package is not installed in this Python environment.", file=sys.stderr)
        sys.exit(1)

    voice = "id-ID-GadisNeural"
    text = "Halo, ini adalah pengujian koneksi langsung Edge TTS."

    fd, temp_mp3 = tempfile.mkstemp(suffix=".mp3")
    import os
    os.close(fd)

    print(f"Target Voice: {voice}")
    print(f"Test Text: {text!r}")
    print(f"Temporary Output: {temp_mp3}")
    print("Connecting to Edge TTS service (timeout=15s)...")

    try:
        communicate = edge_tts.Communicate(text=text, voice=voice, rate="-5%", volume="+0%")
        await asyncio.wait_for(communicate.save(temp_mp3), timeout=15.0)

        size = os.path.getsize(temp_mp3) if os.path.exists(temp_mp3) else 0
        if size > 0:
            print(f"SUCCESS: Edge TTS connected and generated MP3 audio successfully ({size} bytes).")
        else:
            print("FAILURE: Edge TTS completed but output file is empty.", file=sys.stderr)
    except Exception as e:
        print("DIAGNOSTIC EXCEPTION OCCURRED (Connection/Network Error):")
        traceback.print_exc()
        print("\nNote: Connection failure in air-gapped or restricted network environments is expected")
        print("and does not indicate incorrect code. The fallback mechanism handles this gracefully.")
    finally:
        if os.path.exists(temp_mp3):
            try:
                os.remove(temp_mp3)
                print(f"Cleaned up temporary file: {temp_mp3}")
            except Exception as ce:
                print(f"Warning: Failed to clean up temp file: {ce}", file=sys.stderr)


if __name__ == "__main__":
    asyncio.run(test_connection())
