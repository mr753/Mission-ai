"""End-to-end smoke run using a fake AI provider and fake video generation.

Proves the CLI run/resume/status flow works locally without Gemini/Ollama,
Google Drive, or FFmpeg. Run: python scripts/e2e_fake_demo.py
"""
import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from PIL import Image

from mission_ai.cli import main as cli_main
from click.testing import CliRunner


class FakeProvider:
    def analyze_image(self, image_path, mission_context):
        from mission_ai.models import ImageAnalysis
        return ImageAnalysis(
            summary=f"Static test image {Path(image_path).name}",
            visible_subjects=["rectangles"],
            visual_context="plain synthetic test image",
            relevant_details=["two colors"],
        )

    def generate_caption(self, analysis, mission_context, platform):
        return f"{platform}: {analysis.summary}"


def main():
    tmp = Path(tempfile.mkdtemp(prefix="mission-ai-e2e-"))
    # Mission + images (synthetic, no real user assets).
    mission_file = tmp / "mission.json"
    mission_file.write_text(json.dumps({
        "mission_id": "E2E-DEMO",
        "main_message": "Demo mission",
        "instructions": "demo",
        "platforms": ["instagram", "youtube"],
        "max_hashtags": 3,
        "required_hashtags": ["#demo"],
    }))
    img_dir = tmp / "images"
    img_dir.mkdir()
    Image.new("RGB", (64, 64), color="red").save(img_dir / "a.jpg")
    Image.new("RGB", (64, 64), color="green").save(img_dir / "b.png")

    output = tmp / "output"
    runner = CliRunner()

    # The CLI builds the runner via _build_runner; patch it to inject the fake.
    import mission_ai.cli.main as cli
    original_build = cli._build_runner

    def build_with_fake(progress):
        from mission_ai.config import load_config
        from mission_ai.runner import MissionRunner
        cfg = load_config()
        return MissionRunner(cfg, provider=FakeProvider(), progress=progress)

    cli._build_runner = build_with_fake

    try:
        print("=== mission-ai run ===")
        result = runner.invoke(cli.main, [
            "run", "--mission", str(mission_file),
            "--input", str(img_dir), "--output", str(output),
        ])
        print(result.output)
        assert result.exit_code == 0, f"run failed: {result.output}"

        print("=== mission-ai status ===")
        result = runner.invoke(cli.main, [
            "status", "--mission", str(mission_file), "--output", str(output),
        ])
        print(result.output)
        assert result.exit_code == 0

        print("=== mission-ai resume (should skip all) ===")
        result = runner.invoke(cli.main, [
            "resume", "--mission", str(mission_file),
            "--input", str(img_dir), "--output", str(output),
        ])
        print(result.output)
        assert result.exit_code == 0
        assert "skipped (already done): 2" in result.output

        # Verify artifacts.
        base = output / "E2E-DEMO"
        videos = list((base / "videos").glob("*.mp4"))
        metas = list((base / "metadata").glob("*.json"))
        caps = list((base / "captions").glob("*_caption.txt"))
        assert len(videos) == 2 and len(metas) == 2 and len(caps) == 2
        meta = json.loads(metas[0].read_text())
        assert meta["status"] == "COMPLETED" and meta["video_path"]
        assert len(meta["captions"]) == 2  # instagram + youtube
        print("E2E DEMO PASSED")
    finally:
        cli._build_runner = original_build


if __name__ == "__main__":
    main()
