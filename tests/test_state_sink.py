import json
import urllib.request
from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest

from mission_ai.config import AppConfig
from mission_ai.models import MissionContext, JobStatus, ContentPackage, ImageAnalysis, PlatformContent
from mission_ai.jobs.sink import create_state_sink, NullStateSink, SupabaseHttpSink, StateSink
from mission_ai.runner import MissionRunner


class FakeStateSink:
    """Mock StateSink to capture calls for testing."""
    def __init__(self):
        self.missions = []
        self.jobs = []
        self.outputs = []
        self.events = []

    def sink_mission(self, mission: MissionContext) -> None:
        self.missions.append(mission)

    def sink_job_status(
        self,
        job_id: str,
        mission_id: str,
        source_path: str,
        order: int,
        status: JobStatus,
        last_error: str = None
    ) -> None:
        self.jobs.append((job_id, mission_id, source_path, order, status, last_error))

    def sink_output(self, package: ContentPackage) -> None:
        self.outputs.append(package)

    def sink_event(
        self,
        mission_id: str,
        event_type: str,
        status: str = None,
        job_id: str = None,
        payload: dict = None
    ) -> None:
        self.events.append((mission_id, event_type, status, job_id, payload))


class FailingStateSink:
    """StateSink that raises errors to test isolation."""
    def sink_mission(self, mission: MissionContext) -> None:
        raise RuntimeError("Network error")

    def sink_job_status(self, *args, **kwargs) -> None:
        raise RuntimeError("Database connection timed out")

    def sink_output(self, *args, **kwargs) -> None:
        raise RuntimeError("Write failure")

    def sink_event(self, *args, **kwargs) -> None:
        raise RuntimeError("Kafka stream down")


def test_create_state_sink_default():
    # If not enabled, we should get NullStateSink
    cfg = AppConfig(enable_supabase_sink=False)
    sink = create_state_sink(cfg)
    assert isinstance(sink, NullStateSink)


def test_create_state_sink_enabled():
    cfg = AppConfig(
        enable_supabase_sink=True,
        supabase_url="https://xyz.supabase.co",
        supabase_key="secret-key"
    )
    sink = create_state_sink(cfg)
    assert isinstance(sink, SupabaseHttpSink)
    assert sink.supabase_url == "https://xyz.supabase.co"
    assert sink.supabase_key == "secret-key"


@patch("urllib.request.urlopen")
def test_supabase_http_sink_send(mock_urlopen):
    # Mock successful HTTP request
    mock_response = MagicMock()
    mock_response.__enter__.return_value = mock_response
    mock_urlopen.return_value = mock_response

    sink = SupabaseHttpSink("https://xyz.supabase.co", "secret-key")
    
    # Test sink_mission
    mission = MissionContext(
        mission_id="M-1",
        instructions="instr",
        main_message="msg",
        key_points=["k1"],
        platforms=["ig"],
        max_hashtags=5,
        required_hashtags=["#req"],
        image_source_url="https://source.com/img.png"
    )
    sink.sink_mission(mission)
    
    assert mock_urlopen.call_count == 1
    req = mock_urlopen.call_args[0][0]
    assert isinstance(req, urllib.request.Request)
    assert req.full_url == "https://xyz.supabase.co/rest/v1/missions"
    assert req.headers["Apikey"] == "secret-key"
    assert req.headers["Authorization"] == "Bearer secret-key"
    
    payload = json.loads(req.data.decode("utf-8"))
    assert payload["mission_id"] == "M-1"
    assert payload["image_source_url"] == "https://source.com/img.png"


@patch("urllib.request.urlopen")
def test_supabase_http_sink_isolation(mock_urlopen):
    # Mock HTTP failure (e.g., HTTP 500)
    mock_urlopen.side_effect = Exception("Internal Server Error")
    
    sink = SupabaseHttpSink("https://xyz.supabase.co", "secret-key")
    
    # Executing the sink should NOT raise, it should isolate the failure internally
    try:
        sink.sink_event("M-1", "MISSION_STARTED")
    except Exception as e:
        pytest.fail(f"Failure was not isolated: {e}")


class FakeProvider:
    """AI provider stub satisfying the AIProvider protocol."""

    def __init__(self, fail_on: set = None):
        self.analyzed: list = []
        self.captions: list = []
        self._fail_on = fail_on or set()

    def analyze_image(self, image_path, mission_context):
        if image_path in self._fail_on:
            raise RuntimeError(f"AI failure for {image_path}")
        self.analyzed.append(image_path)
        return ImageAnalysis(
            summary=f"summary of {Path(image_path).name}",
            visible_subjects=["subject"],
            visual_context="context",
            relevant_details=["detail"],
        )

    def generate_caption(self, image_analysis, mission_context, platform):
        self.captions.append((image_analysis.summary, platform))
        return f"{platform} caption"


def fake_video_gen(image_path, output_path, **kwargs):
    Path(output_path).write_bytes(b"fake video")
    return True


def test_runner_integration_success(tmp_path):
    # Setup test data
    config = AppConfig(
        ffmpeg_path="ffmpeg",
        music_directory=tmp_path / "music",
        default_duration=15,
        video_width=1080,
        video_height=1920,
        fps=30,
        output_directory=tmp_path / "out"
    )
    
    mission_data = {
        "mission_id": "M-SINK",
        "main_message": "Sink Test",
        "instructions": "",
        "key_points": [],
        "platforms": ["ig"],
        "max_hashtags": 5
    }
    mission_file = tmp_path / "mission.json"
    mission_file.write_text(json.dumps(mission_data))
    
    img_dir = tmp_path / "images"
    img_dir.mkdir()
    from PIL import Image
    Image.new("RGB", (10, 10)).save(img_dir / "one.jpg")
    
    # Setup mocks
    provider = FakeProvider()
    fake_sink = FakeStateSink()
    
    runner = MissionRunner(
        config,
        provider=provider,
        video_generator=fake_video_gen,
        progress=lambda msg: None,
        state_sink=fake_sink
    )
    
    # Run pipeline
    summary = runner.run(str(mission_file), str(img_dir), str(config.output_directory))
    
    # Assert local pipeline succeeded
    assert summary.completed == 1
    
    # Assert state was mirrored correctly
    assert len(fake_sink.missions) == 1
    assert fake_sink.missions[0].mission_id == "M-SINK"
    
    assert len(fake_sink.jobs) == 2 # 1 PROCESSING, 1 COMPLETED
    assert fake_sink.jobs[0][0] == fake_sink.jobs[1][0] # Same job ID
    assert fake_sink.jobs[0][4] == JobStatus.PROCESSING
    assert fake_sink.jobs[1][4] == JobStatus.COMPLETED
    
    assert len(fake_sink.outputs) == 1
    assert fake_sink.outputs[0].status == JobStatus.COMPLETED
    
    # Expected events: MISSION_STARTED, JOB_PROCESSING, JOB_COMPLETED, MISSION_COMPLETED
    assert len(fake_sink.events) == 4
    assert fake_sink.events[0][1] == "MISSION_STARTED"
    assert fake_sink.events[1][1] == "JOB_PROCESSING"
    assert fake_sink.events[2][1] == "JOB_COMPLETED"
    assert fake_sink.events[3][1] == "MISSION_COMPLETED"


def test_runner_integration_failure_isolation(tmp_path):
    # Setup runner with FailingStateSink
    config = AppConfig(
        ffmpeg_path="ffmpeg",
        music_directory=tmp_path / "music",
        default_duration=15,
        video_width=1080,
        video_height=1920,
        fps=30,
        output_directory=tmp_path / "out"
    )
    
    mission_data = {
        "mission_id": "M-FAIL-SINK",
        "main_message": "Sink Fail Test",
        "instructions": "",
        "key_points": [],
        "platforms": ["ig"],
        "max_hashtags": 5
    }
    mission_file = tmp_path / "mission.json"
    mission_file.write_text(json.dumps(mission_data))
    
    img_dir = tmp_path / "images"
    img_dir.mkdir()
    from PIL import Image
    Image.new("RGB", (10, 10)).save(img_dir / "one.jpg")
    
    provider = FakeProvider()
    failing_sink = FailingStateSink()
    
    runner = MissionRunner(
        config,
        provider=provider,
        video_generator=fake_video_gen,
        progress=lambda msg: None,
        state_sink=failing_sink
    )
    
    # Run pipeline - should run completely fine despite the failing sink
    try:
        summary = runner.run(str(mission_file), str(img_dir), str(config.output_directory))
    except Exception as e:
        pytest.fail(f"Pipeline crashed due to sink error: {e}")
        
    assert summary.completed == 1
