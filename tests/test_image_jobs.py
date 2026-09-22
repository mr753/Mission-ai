import pytest
from pathlib import Path
from mission_ai.models import MissionContext, JobStatus
from mission_ai.jobs.builder import ImageJobBuilder

@pytest.fixture
def mission():
    return MissionContext(
        mission_id="m1", instructions="do X", main_message="msg", 
        key_points=["p1"], platforms=["ig"], max_hashtags=2
    )

def test_deduplication(tmp_path, mission):
    f1 = tmp_path / "a.jpg"
    f2 = tmp_path / "b.jpg"
    f1.write_text("same")
    f2.write_text("same")
    
    builder = ImageJobBuilder()
    jobs = builder.build(mission, [f1, f2])
    
    assert len(jobs) == 1
    assert jobs[0].job_id.startswith("m1_")

def test_different_files(tmp_path, mission):
    f1 = tmp_path / "a.jpg"
    f2 = tmp_path / "b.jpg"
    f1.write_text("content1")
    f2.write_text("content2")
    
    builder = ImageJobBuilder()
    jobs = builder.build(mission, [f1, f2])
    
    assert len(jobs) == 2

def test_sorting(tmp_path, mission):
    # Ensure determinism: sorting by path
    f1 = tmp_path / "b.jpg"
    f2 = tmp_path / "a.jpg"
    f1.write_text("c1")
    f2.write_text("c2")
    
    builder = ImageJobBuilder()
    jobs = builder.build(mission, [f1, f2])
    
    assert jobs[0].source_path.endswith("a.jpg")
    assert jobs[1].source_path.endswith("b.jpg")

def test_unsupported_extension(tmp_path, mission):
    f1 = tmp_path / "a.txt"
    f1.write_text("text")
    
    builder = ImageJobBuilder()
    jobs = builder.build(mission, [f1])
    assert len(jobs) == 0

def test_missing_file(tmp_path, mission):
    f1 = tmp_path / "missing.jpg"
    
    builder = ImageJobBuilder()
    with pytest.raises(ValueError, match="File does not exist"):
        builder.build(mission, [f1])

def test_output_directory(tmp_path, mission):
    f1 = tmp_path / "a.jpg"
    f1.write_text("content")
    
    builder = ImageJobBuilder()
    jobs = builder.build(mission, [f1])
    out_dir = builder.get_output_dir(tmp_path / "out", jobs[0])
    
    # Expected: 001_{short_hash}
    assert out_dir.name.startswith("001_")
    assert len(out_dir.name) == 12 # 4 + 8 chars
