import json
import sys
from pathlib import Path

import click
from dotenv import load_dotenv

load_dotenv()

from mission_ai.config import load_config
from mission_ai.jobs.checkpoint import CheckpointManager
from mission_ai.mission.parser import MissionParser


@click.group()
def main():
    """Mission AI - Core engine for processing daily mission content."""
    pass


def _build_runner(progress):
    """Build a MissionRunner from the current configuration."""
    from mission_ai.runner import MissionRunner
    config = load_config()
    return MissionRunner(config, progress=progress)


def _execute(mission: str, input: str, output: str):
    """Shared run/resume execution."""
    runner = _build_runner(progress=lambda msg: click.echo(msg))
    summary = runner.run(mission, input, output)
    click.echo(summary.describe())
    return summary


@main.command()
@click.option('--mission', required=True, type=click.Path(exists=True), help='Path to mission file')
@click.option('--input', required=True, type=click.Path(exists=True), help='Path to input images')
@click.option('--output', required=True, type=click.Path(), help='Path to output directory')
def run(mission, input, output):
    """Run the mission processing pipeline."""
    try:
        _execute(mission, input, output)
    except (ValueError, FileNotFoundError, RuntimeError, OSError) as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


@main.command()
@click.option('--mission', required=True, type=click.Path(exists=True), help='Path to mission file')
@click.option('--input', required=True, type=click.Path(exists=True), help='Path to input images')
@click.option('--output', required=True, type=click.Path(), help='Path to output directory')
def resume(mission, input, output):
    """Resume previous processing (completed jobs are skipped)."""
    try:
        _execute(mission, input, output)
    except (ValueError, FileNotFoundError, RuntimeError, OSError) as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


@main.command()
@click.option("--host", default="127.0.0.1", show_default=True, help="Web server host")
@click.option("--port", default=8000, show_default=True, type=int, help="Web server port")
def web(host, port):
    """Start the local Mission AI web dashboard."""
    try:
        import uvicorn
        from mission_ai.web.app import create_app
    except ImportError as e:
        raise click.ClickException(
            "Web dependencies are missing. Install with: pip install -e '.[web]'"
        ) from e
    click.echo(f"Mission AI web: http://{host}:{port}")
    uvicorn.run(create_app(load_config()), host=host, port=port, log_level="info")


@main.command()
@click.option('--mission', required=True, type=click.Path(exists=True), help='Path to mission file')
@click.option('--output', required=True, type=click.Path(), help='Path to output directory')
def status(mission, output):
    """Show checkpoint/output status for a mission."""
    try:
        mission_ctx = MissionParser.parse_json_file(mission)
    except ValueError as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)

    mission_dir = Path(output) / mission_ctx.mission_id
    checkpoint_file = mission_dir / "checkpoint.json"

    click.echo(f"Mission: {mission_ctx.mission_id}")
    click.echo(f"Main message: {mission_ctx.main_message}")
    click.echo(f"Output: {mission_dir}")

    if not mission_dir.exists():
        click.echo("Status: no output yet (not processed)")
        return

    counts = {}
    total = 0
    if checkpoint_file.exists():
        try:
            checkpoint = CheckpointManager(checkpoint_file)
            for value in checkpoint.state.values():
                counts[value] = counts.get(value, 0) + 1
                total += 1
        except (OSError, ValueError):
            click.echo("Warning: checkpoint file unreadable", err=True)

    click.echo(f"Checkpoint: {total} recorded job(s)")
    for status_name in ("COMPLETED", "FAILED", "PROCESSING", "PENDING"):
        if counts.get(status_name):
            click.echo(f"  {status_name}: {counts[status_name]}")

    for sub in ("videos", "captions", "metadata"):
        d = mission_dir / sub
        n = len(list(d.glob("*"))) if d.exists() else 0
        click.echo(f"{sub}: {n} file(s)")

    if total and counts.get("COMPLETED") == total:
        click.echo("Status: all jobs completed")
    elif counts.get("FAILED"):
        click.echo("Status: failures present (run again to retry failed jobs)")
    else:
        click.echo("Status: in progress or not started")


if __name__ == '__main__':
    main()
