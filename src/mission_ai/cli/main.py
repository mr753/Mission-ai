import click
from pathlib import Path
from mission_ai.config.manager import load_config
from mission_ai.mission.parser import MissionParser

@click.group()
def main():
    """Mission AI - Core engine for processing daily mission content."""
    pass

@main.command()
def init():
    """Initialize the project structure and config."""
    click.echo("Initializing project structure...")
    # Logic to create config file if not exists

@main.command()
@click.option('--mission', required=True, type=click.Path(exists=True), help='Path to mission file')
@click.option('--input', required=True, type=click.Path(exists=True), help='Path to input images')
@click.option('--output', required=True, type=click.Path(), help='Path to output directory')
def run(mission, input, output):
    """Run the mission processing."""
    click.echo(f"Processing mission: {mission}")
    mission_data = MissionParser.parse(Path(mission))
    click.echo(f"Mission: {mission_data.content}")
    # Integration logic here

@main.command()
def resume():
    """Resume previous processing."""
    click.echo("Resuming processing...")

@main.command()
def status():
    """Show status of assets."""
    click.echo("Status...")

if __name__ == '__main__':
    main()
