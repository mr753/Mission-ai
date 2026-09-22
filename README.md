# Mission AI

Core engine for processing daily mission content into video/caption assets.

## Phase 1 Architecture

- **CLI:** Entry point using `click`.
- **Mission Parser:** Extracts key points and hashtags.
- **AI Engine:** Modular interface for Gemini/Ollama.
- **Caption Engine:** Platform-specific caption generation.
- **Video Engine:** FFmpeg wrapper (portable).
- **Sources:** Input abstraction (Local/GDRIVE stub).
- **Jobs/State:** Simple file-based persistence for resume capability.

## Documentation
- [Termux Setup](docs/TERMUX.md)
- [Windows Setup](docs/WINDOWS.md)

## Development
- Install dependencies: `pip install -e .`
- Run tests: `pytest`
