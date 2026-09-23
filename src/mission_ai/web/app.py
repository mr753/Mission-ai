import json
import threading
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse

from mission_ai.config import AppConfig
from mission_ai.runner import MissionRunner


HTML = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Mission AI</title>
<style>
body{font-family:system-ui,sans-serif;max-width:900px;margin:40px auto;padding:0 16px;background:#111;color:#eee}
.card{background:#1b1b1b;border:1px solid #333;border-radius:14px;padding:20px;margin:16px 0}
input,button{padding:10px;margin:6px 0;width:100%;box-sizing:border-box}
button{cursor:pointer}
pre{white-space:pre-wrap}
</style>
</head>
<body>
<h1>Mission AI</h1>
<div class="card">
<h2>Run Mission</h2>
<form id="form">
<label>Mission JSON <input type="file" name="mission" accept=".json" required></label>
<label>Images <input type="file" name="images" accept=".jpg,.jpeg,.png,.webp" multiple required></label>
<button type="submit">Start processing</button>
</form>
<pre id="result"></pre>
</div>
<div class="card"><h2>Missions</h2><pre id="missions">Loading...</pre></div>
<script>
async function refresh(){
 const r=await fetch('/api/missions'); document.getElementById('missions').textContent=JSON.stringify(await r.json(),null,2);
}
document.getElementById('form').onsubmit=async e=>{
 e.preventDefault();
 const r=await fetch('/api/missions',{method:'POST',body:new FormData(e.target)});
 document.getElementById('result').textContent=JSON.stringify(await r.json(),null,2);
 refresh();
};
refresh(); setInterval(refresh,3000);
</script>
</body></html>"""


def create_app(config: AppConfig | None = None) -> FastAPI:
    config = config or AppConfig()
    app = FastAPI(title="Mission AI", version="0.1.0")
    output_root = Path(config.output_directory)
    upload_root = output_root / ".web_uploads"
    upload_root.mkdir(parents=True, exist_ok=True)

    @app.get("/", response_class=HTMLResponse)
    def index():
        return HTML

    @app.get("/api/health")
    def health():
        return {"status": "ok", "service": "mission-ai"}

    @app.get("/api/missions")
    def missions():
        output_root.mkdir(parents=True, exist_ok=True)
        result = []
        for p in sorted(output_root.iterdir()):
            if not p.is_dir() or p.name.startswith("."):
                continue
            checkpoint = p / "checkpoint.json"
            state = {}
            if checkpoint.exists():
                try:
                    state = json.loads(checkpoint.read_text()).get("state", {})
                except (OSError, ValueError):
                    state = {}
            counts = {}
            for value in state.values():
                counts[value] = counts.get(value, 0) + 1
            result.append({"mission_id": p.name, "jobs": counts})
        return result

    @app.get("/api/missions/{mission_id}")
    def mission_status(mission_id: str):
        mission_dir = output_root / mission_id
        if not mission_dir.is_dir():
            raise HTTPException(404, "Mission not found")
        checkpoint = mission_dir / "checkpoint.json"
        state = {}
        if checkpoint.exists():
            try:
                state = json.loads(checkpoint.read_text()).get("state", {})
            except (OSError, ValueError):
                pass
        return {"mission_id": mission_id, "state": state}

    @app.post("/api/missions")
    async def create_mission(
        mission: UploadFile = File(...),
        images: list[UploadFile] = File(...),
    ):
        if not mission.filename or not mission.filename.lower().endswith(".json"):
            raise HTTPException(400, "mission must be a JSON file")
        if not images:
            raise HTTPException(400, "at least one image is required")

        raw = await mission.read()
        try:
            mission_data = json.loads(raw.decode("utf-8"))
            mission_id = str(mission_data["mission_id"])
        except (UnicodeDecodeError, json.JSONDecodeError, KeyError, TypeError):
            raise HTTPException(400, "invalid mission JSON")

        job_root = upload_root / mission_id
        image_root = job_root / "images"
        image_root.mkdir(parents=True, exist_ok=True)
        mission_path = job_root / "mission.json"
        mission_path.write_bytes(raw)

        for item in images:
            if not item.filename:
                continue
            suffix = Path(item.filename).suffix.lower()
            if suffix not in {".jpg", ".jpeg", ".png", ".webp"}:
                continue
            (image_root / Path(item.filename).name).write_bytes(await item.read())

        def worker():
            try:
                MissionRunner(config, progress=lambda _: None).run(
                    str(mission_path), str(image_root), str(output_root)
                )
            except Exception:
                # The runner records per-job failures; unexpected mission-level
                # errors remain visible through the API/filesystem.
                pass

        threading.Thread(target=worker, daemon=True).start()
        return {"status": "started", "mission_id": mission_id}

    @app.get("/api/missions/{mission_id}/files")
    def files(mission_id: str):
        mission_dir = output_root / mission_id
        if not mission_dir.is_dir():
            raise HTTPException(404, "Mission not found")
        return {"files": [str(p.relative_to(mission_dir)) for p in mission_dir.rglob("*") if p.is_file()]}

    @app.get("/api/missions/{mission_id}/files/{file_path:path}")
    def download_file(mission_id: str, file_path: str):
        mission_dir = (output_root / mission_id).resolve()
        target = (mission_dir / file_path).resolve()
        if mission_dir not in target.parents or not target.is_file():
            raise HTTPException(404, "File not found")
        return FileResponse(target)

    return app
