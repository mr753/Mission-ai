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
<title>Mission AI Dashboard</title>
<style>
body{font-family:system-ui,sans-serif;max-width:1000px;margin:30px auto;padding:0 16px;background:#121212;color:#e0e0e0}
header{display:flex;justify-content:space-between;align-items:center;border-bottom:1px solid #333;padding-bottom:12px;margin-bottom:20px}
h1,h2,h3{color:#fff}
.card{background:#1e1e1e;border:1px solid #333;border-radius:12px;padding:20px;margin:16px 0;box-shadow:0 4px 6px rgba(0,0,0,0.3)}
input,button{padding:10px;margin:6px 0;width:100%;box-sizing:border-box;background:#2a2a2a;color:#fff;border:1px solid #444;border-radius:6px}
button{background:#0066cc;cursor:pointer;font-weight:600}
button:hover{background:#0052a3}
pre{white-space:pre-wrap;background:#151515;padding:12px;border-radius:6px;border:1px solid #282828;overflow-x:auto}
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(280px,1fr));gap:16px}
.post-card{background:#222;border:1px solid #444;border-radius:10px;padding:16px;margin:10px 0}
.badge{display:inline-block;padding:4px 8px;border-radius:4px;font-size:12px;font-weight:700}
.COMPLETED{background:#1b5e20;color:#c8e6c9}
.FAILED{background:#b71c1c;color:#ffcdd2}
.PENDING{background:#f57f17;color:#fffde7}
.badge-muted{background:#424242;color:#e0e0e0}
.section-title{font-size:14px;font-weight:bold;color:#aaa;margin-top:10px;margin-bottom:4px}
video{width:100%;max-height:300px;background:#000;border-radius:6px;margin-top:8px}
ul{margin:0;padding-left:20px}
</style>
</head>
<body>
<header>
<h1>Mission AI Dashboard</h1>
<div><a href="#run" style="color:#66b2ff;text-decoration:none">Run Mission</a></div>
</header>

<div class="card" id="run">
<h2>Run New Mission</h2>
<form id="form">
<label>Mission JSON <input type="file" name="mission" accept=".json" required></label>
<label>Images <input type="file" name="images" accept=".jpg,.jpeg,.png,.webp" multiple required></label>
<button type="submit">Start Processing</button>
</form>
<pre id="result" style="display:none"></pre>
</div>

<div class="card">
<h2>Missions</h2>
<div id="missions-list">Loading missions...</div>
</div>

<div class="card" id="mission-detail-card" style="display:none">
<h2 id="detail-title">Mission Details</h2>
<div id="posts-container"></div>
</div>

<script>
async function loadMissions(){
 try{
  const res=await fetch('/api/missions');
  const missions=await res.json();
  const container=document.getElementById('missions-list');
  if(!missions.length){container.innerHTML='<p>No missions found.</p>';return;}
  let html='<div class="grid">';
  for(const m of missions){
   const counts=m.jobs||{};
   const comp=counts['COMPLETED']||0;
   const fail=counts['FAILED']||0;
   const total=Object.values(counts).reduce((a,b)=>a+b,0);
   html+=`<div class="post-card" style="cursor:pointer" onclick="loadMissionDetail('${m.mission_id}')">
    <h3>${m.mission_id}</h3>
    <p>Total Jobs: ${total}</p>
    <p><span class="badge COMPLETED">Completed: ${comp}</span> <span class="badge FAILED">Failed: ${fail}</span></p>
   </div>`;
  }
  html+='</div>';
  container.innerHTML=html;
 }catch(e){
  document.getElementById('missions-list').textContent='Error loading missions: '+e.message;
 }
}

async function loadMissionDetail(missionId){
 document.getElementById('mission-detail-card').style.display='block';
 document.getElementById('detail-title').textContent='Mission: '+missionId;
 const container=document.getElementById('posts-container');
 container.innerHTML='Loading posts...';
 try{
  const res=await fetch(`/api/missions/${missionId}/posts`);
  const data=await res.json();
  if(!data.posts || !data.posts.length){
   container.innerHTML='<p>No posts/jobs found for this mission.</p>';
   return;
  }
  let html='';
  for(const p of data.posts){
   const statusClass=p.status||'PENDING';
   html+=`<div class="post-card">
    <div style="display:flex;justify-content:space-between;align-items:center">
     <strong>Image / Job ID: ${p.image_id}</strong>
     <span class="badge ${statusClass}">${p.status}</span>
    </div>`;
   if(p.error){
    html+=`<p style="color:#ff8a80">Error: ${escapeHtml(p.error)}</p>`;
   }
   if(p.video_path){
    const relVideo=p.video_path.replace(/\\\\/g,'/').split('/videos/').pop();
    html+=`<div class="section-title">Video:</div>
    <video controls src="/api/missions/${missionId}/files/videos/${relVideo}"></video>
    <p><a href="/api/missions/${missionId}/files/videos/${relVideo}" download style="color:#66b2ff">Download MP4</a></p>`;
   } else {
    html+=`<div class="section-title">Video:</div><p style="color:#757575">Not generated / unavailable</p>`;
   }
   if(p.voiceover_script && p.voiceover_script.script_text){
    html+=`<div class="section-title">Voice-over Script:</div><pre>${escapeHtml(p.voiceover_script.script_text)}</pre>`;
   }
   if(p.captions && p.captions.length){
    html+=`<div class="section-title">Captions & Hashtags:</div>`;
    for(const c of p.captions){
     html+=`<div style="background:#151515;padding:8px;margin-top:4px;border-radius:4px">
      <span class="badge badge-muted">${c.platform}</span>
      <p style="margin:6px 0">${escapeHtml(c.caption)}</p>
      <p style="color:#4fc3f7;font-size:13px;margin:4px 0">${(c.hashtags||[]).join(' ')}</p>
     </div>`;
    }
   }
   html+=`</div>`;
  }
  container.innerHTML=html;
 }catch(e){
  container.innerHTML='Error loading mission posts: '+e.message;
 }
}

function escapeHtml(text){
 if(!text) return '';
 return text.replace(/&/g,"&amp;").replace(/</g,"&lt;").replace(/>/g,"&gt;");
}

document.getElementById('form').onsubmit=async e=>{
 e.preventDefault();
 const res=await fetch('/api/missions',{method:'POST',body:new FormData(e.target)});
 const json=await res.json();
 const el=document.getElementById('result');
 el.style.display='block';
 el.textContent=JSON.stringify(json,null,2);
 loadMissions();
};
loadMissions(); setInterval(loadMissions,4000);
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
        if not output_root.is_dir():
            return result
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

    @app.get("/api/missions/{mission_id}/posts")
    def mission_posts(mission_id: str):
        mission_dir = (output_root / mission_id).resolve()
        if not mission_dir.is_dir():
            raise HTTPException(404, "Mission not found")

        checkpoint = mission_dir / "checkpoint.json"
        state = {}
        if checkpoint.exists():
            try:
                state = json.loads(checkpoint.read_text()).get("state", {})
            except (OSError, ValueError):
                pass

        metadata_dir = mission_dir / "metadata"
        posts = []
        seen_ids = set()

        if metadata_dir.is_dir():
            for meta_file in sorted(metadata_dir.glob("*.json")):
                try:
                    data = json.loads(meta_file.read_text(encoding="utf-8"))
                    image_id = data.get("image_id") or meta_file.stem
                    seen_ids.add(image_id)
                    if image_id in state:
                        data["status"] = state[image_id]
                    posts.append(data)
                except (OSError, ValueError):
                    pass

        for job_id, job_status in state.items():
            if job_id not in seen_ids:
                posts.append({
                    "image_id": job_id,
                    "status": job_status,
                    "error": "Job recorded in checkpoint without content package metadata."
                })

        return {"mission_id": mission_id, "posts": posts}

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
