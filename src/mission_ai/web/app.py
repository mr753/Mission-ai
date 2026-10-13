import json
import logging
import threading
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse

from mission_ai.config import AppConfig
from mission_ai.runner import MissionRunner
from mission_ai.upload import AccountStore

logger = logging.getLogger("mission_ai.web.app")


HTML = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Mission AI Dashboard</title>
<style>
:root{--bg:#121212;--card:#1e1e1e;--border:#333;--text:#e0e0e0;--text-bright:#fff;--accent:#0066cc;--accent-hover:#0052a3;--success-bg:#1b5e20;--success-text:#c8e6c9;--fail-bg:#b71c1c;--fail-text:#ffcdd2;--pending-bg:#f57f17;--pending-text:#fffde7}
body{font-family:system-ui,-apple-system,sans-serif;max-width:1100px;margin:0 auto;padding:12px;background:var(--bg);color:var(--text);line-height:1.5}
header{display:flex;flex-wrap:wrap;justify-content:space-between;align-items:center;border-bottom:1px solid var(--border);padding-bottom:12px;margin-bottom:16px;gap:10px}
h1,h2,h3{color:var(--text-bright);margin-top:0}
nav{display:flex;flex-wrap:wrap;gap:8px;margin-bottom:16px}
nav button{background:#222;color:var(--text);border:1px solid var(--border);padding:8px 14px;border-radius:6px;cursor:pointer;font-weight:600;width:auto;margin:0}
nav button.active{background:var(--accent);color:#fff;border-color:var(--accent)}
.tab-content{display:none}
.tab-content.active{display:block}
.card{background:var(--card);border:1px solid var(--border);border-radius:12px;padding:16px;margin:16px 0;box-shadow:0 4px 6px rgba(0,0,0,0.3)}
input,textarea,button{padding:10px;margin:6px 0;width:100%;box-sizing:border-box;background:#2a2a2a;color:var(--text-bright);border:1px solid #444;border-radius:6px;font-family:inherit}
button{background:var(--accent);cursor:pointer;font-weight:600;border:none}
button:hover{background:var(--accent-hover)}
button:disabled{background:#555;cursor:not-allowed}
pre{white-space:pre-wrap;background:#151515;padding:12px;border-radius:6px;border:1px solid #282828;overflow-x:auto}
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(260px,1fr));gap:14px}
.post-card{background:#222;border:1px solid #444;border-radius:10px;padding:14px;margin:10px 0}
.badge{display:inline-block;padding:4px 8px;border-radius:4px;font-size:12px;font-weight:700}
.COMPLETED{background:var(--success-bg);color:var(--success-text)}
.FAILED{background:var(--fail-bg);color:var(--fail-text)}
.PENDING,.PROCESSING{background:var(--pending-bg);color:var(--pending-text)}
.badge-muted{background:#424242;color:var(--text)}
.section-title{font-size:14px;font-weight:bold;color:#aaa;margin-top:10px;margin-bottom:4px}
video{width:100%;max-height:320px;background:#000;border-radius:6px;margin-top:8px}
.status-box{background:#151515;border:1px solid var(--border);padding:12px;border-radius:6px;margin-top:10px}
@media(min-width:768px){body{padding:24px}.card{padding:24px}}
</style>
</head>
<body>
<header>
<div>
<h1 style="margin:0">Mission AI</h1>
<small style="color:#aaa">Autonomous Video & Content Production Engine</small>
</div>
<div id="system-status" style="font-size:13px;color:#81c784">System: Connected</div>
</header>

<nav>
<button onclick="switchTab('dashboard')" id="btn-dashboard" class="active">Dashboard</button>
<button onclick="switchTab('missions')" id="btn-missions">Missions / History</button>
<button onclick="switchTab('accounts')" id="btn-accounts">Social Accounts</button>
<button onclick="switchTab('content')" id="btn-content">Content Library</button>
<button onclick="switchTab('activity')" id="btn-activity">Activity Report</button>
</nav>

<!-- TAB 1: DASHBOARD -->
<div id="tab-dashboard" class="tab-content active">
<div class="card" id="run">
<h2>Run New Mission</h2>
<p style="color:#aaa;font-size:13px">Paste your mission instructions below. Include a Google Drive folder link to source input images.</p>
<form id="form">
<label style="font-weight:600;font-size:14px">Mission Text
<textarea name="mission_text" rows="6" placeholder="Example:
Create daily motivation reels from this folder.
Drive Link: https://drive.google.com/drive/folders/abc123xyz" required></textarea>
</label>
<button type="submit" id="submit-btn">Start Processing</button>
</form>
<div id="result-box" class="status-box" style="display:none"></div>
</div>

<div class="card">
<h2>Recent Missions & Status</h2>
<div id="dashboard-missions-summary">Loading summary...</div>
</div>
</div>

<!-- TAB 2: MISSIONS / HISTORY -->
<div id="tab-missions" class="tab-content">
<div class="card">
<h2>Mission History</h2>
<p style="color:#aaa;font-size:13px">Select a mission below to view its full processing status, videos, scripts, and captions, or delete completed missions.</p>
<div id="missions-list">Loading missions...</div>
</div>

<div class="card" id="mission-detail-card" style="display:none">
<div id="detail-title-bar" style="display:flex;justify-content:space-between;align-items:center;margin-bottom:12px">
<h2 id="detail-title" style="margin:0">Mission Output</h2>
<button id="delete-mission-btn" onclick="" style="background:#b71c1c;width:auto;padding:6px 12px;font-size:12px">Delete Mission</button>
</div>
<div id="posts-container"><p>Loading mission output...</p></div>
<div id="activity-log-container" style="margin-top:16px"></div>
</div>
</div>

<!-- TAB 3: SOCIAL ACCOUNTS -->
<div id="tab-accounts" class="tab-content">
<div class="card">
<h2>Social Media Account Management</h2>
<p style="color:#aaa;font-size:13px;margin-bottom:12px">Manage platform connections and check automation readiness. Note: Real automated publishing requires OAuth configuration (currently operating in manual download / simulation mode).</p>
<div id="accounts-list">Loading accounts...</div>
</div>
</div>

<!-- TAB 4: CONTENT LIBRARY -->
<div id="tab-content" class="tab-content">
<div class="card">
<h2>Content Library</h2>
<p style="color:#aaa;font-size:13px">Generated videos, voice-over scripts, captions, and hashtags ready for manual download or posting.</p>
<div id="content-library-container">Loading content library...</div>
</div>
</div>

<!-- TAB 5: ACTIVITY REPORT -->
<div id="tab-activity" class="tab-content">
<div class="card">
<h2>Activity Report & Status Log</h2>
<p style="color:#aaa;font-size:13px">Summary of processed missions, job success rates, and operational status.</p>
<div id="activity-report-container">Loading activity report...</div>
</div>
</div>

<script>
function switchTab(tabName){
 document.querySelectorAll('.tab-content').forEach(el=>el.classList.remove('active'));
 document.querySelectorAll('nav button').forEach(el=>el.classList.remove('active'));
 document.getElementById('tab-'+tabName).classList.add('active');
 document.getElementById('btn-'+tabName).classList.add('active');
 if(tabName==='content'){loadContentLibrary();}
 if(tabName==='activity'){loadActivityReport();}
}

async function loadSystemHealth(){
 try{
  const res=await fetch('/api/health');
  const data=await res.json();
  if(data.status==='ok'){
   document.getElementById('system-status').textContent='System: Online ('+data.service+')';
  }
 }catch(e){
  document.getElementById('system-status').textContent='System: Offline / Error';
  document.getElementById('system-status').style.color='#ff8a80';
 }
}

async function loadMissions(){
 try{
  const res=await fetch('/api/missions');
  const missions=await res.json();

  // Render dashboard summary
  const sumContainer=document.getElementById('dashboard-missions-summary');
  if(!missions.length){
   sumContainer.innerHTML='<p>No missions found. Start a new mission from the Dashboard tab.</p>';
  }else{
   let sumHtml='<div class="grid">';
   for(const m of missions){
    const counts=m.jobs||{};
    const comp=counts['COMPLETED']||0;
    const fail=counts['FAILED']||0;
    const total=Object.values(counts).reduce((a,b)=>a+b,0);
    const isDone = total > 0 && comp === total;
    const statusBadge = isDone ? '<span class="badge COMPLETED">Completed</span>' : (fail > 0 ? '<span class="badge FAILED">Failures Present</span>' : '<span class="badge PROCESSING">Processing / Pending</span>');
    sumHtml+=`<div class="post-card">
     <div style="cursor:pointer" onclick="switchTab('missions'); loadMissionDetail('${escapeJs(m.mission_id)}');">
      <h3>${escapeHtml(m.mission_id)}</h3>
      <p>Jobs: ${total} | ${statusBadge}</p>
      <p style="font-size:13px;color:#aaa">Completed: ${comp} | Failed: ${fail}</p>
     </div>
     <button onclick="deleteMission('${escapeJs(m.mission_id)}')" style="background:#b71c1c;margin-top:8px;padding:4px 8px;font-size:12px;width:auto">Delete Mission</button>
    </div>`;
   }
   sumHtml+='</div>';
   sumContainer.innerHTML=sumHtml;
  }

  // Render missions list
  const container=document.getElementById('missions-list');
  if(!missions.length){
   container.innerHTML='<p>No missions found.</p>';
   return;
  }
  let html='<div class="grid">';
  for(const m of missions){
   const counts=m.jobs||{};
   const comp=counts['COMPLETED']||0;
   const fail=counts['FAILED']||0;
   const total=Object.values(counts).reduce((a,b)=>a+b,0);
   html+=`<div class="post-card">
    <div style="cursor:pointer" onclick="loadMissionDetail('${escapeJs(m.mission_id)}')">
     <h3>${escapeHtml(m.mission_id)}</h3>
     <p>Total Jobs: ${total}</p>
     <p><span class="badge COMPLETED">Completed: ${comp}</span> <span class="badge FAILED">Failed: ${fail}</span></p>
    </div>
    <button onclick="deleteMission('${escapeJs(m.mission_id)}')" style="background:#b71c1c;margin-top:8px;padding:4px 8px;font-size:12px;width:auto">Delete Mission</button>
   </div>`;
  }
  html+='</div>';
  container.innerHTML=html;

  if(missions.length === 1 && !document.getElementById('mission-detail-card').dataset.loaded){
   loadMissionDetail(missions[0].mission_id);
  }
 }catch(e){
  document.getElementById('missions-list').textContent='Error loading missions: '+e.message;
 }
}

async function deleteMission(missionId){
 if(!confirm('Are you sure you want to delete mission ' + missionId + '? This will permanently remove its output files and metadata.')) return;
 try{
  const res = await fetch(`/api/missions/${missionId}`, {method: 'DELETE'});
  const json = await res.json();
  if(res.ok){
   alert('Mission ' + missionId + ' deleted successfully.');
   document.getElementById('mission-detail-card').style.display = 'none';
   loadMissions();
  }else{
   alert('Failed to delete mission: ' + (json.detail || 'Unknown error'));
  }
 }catch(e){
  alert('Error deleting mission: ' + e.message);
 }
}

async function loadAccounts(){
 try{
  const res=await fetch('/api/accounts');
  const data=await res.json();
  const accounts=data.accounts||[];
  const container=document.getElementById('accounts-list');
  if(!accounts.length){container.innerHTML='<p>No accounts found.</p>';return;}
  let html='<div class="grid">';
  for(const a of accounts){
   const isConnected=a.status==='connected';
   const statusClass=isConnected?'COMPLETED':'PENDING';
   html+=`<div class="post-card">
    <div style="display:flex;justify-content:space-between;align-items:center">
     <h3 style="text-transform:capitalize">${escapeHtml(a.platform)}</h3>
     <span class="badge ${statusClass}">${escapeHtml(a.status)}</span>
    </div>
    <p>Account ID: ${escapeHtml(a.account_id || 'Not configured / Unlinked')}</p>
    <p>Display Name: ${escapeHtml(a.display_name || 'N/A')}</p>
    <p>Integration: ${a.has_credential_ref ? 'Credential Ref Configured' : 'OAuth / Credentials Not Configured'}</p>
    <p style="font-size:12px;color:#bbb">Readiness: ${isConnected ? 'Ready for automated dispatch (stub/simulated)' : 'Not connected — Use manual download workflow'}</p>
    <button onclick="alert('OAuth authentication for ${escapeJs(a.platform)} is not yet configured or connected in this environment.')" style="margin-top:8px;background:#333;border:1px solid #555">Manage / Connect</button>
   </div>`;
  }
  html+='</div>';
  container.innerHTML=html;
 }catch(e){
  document.getElementById('accounts-list').textContent='Error loading accounts: '+e.message;
 }
}

async function loadContentLibrary(){
 const container=document.getElementById('content-library-container');
 container.innerHTML='Loading content library...';
 try{
  const res=await fetch('/api/missions');
  const missions=await res.json();
  if(!missions.length){
   container.innerHTML='<p>No content available yet.</p>';
   return;
  }
  let html='';
  for(const m of missions){
   const postsRes=await fetch(`/api/missions/${m.mission_id}/posts`);
   const postsData=await postsRes.json();
   if(postsData.posts && postsData.posts.length){
    html+=`<h3>Mission: ${escapeHtml(m.mission_id)}</h3>`;
    for(const p of postsData.posts){
     if(p.video_path){
      const relVideo=p.video_path.replace(/\\\\/g,'/').split('/videos/').pop();
      html+=`<div class="post-card">
       <strong>Job / Image ID: ${escapeHtml(p.image_id)}</strong>
       <video controls src="/api/missions/${m.mission_id}/files/videos/${relVideo}"></video>
       <p style="margin-top:8px"><a href="/api/missions/${m.mission_id}/files/videos/${relVideo}" download style="color:#66b2ff;font-weight:600">Download MP4 Asset</a></p>
      </div>`;
     }
    }
   }
  }
  container.innerHTML=html || '<p>No video assets generated yet.</p>';
 }catch(e){
  container.innerHTML='Error loading content library: '+e.message;
 }
}

async function loadActivityReport(){
 const container=document.getElementById('activity-report-container');
 container.innerHTML='Loading activity report...';
 try{
  const res=await fetch('/api/missions');
  const missions=await res.json();
  if(!missions.length){
   container.innerHTML='<p>No activity recorded yet.</p>';
   return;
  }
  let html='<div style="overflow-x:auto"><table style="width:100%;border-collapse:collapse;margin-top:8px">';
  html+='<tr style="border-bottom:1px solid #444;text-align:left"><th style="padding:8px">Mission ID</th><th style="padding:8px">Total Jobs</th><th style="padding:8px">Completed</th><th style="padding:8px">Failed</th><th style="padding:8px">Status</th></tr>';
  for(const m of missions){
   const counts=m.jobs||{};
   const comp=counts['COMPLETED']||0;
   const fail=counts['FAILED']||0;
   const total=Object.values(counts).reduce((a,b)=>a+b,0);
   const status = total > 0 && comp === total ? 'Completed' : (fail > 0 ? 'Failures Present' : 'In Progress / Pending');
   html+=`<tr style="border-bottom:1px solid #282828">
    <td style="padding:8px;font-weight:600">${escapeHtml(m.mission_id)}</td>
    <td style="padding:8px">${total}</td>
    <td style="padding:8px;color:#c8e6c9">${comp}</td>
    <td style="padding:8px;color:#ffcdd2">${fail}</td>
    <td style="padding:8px">${status}</td>
   </tr>`;
  }
  html+='</table></div>';
  container.innerHTML=html;
 }catch(e){
  container.innerHTML='Error loading activity report: '+e.message;
 }
}

async function loadMissionDetail(missionId){
 const detailCard=document.getElementById('mission-detail-card');
 detailCard.style.display='block';
 detailCard.dataset.loaded='true';
 document.getElementById('detail-title').textContent='Mission Output: '+missionId;
 document.getElementById('delete-mission-btn').setAttribute('onclick', `deleteMission('${escapeJs(missionId)}')`);

 const container=document.getElementById('posts-container');
 container.innerHTML='Loading posts...';
 const logContainer=document.getElementById('activity-log-container');
 logContainer.innerHTML = '<div class="card"><h3>Activity Log & Execution Stages</h3><div id="activity-stages-container">Loading activity log...</div></div>';

 loadMissionActivity(missionId);

 try{
  const res=await fetch(`/api/missions/${missionId}/posts`);
  const data=await res.json();
  if(!data.posts || !data.posts.length){
   container.innerHTML='<p>No output is available for this mission yet.</p>';
   return;
  }
  let html='';
  for(const p of data.posts){
   const statusClass=p.status||'PENDING';
   html+=`<div class="post-card">
    <div style="display:flex;justify-content:space-between;align-items:center">
     <strong>Image / Job ID: ${escapeHtml(p.image_id)}</strong>
     <span class="badge ${statusClass}">${escapeHtml(p.status)}</span>
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
    html+=`<div class="section-title" style="display:flex;justify-content:space-between;align-items:center">
     <span>Voice-over Script:</span>
     <button onclick="copyText(this, \`${escapeJs(p.voiceover_script.script_text)}\`)" style="width:auto;padding:2px 8px;font-size:11px">Copy Script</button>
    </div><pre>${escapeHtml(p.voiceover_script.script_text)}</pre>`;
   }
   if(p.captions && p.captions.length){
    html+=`<div class="section-title">Captions & Hashtags:</div>`;
    for(const c of p.captions){
     const tagStr = (c.hashtags||[]).join(' ');
     html+=`<div style="background:#151515;padding:8px;margin-top:4px;border-radius:4px">
      <div style="display:flex;justify-content:space-between;align-items:center">
       <span class="badge badge-muted">${escapeHtml(c.platform)}</span>
       <div>
        <button onclick="copyText(this, \`${escapeJs(c.caption)}\`)" style="width:auto;padding:2px 6px;font-size:11px;margin-right:4px">Copy Caption</button>
        <button onclick="copyText(this, \`${escapeJs(tagStr)}\`)" style="width:auto;padding:2px 6px;font-size:11px">Copy Hashtags</button>
       </div>
      </div>
      <p style="margin:6px 0">${escapeHtml(c.caption)}</p>
      <p style="color:#4fc3f7;font-size:13px;margin:4px 0">${escapeHtml(tagStr)}</p>
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

async function loadMissionActivity(missionId){
 try{
  const res = await fetch(`/api/missions/${missionId}/activity`);
  const data = await res.json();
  const stages = data.stages || [];
  const actContainer = document.getElementById('activity-stages-container');
  if(!actContainer) return;
  if(!stages.length){
   actContainer.innerHTML = '<p style="color:#aaa;font-size:13px">No activity stages recorded yet.</p>';
   return;
  }
  let html = '<div style="margin-top:10px">';
  for(const s of stages){
   const stClass = s.status === 'COMPLETED' ? 'COMPLETED' : 'FAILED';
   html += `<div style="background:#151515;border:1px solid #333;padding:8px;margin:6px 0;border-radius:6px;display:flex;justify-content:space-between;align-items:center">
    <div><strong>${escapeHtml(s.stage_name)}</strong><br><small style="color:#aaa">${escapeHtml(s.message || '')}</small></div>
    <div><span class="badge ${stClass}">${escapeHtml(s.status)}</span><br><small style="color:#888;font-size:10px">${escapeHtml(s.timestamp || '')}</small></div>
   </div>`;
  }
  html += '</div>';
  actContainer.innerHTML = html;
 }catch(e){
  const actContainer = document.getElementById('activity-stages-container');
  if(actContainer) actContainer.textContent = 'Error loading activity log: ' + e.message;
 }
}

async function copyText(btn, text){
 try{
  await navigator.clipboard.writeText(text);
  const orig=btn.textContent;
  btn.textContent='Copied!';
  btn.style.background='#1b5e20';
  setTimeout(()=>{btn.textContent=orig;btn.style.background='#0066cc';},1500);
 }catch(e){
  alert('Copy failed: '+e);
 }
}

function escapeJs(str){
 if(!str) return '';
 return str.replace(/\\/g, '\\\\').replace(/`/g, '\\`').replace(/\\$/g, '\\$');
}

function escapeHtml(text){
 if(!text) return '';
 return text.replace(/&/g,"&amp;").replace(/</g,"&lt;").replace(/>/g,"&gt;");
}

document.getElementById('form').onsubmit=async e=>{
 e.preventDefault();
 const submitBtn = document.getElementById('submit-btn');
 const resultBox = document.getElementById('result-box');
 submitBtn.disabled = true;
 submitBtn.textContent = 'Processing Mission...';
 resultBox.style.display = 'block';
 resultBox.innerHTML = '<p style="color:#ffb74d">Mission submitted. Processing in background...</p>';

 try{
  const res=await fetch('/api/missions',{method:'POST',body:new FormData(e.target)});
  const json=await res.json();
  if(res.ok){
   resultBox.innerHTML = `<p style="color:#81c784">Mission started successfully!</p><pre>${escapeHtml(JSON.stringify(json,null,2))}</pre>`;
   loadMissions();
   setTimeout(()=>{
    switchTab('missions');
    if(json.mission_id){loadMissionDetail(json.mission_id);}
   }, 1500);
  }else{
   resultBox.innerHTML = `<p style="color:#ff8a80">Error: ${escapeHtml(json.detail || 'Failed to start mission')}</p>`;
  }
 }catch(err){
  resultBox.innerHTML = `<p style="color:#ff8a80">Request failed: ${escapeHtml(err.message)}</p>`;
 }finally{
  submitBtn.disabled = false;
  submitBtn.textContent = 'Start Processing';
 };
};

loadSystemHealth();
loadMissions();
loadAccounts();
setInterval(()=>{loadSystemHealth(); loadMissions(); loadAccounts();}, 4000);
</script>
</body></html>"""



def create_app(config: AppConfig | None = None) -> FastAPI:
    config = config or AppConfig()
    app = FastAPI(title="Mission AI", version="0.1.0")
    output_root = Path(config.output_directory).resolve()
    upload_root = (output_root / ".web_uploads").resolve()
    upload_root.mkdir(parents=True, exist_ok=True)
    account_store = AccountStore()

    @app.get("/", response_class=HTMLResponse)
    def index():
        return HTML

    @app.get("/api/health")
    def health():
        return {"status": "ok", "service": "mission-ai"}

    @app.get("/api/accounts")
    def get_accounts():
        return {"accounts": [acc.to_dict() for acc in account_store.list_accounts()]}

    @app.get("/api/missions")
    def missions():
        output_root.mkdir(parents=True, exist_ok=True)
        result = []
        if not output_root.is_dir():
            return result
        for p in sorted(output_root.iterdir()):
            if not p.is_dir() or p.name.startswith(("_", ".")):
                continue
            checkpoint = p / "checkpoint.json"
            state = {}
            if checkpoint.exists():
                try:
                    checkpoint_data = json.loads(checkpoint.read_text())
                    # CheckpointManager stores the job map directly. Also accept
                    # the older wrapped {"state": {...}} format for compatibility.
                    state = checkpoint_data.get("state", checkpoint_data) if isinstance(checkpoint_data, dict) else {}
                    if not isinstance(state, dict):
                        state = {}
                except (OSError, ValueError):
                    state = {}
            # The current UI is output-oriented: after candidate selection,
            # only jobs with a generated content package represent usable output.
            # This prevents stale FAILED candidate attempts from previous runs
            # from appearing as the current mission result.
            counts = {}
            metadata_dir = p / "metadata"
            if metadata_dir.is_dir():
                for meta_file in metadata_dir.glob("*.json"):
                    try:
                        data = json.loads(meta_file.read_text(encoding="utf-8"))
                        status = data.get("status")
                        if status:
                            counts[status] = counts.get(status, 0) + 1
                    except (OSError, ValueError):
                        pass
            if not counts:
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

    @app.get("/api/missions/{mission_id}/activity")
    def mission_activity(mission_id: str):
        mission_dir = (output_root / mission_id).resolve()
        if not mission_dir.is_dir():
            raise HTTPException(404, "Mission not found")
        log_path = mission_dir / "activity_log.json"
        if not log_path.exists():
            return {"mission_id": mission_id, "stages": []}
        try:
            data = json.loads(log_path.read_text(encoding="utf-8"))
            return data
        except (OSError, ValueError):
            return {"mission_id": mission_id, "stages": []}

    @app.delete("/api/missions/{mission_id}")
    def delete_mission(mission_id: str):
        if not mission_id or ".." in mission_id or "/" in mission_id or "\\" in mission_id:
            raise HTTPException(400, "Invalid mission ID")
        resolved_output_root = output_root.resolve()
        mission_dir = (resolved_output_root / mission_id).resolve()
        if resolved_output_root not in mission_dir.parents and mission_dir != resolved_output_root:
            raise HTTPException(400, "Invalid mission path")
        if not mission_dir.is_dir():
            raise HTTPException(404, "Mission not found")

        checkpoint = mission_dir / "checkpoint.json"
        if checkpoint.exists():
            try:
                ckpt_data = json.loads(checkpoint.read_text(encoding="utf-8"))
                state = ckpt_data.get("state", ckpt_data) if isinstance(ckpt_data, dict) else {}
                if any(str(v).upper() == "PROCESSING" for v in state.values()):
                    raise HTTPException(400, "Cannot delete a mission while it is currently processing.")
            except HTTPException:
                raise
            except Exception:
                pass

        import shutil
        try:
            shutil.rmtree(mission_dir)
        except OSError as e:
            raise HTTPException(500, f"Failed to delete mission directory: {e}")

        web_upload_mission_dir = (upload_root / mission_id).resolve()
        if upload_root in web_upload_mission_dir.parents and web_upload_mission_dir.is_dir():
            try:
                shutil.rmtree(web_upload_mission_dir)
            except OSError:
                pass

        return {"status": "deleted", "mission_id": mission_id}

    @app.get("/api/missions/{mission_id}/posts")
    def mission_posts(mission_id: str):
        mission_dir = (output_root / mission_id).resolve()
        if not mission_dir.is_dir():
            raise HTTPException(404, "Mission not found")

        checkpoint = mission_dir / "checkpoint.json"
        state = {}
        if checkpoint.exists():
            try:
                checkpoint_data = json.loads(checkpoint.read_text())
                state = checkpoint_data.get("state", checkpoint_data) if isinstance(checkpoint_data, dict) else {}
                if not isinstance(state, dict):
                    state = {}
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

        # Keep failed jobs visible alongside completed jobs so errors remain diagnosable.
        # Metadata and checkpoint state are the source of truth for each job's status.
        return {"mission_id": mission_id, "posts": posts}

    @app.post("/api/missions")
    async def create_mission(
        mission_text: Optional[str] = Form(None),
        mission: Optional[UploadFile] = File(None),
        images: Optional[list[UploadFile]] = File(None),
    ):
        if mission is None or not mission.filename:
            if not mission_text or not mission_text.strip():
                raise HTTPException(400, "Mission text is required.")
            try:
                from mission_ai.mission.parser import MissionParser
                mission_ctx = MissionParser.parse_mission_text(mission_text)
            except ValueError as e:
                msg = str(e)
                if "required" in msg.lower():
                    raise HTTPException(400, "Mission text is required.")
                elif "no supported" in msg.lower():
                    raise HTTPException(400, "No supported Google Drive link was found in the mission text.")
                elif "invalid or unsupported" in msg.lower():
                    raise HTTPException(400, "Google Drive link is invalid or unsupported.")
                else:
                    raise HTTPException(400, msg)

            mission_id = mission_ctx.mission_id
            job_root = upload_root / mission_id
            job_root.mkdir(parents=True, exist_ok=True)
            mission_path = job_root / "mission.json"
            mission_dict = {
                "mission_id": mission_ctx.mission_id,
                "instructions": mission_ctx.instructions,
                "main_message": mission_ctx.main_message,
                "key_points": mission_ctx.key_points,
                "platforms": mission_ctx.platforms,
                "max_hashtags": mission_ctx.max_hashtags,
                "required_hashtags": mission_ctx.required_hashtags,
                "image_source_url": mission_ctx.image_source_url,
            }
            mission_path.write_text(json.dumps(mission_dict, indent=2), encoding="utf-8")
            input_source = mission_ctx.image_source_url

            def worker():
                try:
                    MissionRunner(config, progress=lambda _: None).run(
                        str(mission_path), str(input_source), str(output_root)
                    )
                except Exception as e:
                    logger.error(f"MissionRunner processing failed for mission {mission_id}: {type(e).__name__}: {e}")

            threading.Thread(target=worker, daemon=True).start()
            return {"status": "started", "mission_id": mission_id}
        else:
            if not mission or not mission.filename or not mission.filename.lower().endswith(".json"):
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
                except Exception as e:
                    logger.error(f"MissionRunner processing failed for mission {mission_id}: {type(e).__name__}: {e}")

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
