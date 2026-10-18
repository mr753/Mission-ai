import json
import logging
import threading
from datetime import datetime, timezone
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
<title>Mission AI | Dark Premium Dashboard</title>
<style>
:root {
  --bg: #0a0e17;
  --card-bg: #131b2e;
  --card-border: rgba(255, 255, 255, 0.08);
  --text: #94a3b8;
  --text-bright: #f8fafc;
  --text-muted: #64748b;
  --accent: #3b82f6;
  --accent-hover: #2563eb;
  --accent-glow: rgba(59, 130, 246, 0.25);
  --success-bg: rgba(6, 95, 70, 0.4);
  --success-border: rgba(16, 185, 129, 0.3);
  --success-text: #34d399;
  --fail-bg: rgba(127, 29, 29, 0.4);
  --fail-border: rgba(239, 68, 68, 0.3);
  --fail-text: #f87171;
  --pending-bg: rgba(120, 53, 15, 0.4);
  --pending-border: rgba(245, 158, 11, 0.3);
  --pending-text: #fbbf24;
  --input-bg: #0f172a;
  --input-border: #334155;
}
* { box-sizing: border-box; }
body {
  font-family: system-ui, -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
  max-width: 1200px;
  margin: 0 auto;
  padding: 16px;
  background: var(--bg);
  color: var(--text);
  line-height: 1.6;
}
header {
  display: flex;
  flex-wrap: wrap;
  justify-content: space-between;
  align-items: center;
  background: var(--card-bg);
  border: 1px solid var(--card-border);
  border-radius: 16px;
  padding: 20px 24px;
  margin-bottom: 20px;
  box-shadow: 0 10px 30px rgba(0, 0, 0, 0.4);
}
h1, h2, h3 { color: var(--text-bright); margin-top: 0; font-weight: 700; }
h1 { font-size: 24px; letter-spacing: -0.5px; }
h2 { font-size: 18px; margin-bottom: 12px; }
h3 { font-size: 15px; margin-bottom: 8px; }

nav {
  display: flex;
  flex-wrap: wrap;
  gap: 10px;
  margin-bottom: 24px;
  background: var(--card-bg);
  padding: 8px;
  border-radius: 14px;
  border: 1px solid var(--card-border);
}
nav button {
  background: transparent;
  color: var(--text);
  border: 1px solid transparent;
  padding: 10px 18px;
  border-radius: 10px;
  cursor: pointer;
  font-weight: 600;
  font-size: 14px;
  width: auto;
  margin: 0;
  transition: all 0.2s ease;
}
nav button:hover {
  color: var(--text-bright);
  background: rgba(255, 255, 255, 0.04);
}
nav button.active {
  background: var(--accent);
  color: #fff;
  box-shadow: 0 4px 14px var(--accent-glow);
  border-color: rgba(255, 255, 255, 0.1);
}

.tab-content { display: none; }
.tab-content.active { display: block; animation: fadeIn 0.25s ease; }
@keyframes fadeIn { from { opacity: 0; transform: translateY(4px); } to { opacity: 1; transform: translateY(0); } }

.card {
  background: var(--card-bg);
  border: 1px solid var(--card-border);
  border-radius: 16px;
  padding: 24px;
  margin: 20px 0;
  box-shadow: 0 10px 25px rgba(0, 0, 0, 0.3);
}

label { font-weight: 600; font-size: 13px; color: var(--text-bright); display: block; margin-bottom: 6px; }
input, textarea, select {
  padding: 12px 14px;
  margin: 6px 0 16px 0;
  width: 100%;
  box-sizing: border-box;
  background: var(--input-bg);
  color: var(--text-bright);
  border: 1px solid var(--input-border);
  border-radius: 10px;
  font-family: inherit;
  font-size: 14px;
  transition: border-color 0.2s, box-shadow 0.2s;
}
input:focus, textarea:focus {
  outline: none;
  border-color: var(--accent);
  box-shadow: 0 0 0 3px var(--accent-glow);
}

button {
  background: var(--accent);
  color: #fff;
  cursor: pointer;
  font-weight: 600;
  border: none;
  border-radius: 10px;
  padding: 12px 20px;
  width: 100%;
  font-size: 14px;
  transition: background 0.2s, transform 0.1s, box-shadow 0.2s;
  box-shadow: 0 4px 12px var(--accent-glow);
}
button:hover { background: var(--accent-hover); box-shadow: 0 6px 18px var(--accent-glow); }
button:active { transform: translateY(1px); }
button:disabled { background: #334155; color: var(--text-muted); cursor: not-allowed; box-shadow: none; }

pre {
  white-space: pre-wrap;
  background: #090d16;
  padding: 14px;
  border-radius: 10px;
  border: 1px solid var(--card-border);
  overflow-x: auto;
  font-family: 'Fira Code', Consolas, Monaco, monospace;
  font-size: 13px;
  color: #cbd5e1;
}

.grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(280px, 1fr));
  gap: 16px;
  margin-top: 12px;
}
.post-card {
  background: #182238;
  border: 1px solid var(--card-border);
  border-radius: 12px;
  padding: 16px;
  margin: 8px 0;
  transition: transform 0.2s, border-color 0.2s;
}
.post-card:hover {
  border-color: rgba(59, 130, 246, 0.4);
}

.badge {
  display: inline-flex;
  align-items: center;
  padding: 4px 10px;
  border-radius: 6px;
  font-size: 11px;
  font-weight: 700;
  letter-spacing: 0.5px;
  text-transform: uppercase;
}
.COMPLETED { background: var(--success-bg); color: var(--success-text); border: 1px solid var(--success-border); }
.FAILED { background: var(--fail-bg); color: var(--fail-text); border: 1px solid var(--fail-border); }
.PENDING, .PROCESSING { background: var(--pending-bg); color: var(--pending-text); border: 1px solid var(--pending-border); }
.badge-muted { background: #1e293b; color: var(--text); border: 1px solid #334155; }

.section-title {
  font-size: 13px;
  font-weight: 700;
  color: var(--text-muted);
  text-transform: uppercase;
  letter-spacing: 0.5px;
  margin-top: 16px;
  margin-bottom: 6px;
}
video {
  width: 100%;
  max-height: 360px;
  background: #000;
  border-radius: 10px;
  margin-top: 10px;
  border: 1px solid var(--card-border);
}
.status-box {
  background: #0f172a;
  border: 1px solid var(--card-border);
  padding: 16px;
  border-radius: 12px;
  margin-top: 16px;
}
table {
  width: 100%;
  border-collapse: collapse;
  margin-top: 12px;
}
th, td {
  padding: 12px;
  text-align: left;
  border-bottom: 1px solid var(--card-border);
  font-size: 14px;
}
th {
  color: var(--text-bright);
  font-weight: 600;
  background: rgba(255, 255, 255, 0.02);
}
@media(min-width: 768px) {
  body { padding: 32px; }
}
body {
  max-width: 100%;
  overflow-x: hidden;
  word-break: break-word;
  overflow-wrap: break-word;
  padding: 12px;
}
.card, .post-card, .status-box, pre, video, table, th, td, #mission-progress-banner {
  width: 100%;
  max-width: 100%;
  min-width: 0;
  box-sizing: border-box;
  word-break: break-word;
  overflow-wrap: break-word;
}
pre {
  white-space: pre-wrap;
  word-break: break-all;
  overflow-x: auto;
}
video {
  width: 100%;
  height: auto;
  max-height: 360px;
  object-fit: contain;
}
@keyframes spin {
  0% { transform: rotate(0deg); }
  100% { transform: rotate(360deg); }
}
.spinner {
  width: 18px;
  height: 18px;
  border: 2px solid rgba(255, 255, 255, 0.2);
  border-top-color: var(--accent);
  border-radius: 50%;
  animation: spin 0.8s linear infinite;
  display: inline-block;
  flex-shrink: 0;
}
@media (prefers-reduced-motion: reduce) {
  * {
    animation: none !important;
    transition: none !important;
  }
}
@media (max-width: 480px) {
  body { padding: 8px; }
  .card { padding: 14px; margin: 12px 0; }
  header { padding: 14px; }
  nav button { padding: 8px 12px; font-size: 13px; }
  .grid { grid-template-columns: 1fr; }
}
</style>
</head>
<body>
<header>
<div>
<h1 style="margin:0">Mission AI</h1>
<small style="color:var(--text-muted)">Autonomous Video & Content Production Engine</small>
</div>
<div id="system-status" style="font-size:13px;color:#34d399;font-weight:600;display:flex;align-items:center;gap:6px">
<span style="width:8px;height:8px;background:#34d399;border-radius:50%;display:inline-block;box-shadow:0 0 8px #34d399"></span>
System: Connected
</div>
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
<p style="color:var(--text);font-size:13px;margin-bottom:16px">Paste your mission instructions below. Include a Google Drive folder link to source input images.</p>
<form id="form">
<label>Mission Text
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
<p style="color:var(--text);font-size:13px;margin-bottom:16px">Select a mission below to view its full processing status, videos, scripts, and captions, or delete completed missions.</p>
<div id="missions-list">Loading missions...</div>
</div>

<div class="card" id="mission-detail-card" style="display:none">
<div id="detail-title-bar" style="display:flex;justify-content:space-between;align-items:center;margin-bottom:12px">
<h2 id="detail-title" style="margin:0">Mission Output</h2>
<button id="delete-mission-btn" onclick="" style="background:rgba(239, 68, 68, 0.2);color:#f87171;border:1px solid rgba(239, 68, 68, 0.3);width:auto;padding:6px 14px;font-size:12px;box-shadow:none">Delete Mission</button>
</div>
<div id="posts-container"><p>Loading mission output...</p></div>
<div id="activity-log-container" style="margin-top:16px"></div>
</div>
</div>

<!-- TAB 3: SOCIAL ACCOUNTS -->
<div id="tab-accounts" class="tab-content">
<div class="card">
<h2>Social Media Account Management</h2>
<p style="color:var(--text);font-size:13px;margin-bottom:16px">Manage platform connections and check automation readiness. Note: Real automated publishing requires OAuth configuration (currently operating in manual download / simulation mode).</p>
<div id="accounts-list">Loading accounts...</div>
</div>
</div>

<!-- TAB 4: CONTENT LIBRARY -->
<div id="tab-content" class="tab-content">
<div class="card">
<h2>Content Library</h2>
<p style="color:var(--text);font-size:13px;margin-bottom:16px">Generated videos, voice-over scripts, captions, and hashtags ready for manual download or posting.</p>
<div id="content-library-container">Loading content library...</div>
</div>
</div>

<!-- TAB 5: ACTIVITY REPORT -->
<div id="tab-activity" class="tab-content">
<div class="card">
<h2>Activity Report & Status Log</h2>
<p style="color:var(--text);font-size:13px;margin-bottom:16px">Summary of processed missions, job success rates, and operational status.</p>
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
   document.getElementById('system-status').innerHTML='<span style="width:8px;height:8px;background:#34d399;border-radius:50%;display:inline-block;box-shadow:0 0 8px #34d399"></span> System: Online ('+data.service+')';
  }
 }catch(e){
  document.getElementById('system-status').innerHTML='<span style="width:8px;height:8px;background:#f87171;border-radius:50%;display:inline-block;box-shadow:0 0 8px #f87171"></span> System: Offline / Error';
  document.getElementById('system-status').style.color='#f87171';
 }
}

async function loadMissions(){
 try{
  const res=await fetch('/api/missions');
  const missions=await res.json();
  if(!Array.isArray(missions)) {
   throw new Error("Invalid missions response format");
  }

  // Render dashboard summary
  const sumContainer=document.getElementById('dashboard-missions-summary');
  if(sumContainer){
   if(!missions.length){
    sumContainer.innerHTML='<p style="color:var(--text-muted)">No missions found. Start a new mission from the Dashboard tab.</p>';
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
       <p style="margin:4px 0">Jobs: ${total} | ${statusBadge}</p>
       <p style="font-size:13px;color:var(--text-muted);margin:4px 0">Completed: ${comp} | Failed: ${fail}</p>
      </div>
      <button onclick="deleteMission('${escapeJs(m.mission_id)}')" style="background:rgba(239, 68, 68, 0.2);color:#f87171;border:1px solid rgba(239, 68, 68, 0.3);margin-top:10px;padding:6px 10px;font-size:12px;width:auto;box-shadow:none">Delete Mission</button>
     </div>`;
    }
    sumHtml+='</div>';
    sumContainer.innerHTML=sumHtml;
   }
  }

  // Render missions list
  const container=document.getElementById('missions-list');
  if(container){
   if(!missions.length){
    container.innerHTML='<p style="color:var(--text-muted)">No missions found.</p>';
   } else {
    let html='<div class="grid">';
    for(const m of missions){
     const counts=m.jobs||{};
     const comp=counts['COMPLETED']||0;
     const fail=counts['FAILED']||0;
     const total=Object.values(counts).reduce((a,b)=>a+b,0);
     html+=`<div class="post-card">
      <div style="cursor:pointer" onclick="loadMissionDetail('${escapeJs(m.mission_id)}')">
       <h3>${escapeHtml(m.mission_id)}</h3>
       <p style="margin:4px 0">Total Jobs: ${total}</p>
       <p style="margin:4px 0"><span class="badge COMPLETED">Completed: ${comp}</span> <span class="badge FAILED">Failed: ${fail}</span></p>
      </div>
      <button onclick="deleteMission('${escapeJs(m.mission_id)}')" style="background:rgba(239, 68, 68, 0.2);color:#f87171;border:1px solid rgba(239, 68, 68, 0.3);margin-top:10px;padding:6px 10px;font-size:12px;width:auto;box-shadow:none">Delete Mission</button>
     </div>`;
    }
    html+='</div>';
    container.innerHTML=html;
   }
  }

  const detailCard = document.getElementById('mission-detail-card');
  if(missions.length === 1 && detailCard && !detailCard.dataset.loaded){
   loadMissionDetail(missions[0].mission_id);
  }
 }catch(e){
  const sumContainer = document.getElementById('dashboard-missions-summary');
  if(sumContainer) sumContainer.textContent = 'Error loading summary: ' + e.message;
  const listContainer = document.getElementById('missions-list');
  if(listContainer) listContainer.textContent = 'Error loading missions: ' + e.message;
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
  if(!accounts.length){container.innerHTML='<p style="color:var(--text-muted)">No accounts found.</p>';return;}
  let html='<div class="grid">';
  for(const a of accounts){
   const isConnected=a.status==='connected';
   const statusClass=isConnected?'COMPLETED':'PENDING';
   html+=`<div class="post-card">
    <div style="display:flex;justify-content:space-between;align-items:center">
     <h3 style="text-transform:capitalize;margin:0">${escapeHtml(a.platform)}</h3>
     <span class="badge ${statusClass}">${escapeHtml(a.status)}</span>
    </div>
    <p style="margin:8px 0 4px 0">Account ID: ${escapeHtml(a.account_id || 'Not configured / Unlinked')}</p>
    <p style="margin:4px 0">Display Name: ${escapeHtml(a.display_name || 'N/A')}</p>
    <p style="margin:4px 0">Integration: ${a.has_credential_ref ? 'Credential Ref Configured' : 'OAuth / Credentials Not Configured'}</p>
    <p style="font-size:12px;color:var(--text-muted);margin:4px 0">Readiness: ${isConnected ? 'Ready for automated dispatch (stub/simulated)' : 'Not connected — Use manual download workflow'}</p>
    <button onclick="alert('OAuth authentication for ${escapeJs(a.platform)} is not yet configured or connected in this environment.')" style="margin-top:12px;background:#1e293b;color:var(--text-bright);border:1px solid #334155;box-shadow:none">Manage / Connect</button>
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
   container.innerHTML='<p style="color:var(--text-muted)">No content available yet.</p>';
   return;
  }
  let html='';
  for(const m of missions){
   const postsRes=await fetch(`/api/missions/${m.mission_id}/posts`);
   const postsData=await postsRes.json();
   if(postsData.posts && postsData.posts.length){
    html+=`<h3 style="margin-top:16px">Mission: ${escapeHtml(m.mission_id)}</h3>`;
    for(const p of postsData.posts){
     if(p.video_path){
      const relVideo=p.video_path.replace(/\\\\/g,'/').split('/videos/').pop();
      html+=`<div class="post-card">
       <strong style="color:var(--text-bright)">Job / Image ID: ${escapeHtml(p.image_id)}</strong>
       <video controls src="/api/missions/${m.mission_id}/files/videos/${relVideo}"></video>
       <p style="margin-top:12px"><a href="/api/missions/${m.mission_id}/files/videos/${relVideo}" download style="color:#60a5fa;font-weight:600;text-decoration:none">Download MP4 Asset</a></p>
      </div>`;
     }
    }
   }
  }
  container.innerHTML=html || '<p style="color:var(--text-muted)">No video assets generated yet.</p>';
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
   container.innerHTML='<p style="color:var(--text-muted)">No activity recorded yet.</p>';
   return;
  }
  let html='<div style="overflow-x:auto"><table>';
  html+='<tr><th>Mission ID</th><th>Total Jobs</th><th>Completed</th><th>Failed</th><th>Status</th></tr>';
  for(const m of missions){
   const counts=m.jobs||{};
   const comp=counts['COMPLETED']||0;
   const fail=counts['FAILED']||0;
   const total=Object.values(counts).reduce((a,b)=>a+b,0);
   const status = total > 0 && comp === total ? 'Completed' : (fail > 0 ? 'Failures Present' : 'In Progress / Pending');
   html+=`<tr>
    <td style="font-weight:600;color:var(--text-bright)">${escapeHtml(m.mission_id)}</td>
    <td>${total}</td>
    <td style="color:#34d399">${comp}</td>
    <td style="color:#f87171">${fail}</td>
    <td>${status}</td>
   </tr>`;
  }
  html+='</table></div>';
  container.innerHTML=html;
 }catch(e){
  container.innerHTML='Error loading activity report: '+e.message;
 }
}

let activeMissionPollTimer = null;

async function loadMissionDetail(missionId){
 if(activeMissionPollTimer) { clearInterval(activeMissionPollTimer); activeMissionPollTimer = null; }
 const detailCard=document.getElementById('mission-detail-card');
 detailCard.style.display='block';
 detailCard.dataset.loaded='true';
 document.getElementById('detail-title').textContent='Mission Output: '+missionId;
 document.getElementById('delete-mission-btn').setAttribute('onclick', `deleteMission('${escapeJs(missionId)}')`);

 if(!document.getElementById('mission-progress-banner')) {
  const detailTitleBar = document.getElementById('detail-title-bar');
  if(detailTitleBar) {
   const bannerHtml = `<div id="mission-progress-banner" class="card" style="background:#0f172a;border:1px solid var(--accent);margin:12px 0;padding:16px;box-sizing:border-box;width:100%;max-width:100%">
     <div style="display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:8px">
       <div style="display:flex;align-items:center;gap:8px">
         <div id="progress-spinner-container"><span class="spinner"></span></div>
         <strong id="progress-status-title" style="color:var(--text-bright)">Processing Mission...</strong>
       </div>
       <div id="progress-badge-container"><span class="badge PROCESSING">In Progress</span></div>
     </div>
     <p id="progress-active-stage" style="margin:8px 0 4px 0;font-size:13px;color:var(--text)">Current Stage: Initializing...</p>
     <div style="background:#1e293b;border-radius:6px;height:6px;width:100%;margin:10px 0;overflow:hidden">
       <div id="progress-bar-fill" style="background:var(--accent);height:100%;width:15%;transition:width 0.3s ease"></div>
     </div>
     <div style="display:flex;justify-content:space-between;align-items:center;font-size:12px;color:var(--text-muted);flex-wrap:wrap;gap:4px">
       <span id="progress-time-started">Started: --:--</span>
       <span id="progress-duration">Duration: 0s</span>
     </div>
   </div>`;
   detailTitleBar.insertAdjacentHTML('afterend', bannerHtml);
  }
 }

 await refreshMissionDetailData(missionId);

 activeMissionPollTimer = setInterval(async () => {
  const currentCard = document.getElementById('mission-detail-card');
  if(!currentCard || currentCard.style.display === 'none') {
   clearInterval(activeMissionPollTimer);
   activeMissionPollTimer = null;
   return;
  }
  const isTerminal = await refreshMissionDetailData(missionId);
  if(isTerminal) {
   clearInterval(activeMissionPollTimer);
   activeMissionPollTimer = null;
  }
 }, 3500);
}

async function refreshMissionDetailData(missionId) {
 const container=document.getElementById('posts-container');
 const logContainer=document.getElementById('activity-log-container');
 if(!logContainer.innerHTML.includes('activity-stages-container')) {
  logContainer.innerHTML = '<div class="card" style="margin:0;background:#0f172a"><h3>Activity Log & Execution Stages</h3><div id="activity-stages-container">Loading activity log...</div></div>';
 }

 const hasTerminal = await loadMissionActivity(missionId);
 try{
  const res=await fetch(`/api/missions/${missionId}/posts`);
  const data=await res.json();
  if(!data.posts || !data.posts.length){
   container.innerHTML='<p style="color:var(--text-muted)">No output is available for this mission yet. Processing in progress...</p>';
   return hasTerminal;
  }
  let html='';
  for(const p of data.posts){
   const statusClass=p.status||'PENDING';
   html+=`<div class="post-card">
    <div style="display:flex;justify-content:space-between;align-items:center">
     <strong style="color:var(--text-bright)">Image / Job ID: ${escapeHtml(p.image_id)}</strong>
     <span class="badge ${statusClass}">${escapeHtml(p.status)}</span>
    </div>`;
   if(p.error){
    html+=`<p style="color:#f87171;margin:8px 0;background:rgba(239, 68, 68, 0.1);padding:8px;border-radius:6px;border:1px solid rgba(239, 68, 68, 0.2)"><strong>Error:</strong> ${escapeHtml(p.error)}</p>`;
   }
   if(p.video_path){
    const relVideo=p.video_path.replace(/\\\\/g,'/').split('/videos/').pop();
    html+=`<div class="section-title">Video Asset:</div>
    <video controls src="/api/missions/${missionId}/files/videos/${relVideo}"></video>
    <p style="margin-top:8px"><a href="/api/missions/${missionId}/files/videos/${relVideo}" download style="color:#60a5fa;font-weight:600;text-decoration:none">Download MP4 Asset</a></p>`;
   } else {
    html+=`<div class="section-title">Video Asset:</div><p style="color:var(--text-muted);margin:4px 0">Not generated / unavailable (Processing...)</p>`;
   }
   if(p.voiceover_script && p.voiceover_script.script_text){
    html+=`<div class="section-title" style="display:flex;justify-content:space-between;align-items:center">
     <span>Voice-over Script:</span>
     <button onclick="copyText(this, \`${escapeJs(p.voiceover_script.script_text)}\`)" style="width:auto;padding:4px 10px;font-size:11px;box-shadow:none">Copy Script</button>
    </div><pre>${escapeHtml(p.voiceover_script.script_text)}</pre>`;
   }
   if(p.captions && p.captions.length){
    html+=`<div class="section-title">Captions & Hashtags:</div>`;
    for(const c of p.captions){
     const tagStr = (c.hashtags||[]).join(' ');
     html+=`<div style="background:#0f172a;padding:12px;margin-top:8px;border-radius:10px;border:1px solid var(--card-border)">
      <div style="display:flex;justify-content:space-between;align-items:center">
       <span class="badge badge-muted">${escapeHtml(c.platform)}</span>
       <div>
        <button onclick="copyText(this, \`${escapeJs(c.caption)}\`)" style="width:auto;padding:4px 8px;font-size:11px;margin-right:4px;box-shadow:none">Copy Caption</button>
        <button onclick="copyText(this, \`${escapeJs(tagStr)}\`)" style="width:auto;padding:4px 8px;font-size:11px;box-shadow:none">Copy Hashtags</button>
       </div>
      </div>
      <p style="margin:8px 0;color:var(--text-bright)">${escapeHtml(c.caption)}</p>
      <p style="color:#60a5fa;font-size:13px;margin:4px 0">${escapeHtml(tagStr)}</p>
     </div>`;
    }
   }
   html+=`</div>`;
  }
  container.innerHTML=html;
  return hasTerminal;
 }catch(e){
  container.innerHTML='Error loading mission posts: '+e.message;
  return false;
 }
}

async function loadMissionActivity(missionId){
 try{
  const res = await fetch(`/api/missions/${missionId}/activity`);
  const data = await res.json();
  const stages = data.stages || [];
  const actContainer = document.getElementById('activity-stages-container');
  if(!actContainer) return true;

  if(!stages.length){
   actContainer.innerHTML = '<p style="color:var(--text-muted);font-size:13px">Initializing mission pipeline...</p>';
   return false;
  }

  const startTime = new Date(stages[0].timestamp);
  const lastTime = new Date(stages[stages.length - 1].timestamp);
  let durationSec = Math.floor((lastTime - startTime) / 1000);
  if(isNaN(durationSec) || durationSec < 0) durationSec = 0;
  const durationStr = durationSec < 60 ? durationSec + 's' : Math.floor(durationSec / 60) + 'm ' + (durationSec % 60) + 's';

  const latestStage = stages[stages.length - 1];
  const activeStageName = latestStage ? latestStage.stage_name : "Sedang diproses";
  const hasFailed = stages.some(s => s.status === 'FAILED');
  const hasCompletedAll = stages.some(s => s.stage_name && s.stage_name.includes('Content Package Creation') && s.status === 'COMPLETED');
  const isTerminal = hasFailed || hasCompletedAll;

  const bannerTitle = document.getElementById('progress-status-title');
  const bannerStage = document.getElementById('progress-active-stage');
  const bannerBar = document.getElementById('progress-bar-fill');
  const bannerStarted = document.getElementById('progress-time-started');
  const bannerDuration = document.getElementById('progress-duration');
  const bannerBadge = document.getElementById('progress-badge-container');
  const bannerSpinner = document.getElementById('progress-spinner-container');

  if(bannerTitle) {
   if(hasFailed) {
    bannerTitle.textContent = "Mission Failed / Errors Present";
    if(bannerBadge) bannerBadge.innerHTML = '<span class="badge FAILED">Failed</span>';
    if(bannerSpinner) bannerSpinner.innerHTML = '';
   } else if(hasCompletedAll || isTerminal) {
    bannerTitle.textContent = "Mission Completed Successfully";
    if(bannerBadge) bannerBadge.innerHTML = '<span class="badge COMPLETED">Completed</span>';
    if(bannerSpinner) bannerSpinner.innerHTML = '';
   } else {
    bannerTitle.textContent = "Processing Mission...";
    if(bannerBadge) bannerBadge.innerHTML = '<span class="badge PROCESSING">In Progress</span>';
    if(bannerSpinner) bannerSpinner.innerHTML = '<span class="spinner"></span>';
   }
  }
  if(bannerStage) bannerStage.textContent = "Current Stage: " + activeStageName;
  if(bannerBar) {
   const pct = Math.min(100, Math.max(10, Math.round((stages.length / 9) * 100)));
   bannerBar.style.width = (hasCompletedAll ? 100 : pct) + '%';
   if(hasFailed) bannerBar.style.background = '#f87171';
   else if(hasCompletedAll) bannerBar.style.background = '#34d399';
   else bannerBar.style.background = 'var(--accent)';
  }
  if(bannerStarted) bannerStarted.textContent = "Started: " + startTime.toLocaleTimeString();
  if(bannerDuration) bannerDuration.textContent = "Duration: " + durationStr;

  let summaryHeader = `<div style="margin-bottom:12px;padding:8px 12px;background:#182238;border-radius:8px;font-size:13px;display:flex;flex-wrap:wrap;justify-content:space-between;gap:8px">
   <div><span style="color:var(--text-muted)">Started:</span> ${startTime.toLocaleTimeString()}</div>
   <div><span style="color:var(--text-muted)">Duration:</span> ${durationStr}</div>
   <div><span style="color:var(--text-muted)">Total Stages:</span> ${stages.length}</div>
  </div>`;

  let html = summaryHeader + '<div style="margin-top:10px">';
  for(const s of stages){
   const stClass = s.status === 'COMPLETED' ? 'COMPLETED' : (s.status === 'FAILED' ? 'FAILED' : 'PROCESSING');
   html += `<div style="background:#131b2e;border:1px solid var(--card-border);padding:10px 14px;margin:8px 0;border-radius:10px;display:flex;justify-content:space-between;align-items:center">
    <div><strong style="color:var(--text-bright)">${escapeHtml(s.stage_name)}</strong><br><small style="color:var(--text)">${escapeHtml(s.message || '')}</small></div>
    <div style="text-align:right"><span class="badge ${stClass}">${escapeHtml(s.status)}</span><br><small style="color:var(--text-muted);font-size:11px">${escapeHtml(s.timestamp ? new Date(s.timestamp).toLocaleTimeString() : '')}</small></div>
   </div>`;
  }
  html += '</div>';
  actContainer.innerHTML = html;

  return isTerminal;
 }catch(e){
  const actContainer = document.getElementById('activity-stages-container');
  if(actContainer) actContainer.textContent = 'Error loading activity log: ' + e.message;
  return false;
 }
}

async function copyText(btn, text){
 try{
  await navigator.clipboard.writeText(text);
  const orig=btn.textContent;
  btn.textContent='Copied!';
  btn.style.background='#065f46';
  setTimeout(()=>{btn.textContent=orig;btn.style.background='var(--accent)';},1500);
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
 resultBox.innerHTML = '<p style="color:#fbbf24;margin:0">Mission submitted. Processing in background...</p>';

 try{
  const res=await fetch('/api/missions',{method:'POST',body:new FormData(e.target)});
  const json=await res.json();
  if(res.ok){
   resultBox.innerHTML = `<p style="color:#34d399;margin-bottom:8px">Mission started successfully!</p><pre>${escapeHtml(JSON.stringify(json,null,2))}</pre>`;
   loadMissions();
   setTimeout(()=>{
    switchTab('missions');
    if(json.mission_id){loadMissionDetail(json.mission_id);}
   }, 1500);
  }else{
   resultBox.innerHTML = `<p style="color:#f87171;margin:0">Error: ${escapeHtml(json.detail || 'Failed to start mission')}</p>`;
  }
 }catch(err){
  resultBox.innerHTML = `<p style="color:#f87171;margin:0">Request failed: ${escapeHtml(err.message)}</p>`;
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
                err_msg = "Job recorded in checkpoint without content package metadata."
                if str(job_status).upper() == "PROCESSING":
                    err_msg = "Job is currently processing (video rendering / asset generation in progress)..."
                elif str(job_status).upper() == "PENDING":
                    err_msg = "Job is pending execution."
                elif str(job_status).upper() == "FAILED":
                    err_msg = "Job failed during execution."
                posts.append({
                    "image_id": job_id,
                    "status": job_status,
                    "error": err_msg
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

            def log_worker_failure(err):
                try:
                    log_path = output_root / mission_id / "activity_log.json"
                    log_path.parent.mkdir(parents=True, exist_ok=True)
                    data = {"mission_id": mission_id, "stages": []}
                    if log_path.exists():
                        try:
                            data = json.loads(log_path.read_text(encoding="utf-8"))
                        except (OSError, ValueError):
                            pass
                    stages = data.get("stages", [])
                    has_terminal = any(s.get("status") in ("COMPLETED", "FAILED") for s in stages if "Initialization" not in s.get("stage_name", ""))
                    if not has_terminal:
                        err_msg = f"{type(err).__name__}: {err}"
                        stages.append({
                            "stage_name": "Pipeline Execution",
                            "status": "FAILED",
                            "timestamp": datetime.now(timezone.utc).isoformat(),
                            "message": err_msg
                        })
                        data["stages"] = stages
                        log_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
                except Exception as log_err:
                    logger.debug(f"Failed to log worker failure stage: {log_err}")

            def worker():
                try:
                    MissionRunner(config, progress=lambda _: None).run(
                        str(mission_path), str(input_source), str(output_root)
                    )
                except Exception as e:
                    logger.error(f"MissionRunner processing failed for mission {mission_id}: {type(e).__name__}: {e}")
                    log_worker_failure(e)

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
                    log_worker_failure(e)

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
