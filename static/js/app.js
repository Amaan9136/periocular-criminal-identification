async function registerCriminal(e){
e.preventDefault();
const f=e.target;
const fd=new FormData();
fd.append('criminal_id',f.criminal_id.value);
fd.append('name',f.name.value);
fd.append('metadata',f.metadata.value||'{}');
for(const file of f.images.files)fd.append('images',file);
const out=document.getElementById('register-result');
out.innerHTML='Submitting...';
const res=await fetch('/criminals',{method:'POST',body:fd});
const data=await res.json();
if(!res.ok){out.innerHTML=`<div class="warn">${data.detail||'Error'}</div>`;return}
out.innerHTML=`<div class="warn" style="background:#1b3a2c;border-color:#2a5a3c;color:#5fd694">Registered ${data.criminal_id}: ${data.images_registered} image(s) indexed, ${data.images_rejected} rejected</div>`+
`<div class="imgrow">${data.image_urls.map(u=>`<img src="${u}">`).join('')}</div>`;
f.reset();
}

async function searchFaces(e){
e.preventDefault();
const f=e.target;
const fd=new FormData();
fd.append('probe_id',f.probe_id.value||('probe_'+Date.now()));
const cfg={top_k:parseInt(f.top_k.value)||10,reconstruction_mode:f.reconstruction_mode.value,use_periocular_only:f.use_periocular_only.checked};
fd.append('config_json',JSON.stringify(cfg));
for(const file of f.images.files)fd.append('images',file);
const out=document.getElementById('search-result');
out.innerHTML='Searching...';
const res=await fetch('/search',{method:'POST',body:fd});
const data=await res.json();
if(!res.ok){out.innerHTML=`<div class="warn">${data.detail||'Error'}</div>`;return}
let html='';
if(data.warnings.length)html+=`<div class="warn">${data.warnings.join('<br>')}</div>`;
html+=`<div class="imgrow">${data.probe_reconstructions.filter(r=>r.image_url).map(r=>`<img src="${r.image_url}">`).join('')}</div>`;
html+='<div class="panel"><h2>Top matches</h2>';
if(!data.candidates.length)html+='<div class="modeldesc">No candidates in gallery yet.</div>';
for(const c of data.candidates){
html+=`<div class="candidate"><span>${c.criminal_id} ${c.metadata&&c.metadata.name?('- '+c.metadata.name):''}</span><span class="sim">${(c.similarity*100).toFixed(1)}%</span></div>`;
}
html+='</div>';
out.innerHTML=html;
}

const modelSourceState={};
const modelPollTimers={};

function fmtBytes(n){
if(n==null)return '';
if(n<1024)return n+' B';
if(n<1024*1024)return (n/1024).toFixed(1)+' KB';
if(n<1024*1024*1024)return (n/(1024*1024)).toFixed(1)+' MB';
return (n/(1024*1024*1024)).toFixed(2)+' GB';
}

function sourceLabel(s){
return s==='download'?'Download':s==='local_path'?'Local path':'Upload';
}

async function loadModels(){
const res=await fetch('/api/models');
const models=await res.json();
const el=document.getElementById('models-list');
el.innerHTML=models.map(renderModelCard).join('');
for(const m of models){
if(m.id==='gpen_bfr_256'){
const job=await fetchStatus(m.id);
if(job&&(job.state==='downloading'||job.state==='finalizing'))startPolling(m.id);
}
}
}

function renderModelCard(m){
const methods=m.install_methods&&m.install_methods.length?m.install_methods:(m.installable?['download']:[]);
const activeSource=modelSourceState[m.id]||methods[0]||'download';
const tabs=methods.map(s=>`<div class="source-tab ${s===activeSource?'active':''}" onclick="selectSource('${m.id}','${s}')">${sourceLabel(s)}</div>`).join('');
let installControls='';
if(m.installable&&methods.length){
if(activeSource==='local_path'){
installControls=`<div class="install-row"><input type="text" id="path-${m.id}" placeholder="C:\\models\\${m.id==='buffalo_l'?'buffalo_l.zip':'GPEN-BFR-256.onnx'}"><button class="secondary" onclick="installModel('${m.id}')">Install</button></div>`;
}else if(activeSource==='upload'){
installControls=`<div class="install-row"><input type="file" id="file-${m.id}" accept=".onnx"><button class="secondary" onclick="installModel('${m.id}')">Install</button></div>`;
}else{
installControls=`<div class="install-row"><input type="text" id="url-${m.id}" placeholder="${m.download_url?m.download_url:'https://... trusted source URL for '+m.name}" value="${m.download_url||''}"><button class="secondary" id="btn-${m.id}" onclick="installModel('${m.id}')">${m.status==='active'?'Re-download':'Download'}</button></div>`;
}
}
const deleteBtn=m.status==='active'?`<button class="danger" onclick="deleteModel('${m.id}')">Delete</button>`:'';
return `
<div class="model-card" id="card-${m.id}">
<div class="model-head">
<div>
<div class="model-name"><span>${m.name}</span> <span class="model-req-tag">${m.required?'required':'optional'}</span> <span class="badge ${m.status}">${m.status}</span></div>
<div class="modeldesc">${m.description}</div>
${m.path?`<div class="model-path">${m.path}</div>`:''}
</div>
<div>${deleteBtn}</div>
</div>
${methods.length?`<div class="source-tabs">${tabs}</div>${installControls}`:''}
<div class="dl-progress" id="dl-${m.id}" style="display:none">
<div class="bar-track"><div class="bar-fill indeterminate" id="dl-bar-${m.id}"></div></div>
<div class="dl-meta"><span id="dl-status-${m.id}"></span><span id="dl-bytes-${m.id}"></span></div>
<div class="dl-error" id="dl-error-${m.id}" style="display:none"></div>
</div>
</div>`;
}

function selectSource(id,source){
modelSourceState[id]=source;
loadModels();
}

async function installModel(id){
const source=modelSourceState[id]||'download';
const fd=new FormData();
fd.append('source',source);
if(source==='local_path'){
const pathInput=document.getElementById('path-'+id);
if(!pathInput.value){alert('Enter a local path first');return}
fd.append('local_path',pathInput.value);
}else if(source==='upload'){
const fileInput=document.getElementById('file-'+id);
if(!fileInput.files.length){alert('Choose a .onnx file first');return}
fd.append('file',fileInput.files[0]);
}else if(source==='download'){
const urlInput=document.getElementById('url-'+id);
if(urlInput&&urlInput.value)fd.append('download_url',urlInput.value);
}
const btn=document.getElementById('btn-'+id);
if(btn)btn.disabled=true;
const dl=document.getElementById('dl-'+id);
if(dl)dl.style.display='block';
try{
const res=await fetch(`/api/models/${id}/install`,{method:'POST',body:fd});
const data=await res.json();
if(!res.ok){alert(data.detail||'Install failed');if(dl)dl.style.display='none';if(btn)btn.disabled=false;return}
if(data.status==='downloading'){startPolling(id);return}
}catch(e){
if(dl)dl.style.display='none';
}
if(btn)btn.disabled=false;
loadModels();
}

async function fetchStatus(id){
try{
const res=await fetch(`/api/models/${id}/status`);
if(!res.ok)return null;
return await res.json();
}catch(e){return null}
}

function startPolling(id){
if(modelPollTimers[id])return;
const tick=async()=>{
const job=await fetchStatus(id);
if(!job){clearInterval(modelPollTimers[id]);delete modelPollTimers[id];return}
const dl=document.getElementById('dl-'+id);
const bar=document.getElementById('dl-bar-'+id);
const statusEl=document.getElementById('dl-status-'+id);
const bytesEl=document.getElementById('dl-bytes-'+id);
const errEl=document.getElementById('dl-error-'+id);
if(!dl||!bar)return;
dl.style.display='block';
if(job.state==='downloading'||job.state==='finalizing'){
if(job.bytes_total){
bar.classList.remove('indeterminate');
bar.style.width=Math.min(100,(job.bytes_done/job.bytes_total*100)).toFixed(1)+'%';
bytesEl.textContent=`${fmtBytes(job.bytes_done)} / ${fmtBytes(job.bytes_total)}`;
}else{
bar.classList.add('indeterminate');
bytesEl.textContent=fmtBytes(job.bytes_done);
}
statusEl.textContent=job.state==='finalizing'?'Finalizing...':'Downloading...';
errEl.style.display='none';
}else if(job.state==='done'){
clearInterval(modelPollTimers[id]);delete modelPollTimers[id];
loadModels();
}else if(job.state==='error'){
clearInterval(modelPollTimers[id]);delete modelPollTimers[id];
bar.classList.remove('indeterminate');
bar.style.width='0%';
statusEl.textContent='Failed';
errEl.textContent=job.error||'Download failed';
errEl.style.display='block';
const btn=document.getElementById('btn-'+id);
if(btn)btn.disabled=false;
}
};
modelPollTimers[id]=setInterval(tick,700);
tick();
}

async function deleteModel(id){
await fetch(`/api/models/${id}`,{method:'DELETE'});
loadModels();
}

async function loadSettings(){
const res=await fetch('/api/config');
const cfg=await res.json();
const f=document.getElementById('settings-form');
f.top_k.value=cfg.top_k;
f.reconstruction_mode_default.value=cfg.reconstruction_mode_default;
f.prefer_gpu.checked=cfg.prefer_gpu;
f.min_det_score.value=cfg.min_det_score;
f.gpen_model_path.value=cfg.gpen_model_path||'';
}

async function saveSettings(e){
e.preventDefault();
const f=e.target;
const body={top_k:parseInt(f.top_k.value),reconstruction_mode_default:f.reconstruction_mode_default.value,prefer_gpu:f.prefer_gpu.checked,min_det_score:parseFloat(f.min_det_score.value),gpen_model_path:f.gpen_model_path.value||null};
const res=await fetch('/api/config',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});
const data=await res.json();
document.getElementById('settings-result').innerHTML='<div class="warn" style="background:#1b3a2c;border-color:#2a5a3c;color:#5fd694">Saved. GPU/model-path changes apply after restart.</div>';
}

async function loadDashboard(){
const res=await fetch('/api/stats');
const s=await res.json();
document.getElementById('stat-gallery').textContent=s.gallery_size;
document.getElementById('stat-vindex').textContent=s.vector_index_status;
document.getElementById('stat-backend').textContent=s.vector_store_backend;
document.getElementById('stat-searches').textContent=s.recent_searches.length;
const logs=document.getElementById('recent-logs');
logs.innerHTML=s.recent_searches.map(l=>`<tr><td>${l.probe_id}</td><td>${l.reconstruction_mode}</td><td>${l.top_candidate||'-'}</td><td>${l.timestamp}</td></tr>`).join('')||'<tr><td colspan="4">No searches yet</td></tr>';
}