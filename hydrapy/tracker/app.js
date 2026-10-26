import {api, PreviewStream} from './backend.js';
import {noteMidi, noteFrequency, canonicalNote, breaksVoice} from './notes.js';
import {ScoreHistory, bounds, getCell, putCell, copyBlock, readBlock, pasteBlock, clearBlock,
  transposeBlock, setBlockVelocity, pianoNote, activity, SCENE, SCENE_NAME, kindOf} from './editor.js';
const $ = id => document.getElementById(id);
const clamp = (n,a,b) => Math.max(a,Math.min(b,n));
const mod = (n,m) => ((n%m)+m)%m;
const hex = n => n.toString(16).toUpperCase().padStart(2,'0');
const pad = (n,len=2) => String(n).padStart(len,'0');
const tracks = [
  {id:'drums',name:'Breaks',sub:'NOTE / VELOCITY',label:'Fracture kit',tag:'DRM',note:'C-2',color:'#a6dfbf'},
  {id:'fm',name:'FM hits',sub:'NOTE / VELOCITY',label:'Glass cluster',tag:'FM',note:'D-4',color:'#e2ac77'},
  {id:'sub',name:'Sub',sub:'NOTE / VELOCITY',label:'Sine foundation',tag:'SUB',note:'D-2',color:'#79b8b5'},
  {id:'air',name:'Texture',sub:'GATE / LEVEL',label:'Noise wash',tag:'ATM',note:'AIR',color:'#b69ab6'},
  {id:'z',name:'Dolly Z',sub:'CAMERA / DEPTH',visual:true,min:0,max:1},
  {id:'roll',name:'Roll',sub:'CAMERA / DEG',visual:true,min:-45,max:45},
  {id:'feedback',name:'Feedback',sub:'IMAGE / AMOUNT',visual:true,min:0,max:.9},
  {id:'scene',name:'Scene',sub:'PATCH SCENE / FADE',scene:true},
];
// The loaded patch's @scene names, from Studio ([{name, output}]).
let patchScenes=[];
const defaults = () => ({version:1,format:'hydrapy-tracker-1',bindings:{z:'Dolly Z',roll:'Roll',feedback:'Feedback'},bpm:138,key:2,sound:false,loop:false,pattern:0,scene:0,modes:{program:'o0',aux:'o1'},mutes:[false,false,false,false],patterns:['DRIFT','FRACTURE','ASCENT','SUSPEND'].map((name,p)=>{
  const audio = tracks.slice(0,4).map(()=>Array(64).fill(null));
  for(let r=0;r<64;r++){
    if(r%4===0 || (p===1 && r%16===14)) audio[0][r]={note:r%8===4?'D-2':'C-2',vel:r%8===4?100:115};
    if(r%4===2) audio[0][r]={note:'F#2',vel:54+(r%8)*3};
    if([6,14,22,31,38,46,54,62].includes(r) && !(p===3&&r%16<10)) audio[1][r]={note:['D-4','A-4','F-4','C-5'][Math.floor(r/8)%4],vel:64+(p*9)};
    if(r%16===0 || (p===2 && r%16===10)) audio[2][r]={note:r<32?'D-2':'A-1',vel:91};
    if(r%32===24) audio[3][r]={note:'AIR',vel:45};
  }
  return {name,audio,scenes:[],easing:{z:'smooth',roll:'smooth',feedback:'smooth'},keys:{
    z:[{r:0,v:[.35,.55,.2,.8][p]},{r:16,v:[.55,.25,.4,.7][p]},{r:32,v:[.45,.8,.7,.45][p]},{r:48,v:[.8,.5,.9,.25][p]},{r:63,v:[.35,.55,.2,.8][p]}],
    roll:[{r:0,v:0},{r:16,v:[-12,22,-6,14][p]},{r:32,v:[9,-18,19,-9][p]},{r:48,v:[-8,16,32,5][p]},{r:63,v:0}],
    feedback:[{r:0,v:.25},{r:24,v:.45+p*.08},{r:40,v:.72-p*.08},{r:56,v:.3},{r:63,v:.25}],
  }};
})});

let state=defaults();
let playing=false,anchorBeat=state.pattern*16,anchorTime=performance.now(),blackout=false,follow=true;
const PAGE_COLUMNS={audio:[0,1,2,3],visual:[4,5,6,7]};
let editorPage='audio';
try{if(sessionStorage.getItem('gorgon-editor-page')==='visual')editorPage='visual';}catch{}
const pageColumns={audio:0,visual:4};
let selected={row:0,col:pageColumns[editorPage]},curve='z',lastStep=-1,lastPattern=-1;
let switchingPage=false;
let editPattern=0, selectionAnchor=null, clipboard=null, pianoEntry=false, entryVelocity=90;
let curveDirty=true, curvePaint=null;
const history = new ScoreHistory();
let audioContext=null,master=null,noiseBuffer=null,nextTick=null,analyser=null,levelData=null;
let voices=new Set(),toastTimer=null;
const outputType=new URL(location.href).searchParams.get('output');
let sessionId=new URL(location.href).searchParams.get('session');
if(!sessionId){
  try{sessionId=sessionStorage.getItem('gorgon-session');}catch{}
  if(!sessionId){sessionId=typeof crypto.randomUUID==='function'?crypto.randomUUID():`${Date.now()}-${Math.random().toString(36).slice(2,9)}`;try{sessionStorage.setItem('gorgon-session',sessionId);}catch{}}
}
const channelName='gorgon-av-'+sessionId;
let channel=null;
try {channel=new BroadcastChannel(channelName);}catch{}
function send(message){
  if(channel) channel.postMessage(message);
  else try{localStorage.setItem(channelName,JSON.stringify({...message,nonce:Math.random()}));}catch{}
}
function listen(callback){
  if(channel)channel.onmessage=e=>callback(e.data);
  else window.addEventListener('storage',e=>{if(e.key===channelName&&e.newValue)try{callback(JSON.parse(e.newValue));}catch{}});
}
function normalizeBeat(beat,s=state){return s.loop?s.pattern*16+mod(beat-s.pattern*16,16):mod(beat,64);}
function rawBeat(){return anchorBeat+(playing?(performance.now()-anchorTime)*state.bpm/60000:0);}
function nowBeat(){return normalizeBeat(rawBeat());}
function valueAt(p,id,row){
  const keys=p.keys[id];if(!keys.length)return id==='z'?.35:id==='feedback'?.25:0;
  if(row<=keys[0].r)return keys[0].v;
  for(let i=1;i<keys.length;i++) if(row<keys[i].r){
    const a=keys[i-1],b=keys[i];let t=(row-a.r)/(b.r-a.r);
    const ease=p.easing[id];if(ease==='hold')t=0;else if(ease==='smooth')t=t*t*(3-2*t);
    return a.v+(b.v-a.v)*t;
  }
  return keys.at(-1).v;
}
function waveform(seed=0,bars=64){
  let out='';for(let i=0;i<bars;i++){const hit=(i+seed)%8===0;const h=hit?21:3+Math.abs(Math.sin(i*2.38+seed)*Math.cos(i*.43))*15;out+=`<path d="M${i*4+1} ${12-h/2}v${h.toFixed(1)}"/>`;}
  return `<svg viewBox="0 0 ${bars*4} 24" preserveAspectRatio="none" aria-hidden="true"><g stroke="currentColor" stroke-width="1.1">${out}</g></svg>`;
}
function patternActivity(pattern){
  const bars=activity(pattern);
  return `<svg class="activity-map" viewBox="0 0 256 24" preserveAspectRatio="none" aria-hidden="true">${bars.map((v,r)=>
    `<path d="M${r*4+1} 21v-${v.audio*4}" stroke="currentColor" stroke-width="2"/>`+
    (v.keys?`<path d="M${r*4+1} 2v${v.keys*2}" stroke="#e2ac77" stroke-width="2"/>`:'')).join('')}</svg>`;
}
function selectionBox(){return bounds(selected,selectionAnchor||selected);}
function focusCell(scroll=false){
  const el=$('trackerRows').querySelector(`[data-row="${selected.row}"][data-col="${selected.col}"]`);
  el?.focus({preventScroll:true});
  if(scroll)el?.scrollIntoView({block:'nearest',inline:'nearest'});
}
function setFollow(value){
  follow=value;$('followBtn').classList.toggle('active',follow);$('followBtn').setAttribute('aria-pressed',follow);
  lastPattern=-1;
}
function updateHistory(){
  $('undoBtn').disabled=!history.past.length;$('redoBtn').disabled=!history.future.length;
  $('undoBtn').title=history.past.length?`Undo ${history.past.at(-1).label} · Ctrl/Cmd+Z`:'Nothing to undo';
  $('redoBtn').title=history.future.length?`Redo ${history.future.at(-1).label} · Ctrl/Cmd+Shift+Z`:'Nothing to redo';
}
function redrawScore(){renderPatterns();renderHeaders();renderRows();renderInstruments();updateControls();updateInspector();updateHistory();}
function changeScore(label,mutate,merge=''){
  if(!remote){toast('Wait for Studio to connect before editing.');return false;}
  try{
    const changed=history.change(state,label,mutate,{pattern:editPattern,selected},merge);
    if(changed){redrawScore();saveLocal();}
    return changed;
  }catch(error){toast(error.message);return false;}
}
function restoreHistory(redo=false){
  const entry=redo?history.redo(state):history.undo(state);
  if(!entry)return;
  editPattern=entry.context.pattern;selected={...entry.context.selected};selectionAnchor=null;
  setFollow(false);stopVoices();redrawScore();saveLocal();focusCell(true);
  toast(`${redo?'Redid':'Undid'} ${entry.label}.`);
}
function toast(message){$('toast').textContent=message;$('toast').classList.add('visible');clearTimeout(toastTimer);toastTimer=setTimeout(()=>$('toast').classList.remove('visible'),4000);}
let remote = null, revision = 0, pendingScore = null, scoreRequest = null, scoreTimer;
let scoreFailed = false, failedDocument = null, acceptedDocument = null;
let transportQueue = Promise.resolve(), conflicted = false;
let transportPending = 0, transportEpoch = 0;
let transportRevision = null;
let previewOutput='program';
try{const saved=sessionStorage.getItem('gorgon-preview-output');if(['program','o0','o1','o2','o3'].includes(saved))previewOutput=saved;}catch{}
const preview = new PreviewStream(outputType === 'aux' ? 'aux' : 'program');
window.gorgonPreview = preview.stats;
const escapeHTML = value => String(value).replace(/[&<>"']/g, c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
function syncPreview(result=remote){
  if(!result)return;
  const inspect=!outputType&&previewOutput!=='program';
  const output=outputType==='aux'?result.document.modes.aux:inspect?previewOutput:result.output;
  preview.select(output,outputType==='aux'?result.aux_generation:result.generation,
    outputType==='aux'?'aux':inspect?'monitor':'program');
  if(outputType)return;
  $('monitorLabel').textContent=inspect?`PREVIEW ${output.toUpperCase()}`:'PROGRAM';
  $('sceneMeta').textContent=`${inspect?'PREVIEW':'PROGRAM'} · ${output.toUpperCase()}`;
  $('previewCanvas').setAttribute('aria-label',`Hydra ${output} ${inspect?'preview':'live Program preview'}`);
  document.querySelectorAll('button[data-preview-output]').forEach(button=>{
    const value=button.dataset.previewOutput;
    button.setAttribute('aria-pressed',String(value===previewOutput));
    button.classList.toggle('on-air',value===result.output);
    button.title=value==='program'?'Follow live Program':`Preview ${value}${value===result.output?' · live Program':''} · [ / ] cycle`;
  });
}
function selectPreview(output){
  previewOutput=output;
  try{sessionStorage.setItem('gorgon-preview-output',output);}catch{}
  syncPreview();
}
function saveLocal(){
  if (!remote) return;
  pendingScore = structuredClone(state);
  if(conflicted||scoreFailed)return;
  $('savedState').textContent = 'Applying edit…';
  // Studio says how long to gather edits: 120 ms normally, 0 with a native output, where an
  // edit goes at once and later ones ride with the next request while one is in flight.
  clearTimeout(scoreTimer); scoreTimer = setTimeout(flushScore, remote.edit_delay_ms ?? 120);
}
async function flushScore(){
  clearTimeout(scoreTimer);
  if (scoreRequest) { await scoreRequest; return flushScore(); }
  if(conflicted||scoreFailed)return false;
  if (!pendingScore) return true;
  const document = pendingScore; pendingScore = null;
  scoreRequest = api('tracker_score', {document, revision}).then(result => {
    revision = result.revision;
    acceptedDocument=structuredClone(result.document);
    // Studio may have switched its displayed output since this edit's base.
    // Keep that choice unless the user has changed the program route again.
    const program = result.document.modes.program;
    if (state.modes.program === document.modes.program) state.modes.program = program;
    if (pendingScore && pendingScore.modes.program === document.modes.program) pendingScore.modes.program = program;
    if (program !== document.modes.program) updateControls();
    $('savedState').textContent = 'Score applied · not saved';
  }).catch(error => {
    pendingScore=pendingScore||structuredClone(state);failedDocument=document;
    conflicted=!!error.status&&error.status<500;scoreFailed=true;
    $('savedState').textContent=conflicted?'Edit rejected · recovery available':'Edit pending · retry';
    $('editConnection').hidden=false;
    $('editConnectionText').textContent=conflicted?`${error.message} Your local edits are preserved for download.`:'Connection interrupted. Local edits have not been confirmed by Studio.';
    $('retryScoreBtn').hidden=conflicted;
    toast(error.message);
  });
  await scoreRequest; scoreRequest = null;
  if(conflicted||scoreFailed)return false;
  if (pendingScore) return flushScore();
  return true;
}
function sameScore(a,b){
  if(!a||!b)return false;
  const left=structuredClone(a),right=structuredClone(b);
  left.modes.program=right.modes.program='o0';
  return JSON.stringify(left)===JSON.stringify(right);
}
async function retryScore(){
  try{
    const result=await api('tracker');
    if(!sameScore(result.document,acceptedDocument)&&!sameScore(result.document,failedDocument)){
      conflicted=true;$('retryScoreBtn').hidden=true;
      throw new Error('Studio has different edits. Download your recovery copy before choosing Use server score.');
    }
    const program=failedDocument?.modes.program;
    if(state.modes.program===program)state.modes.program=result.document.modes.program;
    revision=result.revision;acceptedDocument=structuredClone(result.document);
    scoreFailed=conflicted=false;pendingScore=structuredClone(state);
    $('editConnection').hidden=true;
    await flushScore();
  }catch(error){$('editConnectionText').textContent=error.message;toast(error.message);}
}
function downloadScore(score=state,name='gorgon-recovery.json'){
  const u=URL.createObjectURL(new Blob([JSON.stringify(score,null,2)],{type:'application/json'}));
  const a=document.createElement('a');a.href=u;a.download=name;a.click();setTimeout(()=>URL.revokeObjectURL(u),1000);
}
function adoptTransport(result, elapsed=0){
  const old = rawBeat(), length = state.loop ? 16 : 64;
  const next = normalizeBeat(result.beat + (result.playing ? elapsed * state.bpm / 60 : 0));
  // Keep the audio scheduler on an unwrapped beat across ordinary loop wraps.
  // Explicit seeks still cancel its lookahead, even for a short seek.
  const error = mod(next-normalizeBeat(old)+length/2,length)-length/2;
  const discontinuity = playing !== result.playing || Math.abs(error) > .6 ||
    (result.transport_revision !== undefined && result.transport_revision !== transportRevision);
  if (discontinuity) stopVoices();
  anchorBeat=discontinuity ? next : old+error;
  anchorTime=performance.now(); playing=result.playing;
  transportRevision=result.transport_revision ?? transportRevision;
  if (result.blackout !== undefined) blackout=result.blackout;
  $('blackoutBadge').hidden=!blackout;
  $('blackoutBtn').setAttribute('aria-pressed',blackout);
  $('playBtn').setAttribute('aria-label',playing?'Pause':'Play');
  $('playBtn').innerHTML=playing?'<svg viewBox="0 0 20 20"><path d="M5 3h4v14H5Zm7 0h4v14h-4Z" fill="currentColor"/></svg>':'<svg viewBox="0 0 20 20"><path d="m6 3 11 7-11 7Z" fill="currentColor"/></svg>';
  $('transportStatus').textContent=playing?'PLAYING':result.enabled?'PAUSED':'AUTOMATION OFF';
  $('transportDot').classList.toggle('mint',playing);
  $('automationBtn').setAttribute('aria-pressed',result.enabled);
  $('automationBtn').textContent=result.enabled?'Automation on':'Automation off';
}
function transport(changes,{score=true}={}){
  transportPending++;transportEpoch++;
  transportQueue=transportQueue.then(async()=>{
    const request=typeof changes==='function'?changes():changes;
    // A rejected edit must never trap the operator with a running or visible output.
    const pause=request.playing===false&&request.beat===undefined;
    if(score&&!pause&&!await flushScore())throw new Error('Resolve the pending edit before changing transport. Your recovery copy is available in File.');
    const result=await api('tracker_transport',request);
    adoptTransport(result); if(remote) Object.assign(remote,result);
  }).catch(error=>toast(error.message)).finally(()=>{transportPending--;transportEpoch++;});
  return transportQueue;
}
function renderBindings(controls){
  for(const lane of ['z','roll','feedback']){
    const select=$('bind_'+lane), name=state.bindings[lane];
    const options=[['','Disconnected'],...controls.map(p=>[p.name,p.name])];
    if(name&&!controls.some(p=>p.name===name))options.push([name,name+' (missing)']);
    const signature=JSON.stringify(options);
    if(select.dataset.options!==signature){
      select.replaceChildren(...options.map(([value,label])=>new Option(label,value)));
      select.dataset.options=signature;
    }
    select.value=name;
    for(const option of select.options){option.disabled=!!option.value&&Object.entries(state.bindings).some(([other,bound])=>other!==lane&&bound===option.value);}
    select.classList.toggle('missing',!!name&&!controls.some(p=>p.name===name));
  }
}
let jackState=null;
// Display menu: Program fullscreen from Studio's own window, drawn straight from the GPU (Linux).
let nativeState=null;
function renderNative(native){
  nativeState=native||null;
  const menu=$('displayMenu');
  menu.hidden=!native||(!native.available&&!native.supported);
  if(menu.hidden)return;
  const select=$('nativeMonitor'), button=$('nativeToggleBtn');
  const options=native.available?[...native.monitors.map(m=>[m.name,`${m.name} · ${m.width}×${m.height}${m.refresh?` · ${m.refresh} Hz`:''}${m.primary?' · primary':''}`]),['window','A window on this desktop']]:[];
  const key=JSON.stringify(options);
  if(select.dataset.options!==key){
    select.innerHTML=options.map(([value,label])=>`<option value="${escapeHTML(value)}">${escapeHTML(label)}</option>`).join('');
    select.dataset.options=key;select.dataset.chosen='';
  }
  const want=native.where||select.dataset.chosen||native.monitor||native.suggested;
  if(document.activeElement!==select&&options.some(([value])=>value===want))select.value=want;
  select.disabled=button.disabled=!native.available;
  button.setAttribute('aria-pressed',String(!!native.active));
  const windowed=(native.active?native.where:select.value)==='window';
  button.textContent=native.active?(windowed?'Close Program window':'Close fullscreen output')
    :(windowed?'Show Program in a window':'Show Program fullscreen');
  $('nativeStatus').textContent=!native.available?native.note:native.active
    ?`${native.where} · ${native.size.join(' × ')} · ${Math.round(native.refresh)} Hz · budget ${native.budget_ms==null?'–':native.budget_ms.toFixed(1)} ms · ${native.missed} missed`
    :'Drawn straight from the GPU at the display’s refresh. Studio renders at the monitor’s size while it shows.';
}
function renderJackAudio(audio){
  if(!audio)return;
  jackState=audio;
  $('jackEnableBtn').textContent=audio.active?'Disable JACK audio ports':audio.error?'Retry JACK audio':'Enable JACK audio ports';
  $('jackEnableBtn').setAttribute('aria-pressed',audio.active);
  if(document.activeElement!==$('opticalGain'))$('opticalGain').value=audio.config.optical_gain;
  if(document.activeElement!==$('breaksGain'))$('breaksGain').value=audio.config.breaks_gain;
  if(document.activeElement!==$('audioLead'))$('audioLead').value=audio.config.lead_blocks;
  $('jackStatus').textContent=audio.error||(!audio.active?'Audio ports disabled.':`${audio.samplerate} Hz · ${audio.blocksize} frames · buffer lead ${audio.buffer_lead_ms.toFixed(1)} ms · ${audio.xruns} xruns · ${audio.underruns} source underruns`);
  $('jackPorts').textContent=audio.ports.join('\n');
}
function renderPolarity(routes){
  const parent=$('polarityControls'),signature=JSON.stringify(routes.map(({id,name,mode})=>({id,name,mode})));
  if(parent.dataset.signature!==signature){
    parent.replaceChildren();parent.dataset.signature=signature;
    for(const route of routes){
      const label=document.createElement('label'),name=document.createElement('span'),select=document.createElement('select');
      name.textContent=route.name;select.setAttribute('aria-label','ES-40 polarity '+route.name);
      for(const [value,text] of [['normal','Normal polarity'],['invert_left','Invert left'],['invert_right','Invert right'],['invert_both','Invert both']])select.add(new Option(text,value));
      select.value=route.mode;
      select.onchange=async()=>{select.disabled=true;try{await api('cv_polarity',{route:route.id,mode:select.value});}catch(error){select.value=route.mode;toast(error.message);}finally{select.disabled=false;}};
      label.append(name,select);parent.append(label);
    }
  }
  const active=routes.filter(r=>r.mode!=='normal'),clipped=routes.reduce((n,r)=>n+r.clipped,0);
  $('polarityStatus').textContent=active.length?`ES-40 polarity inversion active on ${active.length} pair(s).${clipped?' Full-scale clipping encountered; verify the digital path.':''}`:'';
}
async function pollRemote(){
  const started=performance.now(), epoch=transportEpoch;
  try{
    const result=await api('tracker');
    if(!result.document)throw new Error('Studio is starting.');
    if(result.server_now-result.sampled>3)throw new Error('Studio renderer is not responding.');
    if(!remote||(!conflicted&&!scoreFailed&&!pendingScore&&!scoreRequest&&result.revision>revision)){
      if(!remote){editPattern=result.document.pattern;$('savedState').textContent='Live score · save to keep';}
      else if(!sameScore(result.document,acceptedDocument)){history.clear();updateHistory();}
      state=result.document;revision=result.revision;
      acceptedDocument=structuredClone(result.document);
      if(!outputType){renderPatterns();renderHeaders();renderRows();renderInstruments();updateControls();updateInspector();}
    }
    if(!transportPending&&epoch===transportEpoch){
      remote=result;
      adoptTransport(result, Math.max(0,result.server_now-result.sampled)+(performance.now()-started)/2000);
    }else if(!remote)remote=result;
    syncPreview(result);
    if(!outputType){
      renderBindings(result.controls);
      if(JSON.stringify(result.scenes??[])!==JSON.stringify(patchScenes)){
        patchScenes=result.scenes??[];renderRows();updateInspector();
      }
      $('sceneName').textContent=result.path?.split('/').pop()||'Choose a patch';
    }
    $('connectionStatus').textContent=result.playing?'Transport running':'Transport paused';
    const native=result.native?.active?` · ${result.native.where==='window'?'native window':'fullscreen '+result.native.where} ${Math.round(result.native.refresh)} Hz`:'';
    $('backendStatus').textContent=result.error||(result.path?`${result.backend} · ${result.fps.toFixed(1)} render fps${native} · ${result.path}`:'Load a patch to begin');
    // While Program shows fullscreen, this monitor is a small confidence preview of it.
    $('resolution').textContent=result.native?.active?`${result.size.join(' × ')} · ${result.native.where==='window'?'in a window':'on '+result.native.where} · preview`:result.size.join(' × ');
    $('cvStatus').textContent=!result.cv?'CV off · configure at launch':result.cv.error?`CV error: ${result.cv.error}`:`CV active · ${result.cv.underflows||0} underflows`;
    $('cvStatus').title=result.cv?JSON.stringify(result.cv,null,2):'';
    $('cvPorts').textContent=(result.cv?.ports||[]).join('\n');
    if(!outputType){renderPolarity(result.cv?.polarity_routes||[]);renderJackAudio(result.audio);renderNative(result.native);}
  }catch(error){
    playing=false;stopVoices();$('backendStatus').textContent=error.message;
    $('connectionStatus').textContent='Studio disconnected';
  }finally{setTimeout(pollRemote,150);}
}
async function fullscreen(){
  try{if(document.fullscreenElement)await document.exitFullscreen();else await document.documentElement.requestFullscreen();}
  catch(error){console.warn('Fullscreen request unavailable:',error.name,error.message);toast('Fullscreen is unavailable in this view. Open the site in a browser window and try F again.');}
}
function sizeCanvas(canvas,ratio=1.4){
  const box=canvas.getBoundingClientRect();const d=Math.min(devicePixelRatio||1,ratio);const w=Math.max(1,Math.round(box.width*d)),h=Math.max(1,Math.round(box.height*d));
  if(canvas.width!==w||canvas.height!==h){canvas.width=w;canvas.height=h;}return [w,h];
}

function drawVisual(canvas,beat,mode='full',s=state,isBlack=false){preview.draw(canvas,isBlack);}

function ensureAudio(){
  if(!audioContext){
    const AC=window.AudioContext||window.webkitAudioContext;if(!AC){toast('Audio is unavailable in this browser. Visual transport still works.');return Promise.resolve();}
    audioContext=new AC();master=audioContext.createGain();master.gain.value=state.sound?.23:0;
    const limiter=audioContext.createDynamicsCompressor();limiter.threshold.value=-12;limiter.knee.value=10;limiter.ratio.value=8;analyser=audioContext.createAnalyser();analyser.fftSize=256;levelData=new Uint8Array(analyser.fftSize);master.connect(limiter);limiter.connect(analyser);analyser.connect(audioContext.destination);
    noiseBuffer=audioContext.createBuffer(1,audioContext.sampleRate*2,audioContext.sampleRate);const d=noiseBuffer.getChannelData(0);let seed=17421;for(let i=0;i<d.length;i++){seed=(seed*16807)%2147483647;d[i]=seed/1073741823.5-1;}
  }
  return audioContext.resume().catch(()=>toast('Click Sound on to enable browser audio.'));
}
function stopVoices(){for(const n of voices){try{n.stop();}catch{}}voices.clear();nextTick=null;}
function connectVoice(source,gain,t,duration){source.connect(gain);gain.connect(master);voices.add(source);source.onended=()=>{voices.delete(source);source.disconnect();gain.disconnect();};source.start(t);source.stop(t+duration);}
function noteFreq(note){return noteFrequency(note, state.key);}
function soundHit(track,hit,t){
  if(!audioContext||!master||!hit)return;const vel=hit.vel/127;if(vel<=0)return;if((track===1||track===2)&&noteMidi(hit.note)===null)return;
  const drum=track===0?breaksVoice(hit.note):null;
  if(drum==='kick'){
    const o=audioContext.createOscillator(),g=audioContext.createGain();o.frequency.setValueAtTime(145,t);o.frequency.exponentialRampToValueAtTime(44,t+.13);g.gain.setValueAtTime(.95*vel,t);g.gain.exponentialRampToValueAtTime(.001,t+.29);connectVoice(o,g,t,.3);
  }else if(track===0){
    const snare=drum==='snare',n=audioContext.createBufferSource(),f=audioContext.createBiquadFilter(),g=audioContext.createGain();n.buffer=noiseBuffer;f.type='highpass';f.frequency.value=snare?1000:7200;n.connect(f);f.connect(g);g.connect(master);g.gain.setValueAtTime(vel*(snare?.44:.18),t);g.gain.exponentialRampToValueAtTime(.001,t+(snare?.15:.04));voices.add(n);n.onended=()=>{voices.delete(n);n.disconnect();f.disconnect();g.disconnect();};n.start(t,mod(t*.17,1));n.stop(t+(snare?.17:.06));
  }else if(track===1){
    const o=audioContext.createOscillator(),m=audioContext.createOscillator(),mg=audioContext.createGain(),g=audioContext.createGain();const freq=noteFreq(hit.note);o.frequency.value=freq;m.frequency.value=freq*2;mg.gain.setValueAtTime(freq*2,t);mg.gain.exponentialRampToValueAtTime(1,t+.3);m.connect(mg);mg.connect(o.frequency);g.gain.setValueAtTime(.17*vel,t);g.gain.exponentialRampToValueAtTime(.001,t+.5);connectVoice(o,g,t,.52);voices.add(m);m.onended=()=>{voices.delete(m);m.disconnect();mg.disconnect();};m.start(t);m.stop(t+.52);
  }else if(track===2){
    const o=audioContext.createOscillator(),g=audioContext.createGain();o.frequency.value=noteFreq(hit.note);g.gain.setValueAtTime(.001,t);g.gain.linearRampToValueAtTime(.38*vel,t+.01);g.gain.exponentialRampToValueAtTime(.001,t+.65);connectVoice(o,g,t,.7);
  }else{
    const n=audioContext.createBufferSource(),f=audioContext.createBiquadFilter(),g=audioContext.createGain();n.buffer=noiseBuffer;f.type='bandpass';f.frequency.value=1400;f.Q.value=2;n.connect(f);f.connect(g);g.connect(master);g.gain.setValueAtTime(.001,t);g.gain.linearRampToValueAtTime(.13*vel,t+.12);g.gain.exponentialRampToValueAtTime(.001,t+.9);voices.add(n);n.onended=()=>{voices.delete(n);n.disconnect();f.disconnect();g.disconnect();};n.start(t);n.stop(t+.95);
  }
}
function scheduler(){
  if(!playing||!state.sound||audioContext?.state!=='running')return;
  const raw=rawBeat();if(nextTick===null)nextTick=Math.ceil(raw*4-.08);
  // Skip missed ticks after a sleeping tab; never replay a backlog of sounds.
  if(nextTick/4<raw-.1)nextTick=Math.ceil(raw*4);
  while(nextTick/4<raw+.11*state.bpm/60){
    const b=normalizeBeat(nextTick/4),p=state.patterns[Math.floor(b/16)],r=Math.round(mod(b,16)*4)%64;
    const t=audioContext.currentTime+Math.max(.005,(nextTick/4-raw)*60/state.bpm);
    p.audio.forEach((a,i)=>{if(a[r]&&!state.mutes[i])soundHit(i,a[r],t);});nextTick++;
  }
}
function seek(beat){stopVoices();return transport({beat:normalizeBeat(beat)});}
async function togglePlay(){
  if(!playing&&state.sound)await ensureAudio();
  return transport(()=>({playing:!playing,enabled:true}));
}
function renderPatterns(){
  $('patterns').innerHTML=state.patterns.map((p,i)=>`<div class="pattern-slot"><button class="pattern-card ${i===editPattern?'selected':''}" data-pattern="${i}" aria-label="Select pattern ${i+1}: ${escapeHTML(p.name)}" aria-pressed="${i===editPattern}"><div class="pattern-top"><b>${pad(i+1)}</b><span>${escapeHTML(p.name)}</span><span class="bars">${pad(i*4+1)}—${pad(i*4+4)}</span></div>${patternActivity(p)}<i class="pattern-playhead"></i></button><button class="pattern-launch" data-launch="${i}" aria-label="Play from pattern ${i+1}" title="Launch pattern ${i+1}">▶</button></div>`).join('');
  $('patterns').querySelectorAll('[data-pattern]').forEach(b=>b.onclick=()=>selectPattern(Number(b.dataset.pattern)));
  $('patterns').querySelectorAll('[data-launch]').forEach(b=>b.onclick=()=>launchPattern(Number(b.dataset.launch)));
}
function syncEditorPage(){
  const next=selected.col<4?'audio':'visual';
  if(next!==editorPage){editorPage=next;selectionAnchor=null;}
  pageColumns[editorPage]=selected.col;
  document.body.dataset.editorPage=editorPage;
  try{sessionStorage.setItem('gorgon-editor-page',editorPage);}catch{}
  document.querySelectorAll('button[data-editor-page]').forEach(button=>{
    const active=button.dataset.editorPage===editorPage;
    button.setAttribute('aria-selected',active);button.tabIndex=active?0:-1;
  });
  $('laneEditor').setAttribute('aria-labelledby',editorPage+'PageTab');
  $('trackerRows').setAttribute('aria-label',editorPage==='audio'?'Audio pattern editor':'Visual pattern editor');
  $('trackerRows').setAttribute('aria-colcount','5');
  $('trackerRows').setAttribute('aria-description','Tab switches Audio and Visual pages. Shift+Tab leaves the grid.');
  $('libraryBtn').textContent=editorPage==='audio'?'Instruments':'Visual source';
  $('libraryBtn').title=editorPage==='audio'?'Show or hide instruments':'Show or hide the visual source';
}
function selectEditorPage(next,{focus=false}={}){
  if(!PAGE_COLUMNS[next])return;
  pageColumns[editorPage]=selected.col;
  selected.col=pageColumns[next];selectionAnchor=null;
  switchingPage=true;
  try{
    renderHeaders();renderRows();renderInstruments();updateInspector();
    if(focus)focusCell(true);
  }finally{switchingPage=false;}
}
function renderHeaders(){
  syncEditorPage();
  $('trackHeaders').innerHTML='<div class="track-header row-index">ROW</div>'+PAGE_COLUMNS[editorPage].map(i=>{const t=tracks[i];return `<div class="track-header ${t.visual?'visual':''} ${t.scene?'scene':''}"><div><strong>${t.name}</strong>${i<4?`<button class="mute" data-mute="${i}" aria-label="Mute ${t.name}" aria-pressed="${state.mutes[i]}">M</button>`:''}</div><span>${t.sub}</span></div>`;}).join('');
  $('trackHeaders').querySelectorAll('[data-mute]').forEach(b=>b.onclick=()=>{
    changeScore('mute lane',()=>{state.mutes[Number(b.dataset.mute)]=!state.mutes[Number(b.dataset.mute)];});stopVoices();
  });
}
function formatValue(id,v){return id==='roll'?`${v>=0?'+':''}${Math.round(v)}°`:v.toFixed(2);}
function sceneLabel(key){return `${escapeHTML(key.name)}${key.fade?`<span class="velocity">F${hex(key.fade)}</span>`:''}`;}
// Enter on an empty SCENE cell: the scene after the one playing above it, so repeated entries step through.
function nextSceneName(p,row){
  const names=patchScenes.map(x=>x.name);if(!names.length)return 'scene';
  const above=[...(p.scenes??[])].reverse().find(k=>k.r<row);
  return names[(names.indexOf(above?.name)+1)%names.length];
}
function renderRows(){
  if((selected.col<4?'audio':'visual')!==editorPage)renderHeaders();
  const grid=$('trackerRows'),hadFocus=grid.contains(document.activeElement),p=state.patterns[editPattern],box=selectionBox();let html='';
  for(let r=0;r<64;r++){
    html+=`<div class="tracker-row ${r%4===0?'beat':''} ${r%16===0?'bar':''}" role="row" data-row="${r}"><button class="row-number" role="rowheader" tabindex="-1" data-seek-row="${r}" title="Double-click to seek to row ${hex(r)}">${hex(r)}</button>`;
    PAGE_COLUMNS[editorPage].forEach((c,index)=>{
      const t=tracks[c];
      const key=getCell(p,r,c),isSelected=selected.row===r&&selected.col===c;
      const inBlock=r>=box.r0&&r<=box.r1&&c>=box.c0&&c<=box.c1;
      const content=key?(t.scene?sceneLabel(key):t.visual?formatValue(t.id,key.v):`${escapeHTML(key.note)}<span class="velocity">${hex(key.vel)}</span>`):'···';
      const missing=t.scene&&key&&remote&&!patchScenes.some(x=>x.name===key.name);
      html+=`<button role="gridcell" aria-colindex="${index+2}" tabindex="${isSelected?'0':'-1'}" class="cell ${t.visual?'visual':''} ${t.scene?'scene':''} ${missing?'missing':''} ${!key?'empty':''} ${isSelected?'selected':''} ${inBlock?'in-selection':''} ${c<4&&state.mutes[c]?'muted':''}" data-row="${r}" data-col="${c}" aria-label="${t.name}, row ${hex(r)}, ${key?(t.scene?escapeHTML(key.name)+(missing?' (not in this patch)':''):t.visual?formatValue(t.id,key.v):escapeHTML(key.note)):'empty'}" aria-selected="${inBlock}">${content}</button>`;
    });html+='</div>';
  }
  grid.innerHTML=html;lastStep=-1;
  if(hadFocus)focusCell();
  updateSelection();
}
function updateSelection(){
  const box=selectionBox();
  $('trackerRows').querySelectorAll('[data-col]').forEach(el=>{
    const r=Number(el.dataset.row),c=Number(el.dataset.col),active=r===selected.row&&c===selected.col;
    const inside=r>=box.r0&&r<=box.r1&&c>=box.c0&&c<=box.c1;
    el.classList.toggle('selected',active);el.classList.toggle('in-selection',inside);
    el.setAttribute('aria-selected',inside);el.tabIndex=active?0:-1;
  });
  const count=(box.r1-box.r0+1)*(box.c1-box.c0+1);
  $('selectionInfo').textContent=count===1?'1 cell':`${box.r1-box.r0+1} × ${box.c1-box.c0+1} cells`;
  document.querySelectorAll('[data-transpose]').forEach(el=>{el.disabled=box.c0>=4;});
}
function renderInstruments(){
  $('instruments').innerHTML=tracks.slice(0,4).map((t,i)=>`<button class="instrument ${selected.col===i?'active':''}" data-instrument="${i}" title="Audition ${t.label}"><span class="instrument-name"><strong>${t.label}</strong><small>${t.tag}</small></span>${waveform(i*7,38)}</button>`).join('');
  $('instruments').querySelectorAll('button').forEach(b=>b.onclick=async()=>{
    selectCell(selected.row,Number(b.dataset.instrument));await ensureAudio();
    if(master&&state.sound)soundHit(selected.col,{note:tracks[selected.col].note,vel:90},audioContext.currentTime+.005);
    renderInstruments();focusCell(true);
  });
}
function selectPattern(index){
  editPattern=index;selectionAnchor=null;setFollow(false);
  renderPatterns();renderRows();updateInspector();renderInstruments();
}
function launchPattern(index){
  selectPattern(index);
  if(state.loop){state.pattern=index;saveLocal();}
  if(state.sound)ensureAudio();
  return transport({beat:index*16,playing:true,enabled:true});
}
function seekSelected(){
  if(state.loop&&state.pattern!==editPattern){state.pattern=editPattern;saveLocal();updateControls();}
  return seek(editPattern*16+selected.row/4);
}
function selectCell(row,col,extend=false){
  if(extend){if(!selectionAnchor)selectionAnchor={...selected};}else selectionAnchor=null;
  selected={row,col};if(tracks[col].visual)curve=tracks[col].id;
  pageColumns[editorPage]=col;
  if(playing&&follow)setFollow(false);
  updateSelection();updateInspector();
}
function toggleCell(){
  changeScore('toggle cell',()=>{
    const p=state.patterns[editPattern],t=tracks[selected.col];
    const value=getCell(p,selected.row,selected.col)?null:t.scene?{name:nextSceneName(p,selected.row),fade:0}:t.visual?{v:valueAt(p,t.id,selected.row)}:{note:t.note,vel:entryVelocity};
    putCell(p,selected.row,selected.col,value);
  });
}
function clearCell(){changeScore('clear cells',()=>clearBlock(state.patterns[editPattern],selectionBox()));}
function setKey(id,value){
  const t=tracks.find(t=>t.id===id),v=clamp(value,t.min,t.max);
  if(!Number.isFinite(v)){toast('Enter a finite key value.');return;}
  curve=id;selected.col=tracks.findIndex(t=>t.id===id);selectionAnchor=null;
  if(!changeScore('automation value',()=>putCell(state.patterns[editPattern],selected.row,selected.col,{v}),`key:${editPattern}:${id}:${selected.row}`)){
    renderRows();updateInspector();
  }
}
function updateInspector(){
  if((selected.col<4?'audio':'visual')!==editorPage){renderHeaders();renderRows();renderInstruments();}
  pageColumns[editorPage]=selected.col;
  curveDirty=true;
  const p=state.patterns[editPattern];$('patternTitle').textContent=`${pad(editPattern+1)} / ${p.name}`;
  $('patternName').value=p.name;
  const options=state.patterns.map((pattern,i)=>new Option(`${pad(i+1)} / ${pattern.name}`,String(i)));
  const previous=$('duplicateTarget').value;
  $('duplicateTarget').replaceChildren(...options);
  $('duplicateTarget').value=previous!==''&&Number(previous)!==editPattern?previous:String((editPattern+1)%4);
  $('keyLocation').textContent=`KEY / ${hex(selected.row)}`;
  ['z','roll','feedback'].forEach(id=>{const v=valueAt(p,id,selected.row);$(id+'Slider').value=v;$(id+'Value').textContent=formatValue(id,v);});
  $('selectedTrack').textContent=tracks[selected.col].name;$('selectedRow').textContent=`P${editPattern+1} / ROW ${hex(selected.row)}`;
  const hit=selected.col<4?p.audio[selected.col][selected.row]:null;
  const kind=kindOf(selected.col);
  $('noteEditor').hidden=kind!=='audio';$('visualCellEditor').hidden=!tracks[selected.col].visual;$('sceneEditor').hidden=kind!=='scene';
  $('noteInput').value=hit?.note||tracks[selected.col].note||'C-4';
  $('velocityInput').value=hit?.vel??entryVelocity;$('auditionBtn').disabled=!hit;
  $('selectedHint').textContent=kind==='audio'?'Enter toggles · Note and velocity write on change':
    kind==='scene'?(patchScenes.length?'Enter steps to the next scene · Fade crossfades over rows':'Declare @scene in the patch to choose scenes'):
    'Writes at the edit cursor; playback stays in place.';
  if(kind==='scene'){
    const key=getCell(p,selected.row,SCENE),choice=$('sceneChoice');
    const names=[...new Set([...patchScenes.map(x=>x.name),...(key?[key.name]:[])])];
    choice.replaceChildren(...names.map(n=>new Option(patchScenes.some(x=>x.name===n)?`${n} · ${patchScenes.find(x=>x.name===n).output}`:`${n} (not in patch)`,n)));
    choice.value=key?.name??nextSceneName(p,selected.row);$('sceneFade').value=key?.fade??0;
    $('writeSceneBtn').disabled=!names.length&&!key;
  }
  if(tracks[selected.col].visual){
    const t=tracks[selected.col],input=$('cellValue');input.min=t.min;input.max=t.max;input.step=t.id==='roll'?1:.01;input.value=valueAt(p,t.id,selected.row).toFixed(t.id==='roll'?0:2);
  }
  $('curveTabs').querySelectorAll('button').forEach(b=>b.classList.toggle('active',b.dataset.curve===curve));
  $('easeButtons').querySelectorAll('button').forEach(b=>b.classList.toggle('active',b.dataset.ease===p.easing[curve]));
  $('curveValue').textContent=formatValue(curve,valueAt(p,curve,selected.row));$('curveUnit').textContent=curve==='roll'?'DEGREES':curve==='z'?'SCALE':'AMOUNT';
  updateHistory();
}
function drawCurve(beat){
  const c=$('curveCanvas');if(!c.getClientRects().length)return;
  const [w,h]=sizeCanvas(c),position=Math.floor(beat/16)===editPattern?beat:null;
  if(!curveDirty&&curvePaint?.w===w&&curvePaint?.h===h&&curvePaint?.position===position)return;
  curveDirty=false;curvePaint={w,h,position};
  const ctx=c.getContext('2d'),p=state.patterns[editPattern],t=tracks.find(t=>t.id===curve),left=32,right=w-16,top=18,bottom=h-23;
  ctx.clearRect(0,0,w,h);ctx.fillStyle='#121b16';ctx.fillRect(0,0,w,h);ctx.font='9px ui-monospace, monospace';ctx.lineWidth=.6;
  for(let i=0;i<=4;i++){const y=top+(bottom-top)*i/4;ctx.strokeStyle='#2a3b2e';ctx.beginPath();ctx.moveTo(left,y);ctx.lineTo(right,y);ctx.stroke();if(i===0||i===4){ctx.fillStyle='#546b59';ctx.fillText((t.max-(t.max-t.min)*i/4).toFixed(t.id==='roll'?0:1),3,y+3);}}
  for(let i=0;i<5;i++){const x=left+(right-left)*i/4;ctx.strokeStyle='#344431';ctx.beginPath();ctx.moveTo(x,top);ctx.lineTo(x,bottom);ctx.stroke();ctx.fillStyle='#637c66';ctx.fillText(`${pad(editPattern*4+i+1)}`,x-5,h-7);}
  const point=(r,v)=>[left+r/63*(right-left),bottom-(v-t.min)/(t.max-t.min)*(bottom-top)];
  ctx.beginPath();for(let r=0;r<=63;r+=.25){const xy=point(r,valueAt(p,curve,r));if(r===0)ctx.moveTo(...xy);else ctx.lineTo(...xy);}ctx.strokeStyle='#d6b27e';ctx.lineWidth=1.5;ctx.stroke();ctx.lineTo(right,bottom);ctx.lineTo(left,bottom);ctx.closePath();ctx.fillStyle='#d0ab6a0b';ctx.fill();
  p.keys[curve].forEach(k=>{const [x,y]=point(k.r,k.v);ctx.fillStyle=k.r===selected.row?'#f4d79c':'#a58959';ctx.fillRect(x-2.5,y-2.5,5,5);});
  if(Math.floor(beat/16)===editPattern){const x=left+mod(beat,16)/16*(right-left);ctx.beginPath();ctx.moveTo(x,top-5);ctx.lineTo(x,bottom+4);ctx.strokeStyle='#a5d3a7';ctx.lineWidth=.8;ctx.stroke();}
}
function updateControls(){
  if(master)master.gain.setTargetAtTime(state.sound?.23:0,audioContext.currentTime,.02);
  $('tempo').value=state.bpm;$('key').value=state.key;$('audioBtn').setAttribute('aria-pressed',state.sound);$('audioLabel').textContent=state.sound?'Sound on':'Sound off';$('loopBtn').setAttribute('aria-pressed',state.loop);$('loopBtn').classList.toggle('active',state.loop);
  $('loopBtn').title=state.loop?`Looping pattern ${state.pattern+1}; click to disable`:`Loop selected pattern ${editPattern+1}`;
  $('programMode').value=state.modes.program;$('auxMode').value=state.modes.aux;$('sceneName').textContent=remote?.path?.split('/').pop()||'Choose a patch';syncPreview();
  const sec=64*60/state.bpm;$('duration').textContent=`${pad(Math.floor(sec/60))}:${(sec%60).toFixed(1).padStart(4,'0')}`;
}

if(outputType){
  $('studio').hidden=true;$('outputView').hidden=false;
  const kind=outputType==='aux'?'aux':'program';
  $('outputName').textContent=kind==='aux'?'AUXILIARY / 03':'PROGRAM / 02';
  $('outputWelcomeTitle').textContent=kind==='aux'?'Auxiliary / 03':'Program / 02';
  document.title=`Gorgon — ${kind} output`;
  const announce=()=>send({type:'presence',kind,at:Date.now()});announce();setInterval(announce,1000);
  $('outputEnter').onclick=async()=>{$('outputWelcome').hidden=true;await fullscreen();};
  $('outputFullscreen').onclick=fullscreen;$('outputDismiss').onclick=()=>{$('outputWelcome').hidden=true;};
  let hudTimer;
  function showHud(){
    $('outputView').classList.remove('hud-hidden');clearTimeout(hudTimer);
    hudTimer=setTimeout(()=>{if($('outputWelcome').hidden)$('outputView').classList.add('hud-hidden');},2300);
  }
  window.addEventListener('pointermove',showHud);showHud();
  window.addEventListener('keydown',e=>{
    if(e.key.toLowerCase()==='f'||e.key==='Enter'){e.preventDefault();$('outputWelcome').hidden=true;fullscreen();}
    if(e.code==='Space'){e.preventDefault();transport(()=>({playing:!playing,enabled:true}));}
    if(e.key.toLowerCase()==='b')transport(()=>({blackout:!blackout}),{score:false});
  });
  function outputFrame(){preview.draw($('outputCanvas'),blackout);requestAnimationFrame(outputFrame);}
  requestAnimationFrame(outputFrame);
}else{
  const outputPresence={program:0,aux:0};
  document.querySelectorAll('button[data-preview-output]').forEach(button=>{
    button.onclick=()=>selectPreview(button.dataset.previewOutput);
  });
  function toggleBlackout(){return transport(()=>({blackout:!blackout}),{score:false});}
  // Switching on can reload the patch at the monitor's size (shaders compile), so allow it time.
  async function setNative(enabled){
    const button=$('nativeToggleBtn');button.disabled=true;
    try{renderNative(await api('native_output',{enabled,monitor:$('nativeMonitor').value},AbortSignal.timeout(30000)));}
    catch(error){toast(error.message);}
    finally{button.disabled=!nativeState?.available;}
  }
  $('nativeToggleBtn').onclick=()=>{$('displayMenu').open=false;setNative($('nativeToggleBtn').getAttribute('aria-pressed')!=='true');};
  $('nativeMonitor').onchange=()=>{$('nativeMonitor').dataset.chosen=$('nativeMonitor').value;if(nativeState?.active)setNative(true);else renderNative(nativeState);};
  listen(m=>{if(m.type==='presence'&&['program','aux'].includes(m.kind))outputPresence[m.kind]=Date.now();});
  function openOutput(kind){
    const u=new URL(location.href);u.search='';u.searchParams.set('output',kind);u.searchParams.set('session',sessionId);
    const win=window.open(u.href,`gorgon-${sessionId}-${kind}`,'popup=yes,width=1280,height=800');
    if(!win)toast('The output window was blocked. Allow pop-ups for this site, then open the output again.');
    else {win.focus();toast(`Move the ${kind==='aux'?'auxiliary':'program'} window to your display and press F for fullscreen.`);}
  }
  $('playBtn').onclick=togglePlay;
  $('stopBtn').onclick=()=>scoreFailed||conflicted?transport({playing:false},{score:false}):transport({playing:false,beat:state.loop?state.pattern*16:editPattern*16});
  $('loopBtn').onclick=()=>{
    const beat=nowBeat();state.loop=!state.loop;
    if(state.loop)state.pattern=editPattern;
    saveLocal();seek(state.loop?editPattern*16:beat);updateControls();
  };
  $('tempo').onchange=()=>changeScore('tempo',()=>{state.bpm=clamp(Number($('tempo').value)||138,60,200);});
  $('key').onchange=()=>{changeScore('key transposition',()=>{state.key=Number($('key').value);});stopVoices();};
  $('audioBtn').onclick=async()=>{state.sound=!state.sound;if(state.sound)await ensureAudio();if(master)master.gain.setTargetAtTime(state.sound?.23:0,audioContext.currentTime,.02);if(!state.sound)stopVoices();updateControls();saveLocal();};
  $('fullscreenBtn').onclick=fullscreen;$('programBtn').onclick=()=>openOutput('program');$('expandPreview').onclick=()=>openOutput('program');$('auxBtn').onclick=()=>openOutput('aux');$('blackoutBtn').onclick=toggleBlackout;
  $('followBtn').onclick=()=>setFollow(!follow);
  document.querySelectorAll('button[data-editor-page]').forEach(button=>{
    button.onclick=()=>selectEditorPage(button.dataset.editorPage);
    button.onkeydown=e=>{
      if(!['ArrowLeft','ArrowRight','Home','End'].includes(e.key))return;
      e.preventDefault();
      const next=e.key==='Home'?'audio':e.key==='End'?'visual':editorPage==='audio'?'visual':'audio';
      selectEditorPage(next);$(next+'PageTab').focus();
    };
  });
  $('seekCellBtn').onclick=seekSelected;
  $('trackerRows').addEventListener('dblclick',e=>{
    const row=e.target.closest('[data-seek-row]');
    if(row){selectCell(Number(row.dataset.seekRow),selected.col);seekSelected();}
  });
  let dragging=false,dragAnchor=null,dragMoved=false;
  $('trackerRows').addEventListener('pointerdown',e=>{
    const cell=e.target.closest('[data-col]');if(!cell||e.button!==0)return;
    const row=Number(cell.dataset.row),col=Number(cell.dataset.col);
    selectCell(row,col,e.shiftKey);dragAnchor={...(selectionAnchor||selected)};dragging=true;dragMoved=false;
    if(e.pointerType==='mouse'){e.preventDefault();focusCell();}
  });
  $('trackerRows').addEventListener('pointerover',e=>{
    if(!dragging||!(e.buttons&1))return;
    const cell=e.target.closest('[data-col]');if(!cell)return;
    selectionAnchor=dragAnchor;selected={row:Number(cell.dataset.row),col:Number(cell.dataset.col)};
    if(selected.row!==dragAnchor.row||selected.col!==dragAnchor.col)dragMoved=true;
    if(tracks[selected.col].visual)curve=tracks[selected.col].id;
    updateSelection();updateInspector();focusCell();
  });
  window.addEventListener('pointerup',()=>{dragging=false;});
  window.addEventListener('blur',()=>{dragging=false;});
  $('trackerScroll').addEventListener('scroll',()=>{$('trackHeaders').scrollLeft=$('trackerScroll').scrollLeft;});
  // Native keyboard activation and assistive-technology clicks have no pointerdown.
  $('trackerRows').addEventListener('click',e=>{
    const cell=e.target.closest('[data-col]');if(!cell)return;
    if(!dragMoved&&e.detail===0)selectCell(Number(cell.dataset.row),Number(cell.dataset.col),e.shiftKey);
    dragMoved=false;
  });
  document.querySelector('.editor-stack').addEventListener('focusin',e=>{if(playing&&follow&&!switchingPage&&!e.target.closest('button[data-editor-page]'))setFollow(false);});
  ['z','roll','feedback'].forEach(id=>$(id+'Slider').addEventListener('input',()=>setKey(id,Number($(id+'Slider').value))));
  $('curveTabs').querySelectorAll('button').forEach(b=>b.onclick=()=>{curve=b.dataset.curve;selected.col=tracks.findIndex(t=>t.id===curve);selectionAnchor=null;renderRows();updateInspector();});
  $('easeButtons').querySelectorAll('button').forEach(b=>b.onclick=()=>changeScore('interpolation',()=>{state.patterns[editPattern].easing[curve]=b.dataset.ease;}));
  $('clearCellBtn').onclick=clearCell;
  $('undoBtn').onclick=()=>restoreHistory();$('redoBtn').onclick=()=>restoreHistory(true);
  function selectBlock(all=false){
    const columns=PAGE_COLUMNS[editorPage];
    selectionAnchor={row:0,col:all?columns[0]:selected.col};selected={row:63,col:all?columns.at(-1):selected.col};
    setFollow(false);updateSelection();updateInspector();focusCell();
  }
  $('selectLaneBtn').onclick=()=>selectBlock();$('selectAllBtn').onclick=()=>selectBlock(true);
  function copySelection(cut=false){
    clipboard=copyBlock(state.patterns[editPattern],selectionBox());
    if(cut)clearCell();
    toast(`${cut?'Cut':'Copied'} ${clipboard.rows.length} × ${clipboard.columns.length} cells.`);
    return JSON.stringify(clipboard);
  }
  function pasteSelection(block=clipboard){
    if(!block){toast('Copy a Gorgon block first.');return;}
    try{
      block=readBlock(JSON.stringify(block));
      let result;
      changeScore('paste cells',()=>{result=pasteBlock(state.patterns[editPattern],selected,block,{lastColumn:PAGE_COLUMNS[editorPage].at(-1)});});
      if(result)toast(`Pasted ${result.written} cells.${result.skipped?' '+result.skipped+' outside this page/pattern or in incompatible lanes were skipped.':''}`);
    }catch(error){toast(error.message);}
  }
  $('copyBtn').onclick=()=>copySelection();$('cutBtn').onclick=()=>copySelection(true);$('pasteBtn').onclick=()=>pasteSelection();
  $('trackerRows').addEventListener('copy',e=>{e.preventDefault();e.clipboardData.setData('text/plain',copySelection());});
  $('trackerRows').addEventListener('cut',e=>{e.preventDefault();e.clipboardData.setData('text/plain',copySelection(true));});
  $('trackerRows').addEventListener('paste',e=>{
    e.preventDefault();const text=e.clipboardData.getData('text/plain');
    try{pasteSelection(text?readBlock(text):clipboard);}catch(error){toast(error.message);}
  });
  function transposeSelection(amount){changeScore('transpose notes',()=>transposeBlock(state.patterns[editPattern],selectionBox(),amount));}
  document.querySelectorAll('[data-transpose]').forEach(b=>b.onclick=()=>transposeSelection(Number(b.dataset.transpose)));
  function velocity(){
    const value=Number($('velocityInput').value);
    if(!Number.isInteger(value)||value<0||value>127)throw new Error('Velocity must be 0–127.');
    return value;
  }
  function writeNote(note=$('noteInput').value){
    if(selected.col>=4)return;
    try{
      note=canonicalNote(note.toUpperCase().trim());entryVelocity=velocity();
      changeScore('write note',()=>putCell(state.patterns[editPattern],selected.row,selected.col,{note,vel:entryVelocity}));
    }catch(error){toast(error.message);}
  }
  $('noteInput').onchange=()=>writeNote();
  $('velocityInput').onchange=()=>{
    try{
      entryVelocity=velocity();const box=selectionBox();
      if(box.r0===box.r1&&box.c0===box.c1)writeNote();
    }catch(error){toast(error.message);}
  };
  $('insertNoteBtn').onclick=()=>writeNote();
  function writeScene(){
    if(selected.col!==SCENE)return;
    const name=$('sceneChoice').value,fade=Number($('sceneFade').value);
    if(!SCENE_NAME.test(name)){toast('Choose a scene declared in the patch.');return;}
    if(!Number.isInteger(fade)||fade<0||fade>64){toast('Fade is 0–64 rows.');return;}
    changeScore('write scene',()=>putCell(state.patterns[editPattern],selected.row,SCENE,{name,fade}));
  }
  $('writeSceneBtn').onclick=writeScene;
  // Changing the name or fade rewrites an existing change; an empty cell waits for Write scene.
  $('sceneChoice').onchange=$('sceneFade').onchange=()=>{if(getCell(state.patterns[editPattern],selected.row,SCENE))writeScene();};
  $('applyVelocityBtn').onclick=()=>{
    try{const v=velocity();entryVelocity=v;changeScore('block velocity',()=>setBlockVelocity(state.patterns[editPattern],selectionBox(),v));}catch(error){toast(error.message);}
  };
  $('pianoBtn').onclick=()=>{
    pianoEntry=!pianoEntry;$('pianoBtn').setAttribute('aria-pressed',pianoEntry);$('pianoBtn').textContent=pianoEntry?'Piano keys on':'Piano keys off';
    $('entryHint').textContent=pianoEntry?'Z–M: this octave · Q–U: next octave · B enters G while the note grid is focused':'Arrows move · Shift+arrows select · Enter toggles';
    focusCell(true);
  };
  $('octaveInput').onchange=()=>{$('octaveInput').value=clamp(Number($('octaveInput').value)||0,0,8);};
  $('auditionBtn').onclick=async()=>{
    const hit=state.patterns[editPattern].audio[selected.col]?.[selected.row];if(!hit)return;
    state.sound=true;await ensureAudio();updateControls();saveLocal();
    if(audioContext)soundHit(selected.col,hit,audioContext.currentTime+.01);
  };
  $('writeKeyBtn').onclick=()=>setKey(tracks[selected.col].id,Number($('cellValue').value));
  $('cellValue').onchange=()=>setKey(tracks[selected.col].id,Number($('cellValue').value));
  $('curveCanvas').addEventListener('pointerdown',e=>{
    const rect=e.currentTarget.getBoundingClientRect(),d=e.currentTarget.width/rect.width,t=tracks.find(t=>t.id===curve),x=(e.clientX-rect.left)*d,y=(e.clientY-rect.top)*d;
    const row=clamp(Math.round((x-32)/(e.currentTarget.width-48)*63),0,63),value=t.min+(1-(y-18)/(e.currentTarget.height-41))*(t.max-t.min);
    selected.row=row;setFollow(false);setKey(curve,Math.round(value*100)/100);
  });
  $('patternName').onchange=()=>{
    const name=$('patternName').value.trim();if(!name){toast('Give the pattern a name.');updateInspector();return;}
    changeScore('pattern name',()=>{state.patterns[editPattern].name=name;});
  };
  $('duplicatePatternBtn').onclick=()=>{
    const target=Number($('duplicateTarget').value);if(target===editPattern){toast('Choose another destination.');return;}
    changeScore('copy pattern',()=>{state.patterns[target]=structuredClone(state.patterns[editPattern]);});
    $('patternMenu').open=false;toast(`Copied to pattern ${target+1}. Undo restores its previous contents.`);
  };
  $('clearPatternBtn').onclick=()=>{
    changeScore('clear pattern',()=>clearBlock(state.patterns[editPattern],{r0:0,r1:63,c0:0,c1:tracks.length-1}));
    $('patternMenu').open=false;toast('Pattern cleared. Undo restores it.');
  };
  ['program','aux'].forEach(k=>$(k+'Mode').onchange=()=>{state.modes[k]=$(k+'Mode').value;saveLocal();});
  $('sceneBtn').onclick=async()=>{try{const data=await api('catalog');$('patchSelect').replaceChildren(...data.patches.map(p=>new Option(p.path,p.path)));$('patchDialog').showModal();}catch(error){toast(error.message);}};
  $('choosePatchBtn').onclick=()=>{$('fileMenu').open=false;$('sceneBtn').click();};
  $('helpBtn').onclick=()=>$('helpDialog').showModal();$('closeHelp').onclick=()=>$('helpDialog').close();
  $('saveBtn').onclick=async()=>{
    if(!await flushScore()){downloadScore();toast('Downloaded local recovery copy. Resolve the pending edit to save to Studio.');return;}
    try{
      const saved=await api('tracker_save',{revision});
      downloadScore(saved.document,'gorgon-session.json');
      $('savedState').textContent='Saved to workspace';toast('Score saved to .hydrapy/tracker.json and downloaded.');
    }catch(error){toast(error.message);}
  };
  async function importScore(document){
    if(!document)throw new Error('No saved score in this workspace.');
    if(!await flushScore())throw new Error('Resolve the pending edit before importing. Download your recovery copy from File.');
    const result=await api('tracker_score',{document,revision});
    history.change(state,'import score',()=>{
      for(const field of ['patterns','bpm','key','mutes','bindings'])state[field]=structuredClone(result.document[field]);
    },{pattern:editPattern,selected});
    state=result.document;
    revision=result.revision;acceptedDocument=structuredClone(result.document);editPattern=state.pattern;selectionAnchor=null;
    redrawScore();stopVoices();toast('Score loaded.');
  }
  $('importBtn').onclick=()=>{$('fileMenu').open=false;$('importFile').click();};
  $('importFile').onchange=async event=>{
    try{const file=event.target.files[0];if(!file)return;if(file.size>2000000)throw new Error('Score is too large.');await importScore(JSON.parse(await file.text()));}
    catch(error){toast(error.message);}event.target.value='';
  };
  $('restoreBtn').onclick=async()=>{try{$('fileMenu').open=false;await importScore((await api('tracker_saved')).document);}catch(error){toast(error.message);}};
  $('newScoreBtn').onclick=()=>{
    if(!confirm('Start an empty score? You can undo this edit.'))return;
    changeScore('new empty score',()=>{state.patterns.forEach(p=>clearBlock(p,{r0:0,r1:63,c0:0,c1:tracks.length-1}));});
    $('fileMenu').open=false;
  };
  $('downloadScoreBtn').onclick=()=>{downloadScore();$('fileMenu').open=false;};
  $('recoveryBtn').onclick=()=>downloadScore();$('retryScoreBtn').onclick=retryScore;
  $('reloadScoreBtn').onclick=async()=>{
    if(!confirm('Replace local edits with Studio’s score? Download a recovery copy first if you need them.'))return;
    try{
      await scoreRequest;const result=await api('tracker');
      clearTimeout(scoreTimer);pendingScore=null;scoreFailed=conflicted=false;
      state=result.document;acceptedDocument=structuredClone(state);revision=result.revision;editPattern=state.pattern;selectionAnchor=null;
      history.clear();redrawScore();$('editConnection').hidden=true;$('savedState').textContent='Server score loaded';
    }catch(error){toast(error.message);}
  };
  let audioRequest=false;
  async function configureJack(enabled){
    if(audioRequest)return;
    audioRequest=true;$('jackEnableBtn').disabled=true;
    try{renderJackAudio(await api('jack_audio',{enabled,optical_gain:Number($('opticalGain').value),breaks_gain:Number($('breaksGain').value),lead_blocks:Number($('audioLead').value)}));}
    catch(error){toast(error.message);$('jackStatus').textContent=error.message;}
    finally{audioRequest=false;$('jackEnableBtn').disabled=false;}
  }
  $('jackEnableBtn').onclick=()=>configureJack(!jackState?.active);
  ['opticalGain','breaksGain','audioLead'].forEach(id=>$(id).onchange=()=>configureJack(!!jackState?.active));
  $('automationBtn').onclick=()=>transport(()=>({enabled:!remote?.enabled}));
  ['z','roll','feedback'].forEach(lane=>$('bind_'+lane).onchange=()=>changeScore('parameter binding',()=>{state.bindings[lane]=$('bind_'+lane).value;}));
  $('loadPatch').onclick=async()=>{try{if(!await flushScore())return;await api('load',{path:$('patchSelect').value});$('patchDialog').close();}catch(error){toast(error.message);}};
  $('closePatch').onclick=()=>$('patchDialog').close();
  $('studioLink').href='/#'+location.hash.slice(1);
  $('studioLink').onclick=async event=>{event.preventDefault();if(await flushScore())location.href=$('studioLink').href;};
  $('resetBtn').onclick=async()=>{
    if(!confirm('Replace the current score with the starting study?'))return;
    try{await transport({playing:false,enabled:false,beat:0});await importScore(defaults());}catch(error){toast(error.message);}
  };
  $('libraryBtn').onclick=()=>{
    if(innerWidth<=1100)document.body.classList.toggle('library-open');
    else document.body.classList.toggle('library-collapsed');
    $('libraryBtn').setAttribute('aria-pressed',innerWidth<=1100?document.body.classList.contains('library-open'):!document.body.classList.contains('library-collapsed'));
  };
  const updateLibraryButton=()=>$('libraryBtn').setAttribute('aria-pressed',innerWidth<=1100?document.body.classList.contains('library-open'):!document.body.classList.contains('library-collapsed'));
  updateLibraryButton();window.addEventListener('resize',updateLibraryButton);
  $('textSize').onchange=()=>document.body.classList.toggle('large-text',$('textSize').value==='large');
  $('curveVisibilityBtn').onclick=()=>{
    const collapsed=document.body.classList.toggle('curve-collapsed');$('curveVisibilityBtn').setAttribute('aria-pressed',!collapsed);
  };
  document.addEventListener('click',e=>{
    document.querySelectorAll('details[open]').forEach(d=>{if(!d.contains(e.target))d.open=false;});
  });
  window.addEventListener('beforeunload',e=>{if(pendingScore||scoreRequest||scoreFailed||conflicted){e.preventDefault();e.returnValue='';}});
  window.addEventListener('keydown',e=>{
    if($('helpDialog').open||$('patchDialog').open)return;
    const target=document.activeElement,form=['INPUT','SELECT','TEXTAREA'].includes(target.tagName),command=e.ctrlKey||e.metaKey;
    if(command&&e.key.toLowerCase()==='s'){e.preventDefault();$('saveBtn').click();return;}
    if(form)return;
    const inGrid=!!target.closest('#trackerRows');
    if(inGrid&&e.key==='Tab'&&!command&&!e.altKey&&!e.shiftKey){
      e.preventDefault();selectEditorPage(editorPage==='audio'?'visual':'audio',{focus:true});return;
    }
    if(command){
      const key=e.key.toLowerCase();
      if(key==='z'){e.preventDefault();restoreHistory(e.shiftKey);}
      else if(key==='y'){e.preventDefault();restoreHistory(true);}
      else if(inGrid&&key==='a'){e.preventDefault();selectBlock(true);}
      // Native copy/cut/paste events carry a validated block across controllers.
      return;
    }
    if(!e.altKey&&(e.key==='['||e.key===']')){
      e.preventDefault();
      const current=Number((previewOutput==='program'?remote?.output||'o0':previewOutput)[1]);
      selectPreview(`o${mod(current+(e.key===']'?1:-1),4)}`);return;
    }
    if(e.key==='Escape'){
      selectionAnchor=null;updateSelection();document.querySelectorAll('details[open]').forEach(d=>{d.open=false;});return;
    }
    if(inGrid&&pianoEntry&&selected.col<4&&!e.altKey){
      const note=pianoNote(e.code,Number($('octaveInput').value));
      if(note){
        e.preventDefault();if(e.repeat)return;
        const col=selected.col;writeNote(note);
        if(state.sound&&audioContext?.state==='running')soundHit(col,{note,vel:entryVelocity},audioContext.currentTime+.005);
        selectCell(Math.min(63,selected.row+Number($('entryStep').value)),col);focusCell(true);return;
      }
    }
    if(e.code==='Space'){e.preventDefault();togglePlay();return;}
    if(e.shiftKey&&e.key.toLowerCase()==='f'&&nativeState?.available){e.preventDefault();setNative(!nativeState.active);return;}
    if(e.key.toLowerCase()==='f'){e.preventDefault();fullscreen();return;}
    if(e.key.toLowerCase()==='b'){e.preventDefault();toggleBlackout();return;}
    if(!inGrid)return;
    if(e.altKey&&(e.key==='ArrowUp'||e.key==='ArrowDown')){e.preventDefault();transposeSelection((e.key==='ArrowUp'?1:-1)*(e.shiftKey?12:1));return;}
    if(['ArrowUp','ArrowDown','ArrowLeft','ArrowRight','Home','End','PageUp','PageDown'].includes(e.key)){
      e.preventDefault();let r=selected.row,c=selected.col;
      if(e.key==='Home')r=0;else if(e.key==='End')r=63;else if(e.key==='PageUp')r-=16;else if(e.key==='PageDown')r+=16;
      else {r+=e.key==='ArrowDown'?1:e.key==='ArrowUp'?-1:0;c+=e.key==='ArrowRight'?1:e.key==='ArrowLeft'?-1:0;}
      const columns=PAGE_COLUMNS[editorPage];
      selectCell(clamp(r,0,63),clamp(c,columns[0],columns.at(-1)),e.shiftKey);focusCell(true);
    }
    if(e.key==='Delete'||e.key==='Backspace'){e.preventDefault();clearCell();}
    if(e.key==='Enter'){e.preventDefault();if(e.altKey)seekSelected();else toggleCell();}
  });
  renderPatterns();renderHeaders();renderRows();renderInstruments();updateControls();updateInspector();drawVisual($('sceneThumb'),0,'full');
  setInterval(scheduler,25);
  setInterval(()=>{for(const k of ['program','aux']){const fullscreen=k==='program'&&nativeState?.active;const connected=fullscreen||Date.now()-outputPresence[k]<2600;$(k+'Btn').classList.toggle('connected',connected);$(k+'Status').textContent=fullscreen?(nativeState.where==='window'?'In a window':`Fullscreen on ${nativeState.where}`):connected?'Window connected':'Open output';$(k==='aux'?'routeAux':'routeProgram').classList.toggle('mint',connected);}},800);
  function frame(){
    const beat=nowBeat(),step=Math.floor(beat*4),pattern=Math.floor(beat/16),row=step%64;
    if(playing&&follow&&editPattern!==pattern){editPattern=pattern;selectionAnchor=null;renderPatterns();renderRows();updateInspector();}
    if(step!==lastStep){
      $('position').innerHTML=`${pad(Math.floor(beat/4)+1,3)}<span>:</span>${pad(Math.floor(beat%4)+1)}<span>:</span>${pad(step%4)}`;
      $('trackerRows').querySelectorAll('.playing').forEach(e=>e.classList.remove('playing'));
      if(pattern===editPattern){const el=$('trackerRows').querySelector(`.tracker-row[data-row="${row}"]`);el?.classList.add('playing');if(playing&&follow&&el){const area=$('trackerScroll');if(el.offsetTop<area.scrollTop||el.offsetTop+el.offsetHeight>area.scrollTop+area.clientHeight)area.scrollTop=Math.max(0,el.offsetTop-area.clientHeight*.3);}}
      lastStep=step;lastPattern=pattern;
    }
    $('patterns').querySelectorAll('.pattern-card').forEach((e,i)=>{e.classList.toggle('current',i===pattern);if(i===pattern)e.querySelector('.pattern-playhead').style.left=`${mod(beat,16)/16*100}%`;});
    drawVisual($('previewCanvas'),beat,state.modes.program,state,blackout);drawVisual($('sceneThumb'),beat);drawCurve(beat);
    let level=0;if(analyser&&state.sound){analyser.getByteTimeDomainData(levelData);let peak=0;for(const v of levelData)peak=Math.max(peak,Math.abs(v-128)/128);level=clamp((20*Math.log10(Math.max(.0001,peak))+54)/5.4,0,10);}document.querySelectorAll('.meter i').forEach((e,i)=>e.classList.toggle('on',i<level));
    requestAnimationFrame(frame);
  }requestAnimationFrame(frame);
  window.addEventListener('pagehide',()=>{playing=false;stopVoices();});
}

pollRemote();
window.addEventListener('pageshow', event=>{if(event.persisted)location.reload();});
