import {api, PreviewStream} from './backend.js';
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
];
const defaults = () => ({version:1,format:'hydrapy-tracker-1',bindings:{z:'Dolly Z',roll:'Roll',feedback:'Feedback'},bpm:138,key:2,sound:false,loop:false,pattern:0,scene:0,modes:{program:'o0',aux:'o1'},mutes:[false,false,false,false],patterns:['DRIFT','FRACTURE','ASCENT','SUSPEND'].map((name,p)=>{
  const audio = tracks.slice(0,4).map(()=>Array(64).fill(null));
  for(let r=0;r<64;r++){
    if(r%4===0 || (p===1 && r%16===14)) audio[0][r]={note:r%8===4?'D-2':'C-2',vel:r%8===4?100:115};
    if(r%4===2) audio[0][r]={note:'F#2',vel:54+(r%8)*3};
    if([6,14,22,31,38,46,54,62].includes(r) && !(p===3&&r%16<10)) audio[1][r]={note:['D-4','A-4','F-4','C-5'][Math.floor(r/8)%4],vel:64+(p*9)};
    if(r%16===0 || (p===2 && r%16===10)) audio[2][r]={note:r<32?'D-2':'A-1',vel:91};
    if(r%32===24) audio[3][r]={note:'AIR',vel:45};
  }
  return {name,audio,easing:{z:'smooth',roll:'smooth',feedback:'smooth'},keys:{
    z:[{r:0,v:[.35,.55,.2,.8][p]},{r:16,v:[.55,.25,.4,.7][p]},{r:32,v:[.45,.8,.7,.45][p]},{r:48,v:[.8,.5,.9,.25][p]},{r:63,v:[.35,.55,.2,.8][p]}],
    roll:[{r:0,v:0},{r:16,v:[-12,22,-6,14][p]},{r:32,v:[9,-18,19,-9][p]},{r:48,v:[-8,16,32,5][p]},{r:63,v:0}],
    feedback:[{r:0,v:.25},{r:24,v:.45+p*.08},{r:40,v:.72-p*.08},{r:56,v:.3},{r:63,v:.25}],
  }};
})});

let state=defaults();
let playing=false,anchorBeat=state.pattern*16,anchorTime=performance.now(),blackout=false,follow=true;
let selected={row:0,col:4},curve='z',lastStep=-1,lastPattern=-1;
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
function toast(message){$('toast').textContent=message;$('toast').classList.add('visible');clearTimeout(toastTimer);toastTimer=setTimeout(()=>$('toast').classList.remove('visible'),4000);}
let remote = null, revision = 0, pendingScore = null, scoreRequest = null, scoreTimer;
let transportQueue = Promise.resolve(), conflicted = false;
let transportPending = 0, transportEpoch = 0;
const preview = new PreviewStream(outputType === 'aux' ? 'aux' : 'program');
window.gorgonPreview = preview.stats;
const escapeHTML = value => String(value).replace(/[&<>"']/g, c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
function saveLocal(){
  if (!remote || conflicted) return;
  pendingScore = structuredClone(state);
  $('savedState').textContent = 'Unsaved score';
  clearTimeout(scoreTimer); scoreTimer = setTimeout(flushScore, 120);
}
async function flushScore(){
  clearTimeout(scoreTimer);
  if (scoreRequest) { await scoreRequest; return flushScore(); }
  if (!pendingScore) return;
  const document = pendingScore; pendingScore = null;
  scoreRequest = api('tracker_score', {document, revision}).then(result => {
    revision = result.revision;
    // Studio may have switched its displayed output since this edit's base.
    // Keep that choice unless the user has changed the program route again.
    const program = result.document.modes.program;
    if (state.modes.program === document.modes.program) state.modes.program = program;
    if (pendingScore && pendingScore.modes.program === document.modes.program) pendingScore.modes.program = program;
    if (program !== document.modes.program) updateControls();
    $('savedState').textContent = 'Score applied · not saved';
  }).catch(error => {
    conflicted = true; pendingScore = null;
    $('savedState').textContent = 'Edit rejected · reload'; toast(error.message);
  });
  await scoreRequest; scoreRequest = null;
  if (pendingScore) return flushScore();
}
function adoptTransport(result, elapsed=0){
  const old = nowBeat();
  const next = normalizeBeat(result.beat + (result.playing ? elapsed * state.bpm / 60 : 0));
  if (playing !== result.playing || Math.abs(next-old) > .6) stopVoices();
  anchorBeat=next; anchorTime=performance.now(); playing=result.playing;
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
function transport(changes){
  transportPending++;transportEpoch++;
  transportQueue=transportQueue.then(async()=>{
    await flushScore();
    if(conflicted) throw new Error('Reload to recover the last accepted score.');
    const result=await api('tracker_transport',typeof changes==='function'?changes():changes);
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
    select.classList.toggle('missing',!!name&&!controls.some(p=>p.name===name));
  }
}
async function pollRemote(){
  const started=performance.now(), epoch=transportEpoch;
  try{
    const result=await api('tracker');
    if(!result.document)throw new Error('Studio is starting.');
    if(result.server_now-result.sampled>3)throw new Error('Studio renderer is not responding.');
    if(!remote||(!pendingScore&&!scoreRequest&&result.revision>revision)){
      if(!remote)$('savedState').textContent='Live score · save to keep';
      state=result.document;revision=result.revision;
      if(!outputType){renderPatterns();renderHeaders();renderRows();renderInstruments();updateControls();updateInspector();}
    }
    if(!transportPending&&epoch===transportEpoch){
      remote=result;
      adoptTransport(result, Math.max(0,result.server_now-result.sampled)+(performance.now()-started)/2000);
    }else if(!remote)remote=result;
    preview.select(outputType==='aux'?result.document.modes.aux:result.output,
      outputType==='aux'?result.aux_generation:result.generation);
    if(!outputType){
      renderBindings(result.controls);
      $('sceneName').textContent=result.path?.split('/').pop()||'Choose a patch';
      $('sceneMeta').textContent=result.output.toUpperCase();
    }
    $('connectionStatus').textContent=result.playing?'Transport running':'Transport paused';
    $('backendStatus').textContent=result.error||(result.path?`${result.backend} · ${result.fps.toFixed(1)} render fps · ${result.path}`:'Load a patch to begin');
    $('resolution').textContent=result.size.join(' × ');
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
function noteFreq(note){const values={'C-2':36,'D-2':38,'F#2':42,'D-4':62,'A-4':69,'F-4':65,'C-5':72,'A-1':33};const midi=(values[note]||62)+(state.key-2);return 440*2**((midi-69)/12);}
function soundHit(track,hit,t){
  if(!audioContext||!master||!hit)return;const vel=hit.vel/127;
  if(track===0&&hit.note==='C-2'){
    const o=audioContext.createOscillator(),g=audioContext.createGain();o.frequency.setValueAtTime(145,t);o.frequency.exponentialRampToValueAtTime(44,t+.13);g.gain.setValueAtTime(.95*vel,t);g.gain.exponentialRampToValueAtTime(.001,t+.29);connectVoice(o,g,t,.3);
  }else if(track===0){
    const snare=hit.note==='D-2',n=audioContext.createBufferSource(),f=audioContext.createBiquadFilter(),g=audioContext.createGain();n.buffer=noiseBuffer;f.type='highpass';f.frequency.value=snare?1000:7200;n.connect(f);f.connect(g);g.connect(master);g.gain.setValueAtTime(vel*(snare?.44:.18),t);g.gain.exponentialRampToValueAtTime(.001,t+(snare?.15:.04));voices.add(n);n.onended=()=>{voices.delete(n);n.disconnect();f.disconnect();g.disconnect();};n.start(t,mod(t*.17,1));n.stop(t+(snare?.17:.06));
  }else if(track===1){
    const o=audioContext.createOscillator(),m=audioContext.createOscillator(),mg=audioContext.createGain(),g=audioContext.createGain();const freq=noteFreq(hit.note);o.frequency.value=freq;m.frequency.value=freq*2.73;mg.gain.setValueAtTime(freq*2,t);mg.gain.exponentialRampToValueAtTime(1,t+.3);m.connect(mg);mg.connect(o.frequency);g.gain.setValueAtTime(.17*vel,t);g.gain.exponentialRampToValueAtTime(.001,t+.5);connectVoice(o,g,t,.52);voices.add(m);m.onended=()=>{voices.delete(m);m.disconnect();mg.disconnect();};m.start(t);m.stop(t+.52);
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
  $('patterns').innerHTML=state.patterns.map((p,i)=>`<button class="pattern-card ${i===state.pattern?'selected':''}" data-pattern="${i}" aria-label="Select pattern ${i+1}: ${escapeHTML(p.name)}" aria-pressed="${i===state.pattern}"><div class="pattern-top"><b>${pad(i+1)}</b><span>${escapeHTML(p.name)}</span><span class="bars">${pad(i*4+1)}—${pad(i*4+4)}</span></div>${waveform(i*3)}<i class="pattern-playhead"></i></button>`).join('');
  $('patterns').querySelectorAll('button').forEach(b=>b.addEventListener('click',()=>selectPattern(Number(b.dataset.pattern),true)));
}
function renderHeaders(){
  $('trackHeaders').innerHTML='<div class="track-header row-index">ROW</div>'+tracks.map((t,i)=>`<div class="track-header ${t.visual?'visual':''}"><div><strong>${t.name}</strong>${t.visual?'':`<button class="mute" data-mute="${i}" aria-label="Mute ${t.name}" aria-pressed="${state.mutes[i]}">M</button>`}</div><span>${t.sub}</span></div>`).join('');
  $('trackHeaders').querySelectorAll('[data-mute]').forEach(b=>b.addEventListener('click',()=>{state.mutes[Number(b.dataset.mute)]=!state.mutes[Number(b.dataset.mute)];stopVoices();renderHeaders();renderRows();saveLocal();}));
}
function formatValue(id,v){return id==='roll'?`${v>=0?'+':''}${Math.round(v)}°`:v.toFixed(2);}
function renderRows(){
  const p=state.patterns[state.pattern];let html='';
  for(let r=0;r<64;r++){
    html+=`<div class="tracker-row ${r%4===0?'beat':''} ${r%16===0?'bar':''}" role="row" data-row="${r}"><span class="row-number" role="rowheader">${hex(r)}</span>`;
    tracks.forEach((t,c)=>{
      const key=t.visual?p.keys[t.id].find(k=>k.r===r):p.audio[c][r];const content=key?(t.visual?formatValue(t.id,key.v):`${key.note}<span class="velocity">${hex(key.vel)}</span>`):'···';
      html+=`<button role="gridcell" tabindex="${selected.row===r&&selected.col===c?'0':'-1'}" class="cell ${t.visual?'visual':''} ${!key?'empty':''} ${selected.row===r&&selected.col===c?'selected':''} ${c<4&&state.mutes[c]?'muted':''}" data-row="${r}" data-col="${c}" aria-label="${t.name}, row ${hex(r)}, ${key?(t.visual?formatValue(t.id,key.v):key.note):'empty'}" aria-selected="${selected.row===r&&selected.col===c}">${content}</button>`;
    });html+='</div>';
  }
  $('trackerRows').innerHTML=html;lastStep=-1;
}
function renderInstruments(){
  $('instruments').innerHTML=tracks.slice(0,4).map((t,i)=>`<button class="instrument ${selected.col===i?'active':''}" data-instrument="${i}" title="Audition ${t.label}"><span class="instrument-name"><strong>${t.label}</strong><small>${t.tag}</small></span>${waveform(i*7,38)}</button>`).join('');
  $('instruments').querySelectorAll('button').forEach(b=>b.addEventListener('click',async()=>{selected.col=Number(b.dataset.instrument);await ensureAudio();if(master&&state.sound)soundHit(selected.col,{note:tracks[selected.col].note,vel:90},audioContext.currentTime+.005);renderInstruments();renderRows();updateInspector();}));
}
function selectPattern(index,doSeek=false){state.pattern=index;renderPatterns();renderRows();updateInspector();if(doSeek)seek(index*16);saveLocal();}
function selectCell(row,col,seekVisual=false){
  selected={row,col};if(tracks[col].visual){curve=tracks[col].id;if(seekVisual)seek(state.pattern*16+row/4);}
  $('trackerRows').querySelectorAll('.selected').forEach(e=>{e.classList.remove('selected');e.setAttribute('aria-selected','false');e.tabIndex=-1;});
  const cell=$('trackerRows').querySelector(`[data-row="${row}"][data-col="${col}"]`);cell?.classList.add('selected');cell?.setAttribute('aria-selected','true');if(cell)cell.tabIndex=0;
  updateInspector();
}
function toggleCell(){const p=state.patterns[state.pattern],t=tracks[selected.col];if(t.visual){const keys=p.keys[t.id];const found=keys.findIndex(k=>k.r===selected.row);if(found>=0)keys.splice(found,1);else keys.push({r:selected.row,v:valueAt(p,t.id,selected.row)});keys.sort((a,b)=>a.r-b.r);}else{p.audio[selected.col][selected.row]=p.audio[selected.col][selected.row]?null:{note:t.note,vel:90};}renderRows();updateInspector();saveLocal();}
function clearCell(){const t=tracks[selected.col],p=state.patterns[state.pattern];if(t.visual)p.keys[t.id]=p.keys[t.id].filter(k=>k.r!==selected.row);else p.audio[selected.col][selected.row]=null;renderRows();updateInspector();saveLocal();}
function setKey(id,value){
  const t=tracks.find(t=>t.id===id),p=state.patterns[state.pattern],v=clamp(value,t.min,t.max);let key=p.keys[id].find(k=>k.r===selected.row);
  if(key)key.v=v;else p.keys[id].push({r:selected.row,v});p.keys[id].sort((a,b)=>a.r-b.r);curve=id;selected.col=tracks.findIndex(t=>t.id===id);renderRows();updateInspector();saveLocal();
}
function updateInspector(){
  const p=state.patterns[state.pattern];$('patternTitle').textContent=`${pad(state.pattern+1)} / ${escapeHTML(p.name)}`;
  $('keyLocation').textContent=`KEY / ${hex(selected.row)}`;
  ['z','roll','feedback'].forEach(id=>{const v=valueAt(p,id,selected.row);$(id+'Slider').value=v;$(id+'Value').textContent=formatValue(id,v);});
  $('selectedTrack').textContent=tracks[selected.col].name;$('selectedRow').textContent=`ROW ${hex(selected.row)}`;
  const hit=selected.col<4?p.audio[selected.col][selected.row]:null;
  $('selectedHint').textContent=selected.col<4?(hit?`${hit.note} · velocity ${hit.vel}. Enter toggles this event.`:'Empty event. Enter inserts a hit.'):'Move a slider to write a keyframe.';
  $('curveTabs').querySelectorAll('button').forEach(b=>b.classList.toggle('active',b.dataset.curve===curve));
  $('easeButtons').querySelectorAll('button').forEach(b=>b.classList.toggle('active',b.dataset.ease===p.easing[curve]));
  $('curveValue').textContent=formatValue(curve,valueAt(p,curve,selected.row));$('curveUnit').textContent=curve==='roll'?'DEGREES':curve==='z'?'DEPTH':'AMOUNT';
}
function drawCurve(beat){
  const c=$('curveCanvas'),[w,h]=sizeCanvas(c),ctx=c.getContext('2d'),p=state.patterns[state.pattern],t=tracks.find(t=>t.id===curve),left=32,right=w-16,top=18,bottom=h-23;
  ctx.clearRect(0,0,w,h);ctx.fillStyle='#121b16';ctx.fillRect(0,0,w,h);ctx.font='9px ui-monospace, monospace';ctx.lineWidth=.6;
  for(let i=0;i<=4;i++){const y=top+(bottom-top)*i/4;ctx.strokeStyle='#2a3b2e';ctx.beginPath();ctx.moveTo(left,y);ctx.lineTo(right,y);ctx.stroke();if(i===0||i===4){ctx.fillStyle='#546b59';ctx.fillText((t.max-(t.max-t.min)*i/4).toFixed(t.id==='roll'?0:1),3,y+3);}}
  for(let i=0;i<5;i++){const x=left+(right-left)*i/4;ctx.strokeStyle='#344431';ctx.beginPath();ctx.moveTo(x,top);ctx.lineTo(x,bottom);ctx.stroke();ctx.fillStyle='#637c66';ctx.fillText(`${pad(state.pattern*4+i+1)}`,x-5,h-7);}
  const point=(r,v)=>[left+r/63*(right-left),bottom-(v-t.min)/(t.max-t.min)*(bottom-top)];
  ctx.beginPath();for(let r=0;r<=63;r+=.25){const xy=point(r,valueAt(p,curve,r));if(r===0)ctx.moveTo(...xy);else ctx.lineTo(...xy);}ctx.strokeStyle='#d6b27e';ctx.lineWidth=1.5;ctx.stroke();ctx.lineTo(right,bottom);ctx.lineTo(left,bottom);ctx.closePath();ctx.fillStyle='#d0ab6a0b';ctx.fill();
  p.keys[curve].forEach(k=>{const [x,y]=point(k.r,k.v);ctx.fillStyle=k.r===selected.row?'#f4d79c':'#a58959';ctx.fillRect(x-2.5,y-2.5,5,5);});
  if(Math.floor(beat/16)===state.pattern){const x=left+mod(beat,16)/16*(right-left);ctx.beginPath();ctx.moveTo(x,top-5);ctx.lineTo(x,bottom+4);ctx.strokeStyle='#a5d3a7';ctx.lineWidth=.8;ctx.stroke();}
}
function updateControls(){
  if(master)master.gain.setTargetAtTime(state.sound?.23:0,audioContext.currentTime,.02);
  $('tempo').value=state.bpm;$('key').value=state.key;$('audioBtn').setAttribute('aria-pressed',state.sound);$('audioLabel').textContent=state.sound?'Sound on':'Sound off';$('loopBtn').setAttribute('aria-pressed',state.loop);$('loopBtn').classList.toggle('active',state.loop);
  $('programMode').value=state.modes.program;$('auxMode').value=state.modes.aux;$('sceneName').textContent=remote?.path?.split('/').pop()||'Choose a patch';$('sceneMeta').textContent=remote?.output?.toUpperCase()||'HYDRAPY';
  const sec=64*60/state.bpm;$('duration').textContent=`00:${sec.toFixed(1)}`;
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
    if(e.key.toLowerCase()==='b')transport(()=>({blackout:!blackout}));
  });
  function outputFrame(){preview.draw($('outputCanvas'),blackout);requestAnimationFrame(outputFrame);}
  requestAnimationFrame(outputFrame);
}else{
  const outputPresence={program:0,aux:0};
  function toggleBlackout(){return transport(()=>({blackout:!blackout}));}
  listen(m=>{if(m.type==='presence'&&['program','aux'].includes(m.kind))outputPresence[m.kind]=Date.now();});
  function openOutput(kind){
    const u=new URL(location.href);u.search='';u.searchParams.set('output',kind);u.searchParams.set('session',sessionId);
    const win=window.open(u.href,`gorgon-${sessionId}-${kind}`,'popup=yes,width=1280,height=800');
    if(!win)toast('The output window was blocked. Allow pop-ups for this site, then open the output again.');
    else {win.focus();toast(`Move the ${kind==='aux'?'auxiliary':'program'} window to your display and press F for fullscreen.`);}
  }
  $('playBtn').onclick=togglePlay;$('stopBtn').onclick=()=>transport({playing:false,beat:state.pattern*16});
  $('loopBtn').onclick=()=>{const b=nowBeat();state.loop=!state.loop;seek(b);updateControls();saveLocal();};
  $('tempo').addEventListener('change',()=>{state.bpm=clamp(Number($('tempo').value)||138,60,200);updateControls();saveLocal();});
  $('key').onchange=()=>{state.key=Number($('key').value);stopVoices();saveLocal();};
  $('audioBtn').onclick=async()=>{state.sound=!state.sound;if(state.sound)await ensureAudio();if(master)master.gain.setTargetAtTime(state.sound?.23:0,audioContext.currentTime,.02);if(!state.sound)stopVoices();updateControls();saveLocal();};
  $('fullscreenBtn').onclick=fullscreen;$('programBtn').onclick=()=>openOutput('program');$('expandPreview').onclick=()=>openOutput('program');$('auxBtn').onclick=()=>openOutput('aux');$('blackoutBtn').onclick=toggleBlackout;
  $('followBtn').onclick=()=>{follow=!follow;$('followBtn').classList.toggle('active',follow);$('followBtn').setAttribute('aria-pressed',follow);};
  $('trackerRows').addEventListener('click',e=>{const b=e.target.closest('[data-col]');if(!b)return;const col=Number(b.dataset.col);selectCell(Number(b.dataset.row),col,col>=4);if(col<4)toggleCell();});
  ['z','roll','feedback'].forEach(id=>$(id+'Slider').addEventListener('input',()=>setKey(id,Number($(id+'Slider').value))));
  $('curveTabs').querySelectorAll('button').forEach(b=>b.onclick=()=>{curve=b.dataset.curve;selected.col=tracks.findIndex(t=>t.id===curve);renderRows();updateInspector();});
  $('easeButtons').querySelectorAll('button').forEach(b=>b.onclick=()=>{state.patterns[state.pattern].easing[curve]=b.dataset.ease;updateInspector();saveLocal();});
  $('clearCellBtn').onclick=clearCell;
  $('curveCanvas').addEventListener('pointerdown',e=>{
    const rect=e.currentTarget.getBoundingClientRect(),d=e.currentTarget.width/rect.width,t=tracks.find(t=>t.id===curve),x=(e.clientX-rect.left)*d,y=(e.clientY-rect.top)*d;
    const row=clamp(Math.round((x-32)/(e.currentTarget.width-48)*63),0,63),value=t.min+(1-(y-18)/(e.currentTarget.height-41))*(t.max-t.min);
    selected.row=row;seek(state.pattern*16+row/4);setKey(curve,Math.round(value*100)/100);
  });
  ['program','aux'].forEach(k=>$(k+'Mode').onchange=()=>{state.modes[k]=$(k+'Mode').value;saveLocal();});
  $('sceneBtn').onclick=async()=>{try{const data=await api('catalog');$('patchSelect').replaceChildren(...data.patches.map(p=>new Option(p.path,p.path)));$('patchDialog').showModal();}catch(error){toast(error.message);}};
  $('helpBtn').onclick=()=>$('helpDialog').showModal();$('closeHelp').onclick=()=>$('helpDialog').close();$('helpDialog').addEventListener('click',e=>{if(e.target===$('helpDialog')){const r=e.currentTarget.getBoundingClientRect();if(e.clientX<r.left||e.clientX>r.right||e.clientY<r.top||e.clientY>r.bottom)e.currentTarget.close();}});
  $('saveBtn').onclick=async()=>{
    await flushScore();if(conflicted)return;
    try{
      await api('tracker_save',{revision});
      const u=URL.createObjectURL(new Blob([JSON.stringify(state,null,2)],{type:'application/json'}));
      const a=document.createElement('a');a.href=u;a.download='gorgon-session.json';a.click();setTimeout(()=>URL.revokeObjectURL(u),1000);
      $('savedState').textContent='Saved to workspace';toast('Score saved to .hydrapy/tracker.json and downloaded.');
    }catch(error){toast(error.message);}
  };
  async function importScore(document){
    if(!document)throw new Error('No saved score in this workspace.');
    await flushScore();
    const result=await api('tracker_score',{document,revision});
    state=result.document;revision=result.revision;conflicted=false;
    renderPatterns();renderHeaders();renderRows();renderInstruments();updateControls();updateInspector();
    stopVoices();toast('Score loaded.');
  }
  $('importFile').onchange=async event=>{
    try{const file=event.target.files[0];if(file.size>2000000)throw new Error('Score is too large.');await importScore(JSON.parse(await file.text()));}
    catch(error){toast(error.message);}event.target.value='';
  };
  $('restoreBtn').onclick=async()=>{try{await importScore((await api('tracker_saved')).document);}catch(error){toast(error.message);}};
  $('automationBtn').onclick=()=>transport(()=>({enabled:!remote?.enabled}));
  ['z','roll','feedback'].forEach(lane=>$('bind_'+lane).onchange=()=>{state.bindings[lane]=$('bind_'+lane).value;saveLocal();});
  $('loadPatch').onclick=async()=>{try{await api('load',{path:$('patchSelect').value});$('patchDialog').close();}catch(error){toast(error.message);}};
  $('closePatch').onclick=()=>$('patchDialog').close();
  $('studioLink').href='/#'+location.hash.slice(1);
  $('studioLink').onclick=async event=>{event.preventDefault();await flushScore();if(!conflicted)location.href=$('studioLink').href;};
  $('resetBtn').onclick=async()=>{
    if(!confirm('Replace the current score with the starting study?'))return;
    try{await transport({playing:false,enabled:false,beat:0});await importScore(defaults());}catch(error){toast(error.message);}
  };
  window.addEventListener('keydown',e=>{
    if($('helpDialog').open||$('patchDialog').open)return;if(['INPUT','SELECT','TEXTAREA'].includes(document.activeElement.tagName))return;
    if(e.code==='Space'){e.preventDefault();togglePlay();}if(e.key.toLowerCase()==='f'){e.preventDefault();fullscreen();}if(e.key.toLowerCase()==='b'){e.preventDefault();toggleBlackout();}
    if(['ArrowUp','ArrowDown','ArrowLeft','ArrowRight'].includes(e.key)){e.preventDefault();const r=clamp(selected.row+(e.key==='ArrowDown'?1:e.key==='ArrowUp'?-1:0),0,63),c=clamp(selected.col+(e.key==='ArrowRight'?1:e.key==='ArrowLeft'?-1:0),0,6);selectCell(r,c);$('trackerRows').querySelector(`[data-row="${r}"][data-col="${c}"]`)?.focus({preventScroll:true});$('trackerRows').querySelector(`[data-row="${r}"][data-col="${c}"]`)?.scrollIntoView({block:'nearest'});}
    if(e.key==='Delete'||e.key==='Backspace'){e.preventDefault();clearCell();}if(e.key==='Enter'&&document.activeElement.closest('#trackerRows')){e.preventDefault();toggleCell();}
  });
  renderPatterns();renderHeaders();renderRows();renderInstruments();updateControls();updateInspector();drawVisual($('sceneThumb'),0,'full');
  setInterval(scheduler,25);
  setInterval(()=>{for(const k of ['program','aux']){const connected=Date.now()-outputPresence[k]<2600;$(k+'Btn').classList.toggle('connected',connected);$(k+'Status').textContent=connected?'Window connected':'Open output';$(k==='aux'?'routeAux':'routeProgram').classList.toggle('mint',connected);}},800);
  function frame(){
    const beat=nowBeat(),step=Math.floor(beat*4),pattern=Math.floor(beat/16),row=step%64;
    if(playing&&pattern!==lastPattern&&follow&&state.pattern!==pattern){state.pattern=pattern;renderPatterns();renderRows();updateInspector();}
    if(step!==lastStep){
      $('position').innerHTML=`${pad(Math.floor(beat/4)+1,3)}<span>:</span>${pad(Math.floor(beat%4)+1)}<span>:</span>${pad(step%4)}`;
      $('trackerRows').querySelectorAll('.playing').forEach(e=>e.classList.remove('playing'));
      if(pattern===state.pattern){const el=$('trackerRows').querySelector(`.tracker-row[data-row="${row}"]`);el?.classList.add('playing');if(playing&&follow&&el){const area=$('trackerScroll');if(el.offsetTop<area.scrollTop||el.offsetTop+el.offsetHeight>area.scrollTop+area.clientHeight)area.scrollTop=Math.max(0,el.offsetTop-area.clientHeight*.3);}}
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
