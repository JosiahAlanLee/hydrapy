// Run the production transport and audio scheduler with a virtual clock.
// This covers minutes of playback without a browser's wall-clock wait or audio device.
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const source = fs.readFileSync(process.argv[2], 'utf8');
function functions(from, to) {
  const start = source.indexOf(`function ${from}(`), end = source.indexOf(`function ${to}(`, start);
  assert(start >= 0 && end > start);
  return source.slice(start, end);
}
const production = [functions('normalizeBeat', 'valueAt'), functions('stopVoices', 'connectVoice'),
  functions('adoptTransport', 'transport'), functions('scheduler', 'seek')].join('\n');

for (const [loop, pattern] of [[true, 0], [true, 2], [true, 3], [false, 0]]) {
  const context = vm.createContext({assert, loop, pattern});
  vm.runInContext(`
    let ms=0;
    const performance={now:()=>ms}, mod=(n,m)=>((n%m)+m)%m;
    const element={hidden:false,setAttribute(){},classList:{toggle(){}},innerHTML:'',textContent:''};
    const $=()=>element;
    const state={loop,pattern,bpm:120,sound:true,mutes:[false,false,false,false],
      patterns:Array.from({length:4},()=>({audio:[Array.from({length:64},()=>({note:'C-2',vel:100})),
        Array(64).fill(null),Array(64).fill(null),Array(64).fill(null)]}))};
    const start=loop?pattern*16:0, length=loop?16:64, seconds=length/2;
    let playing=true,anchorBeat=start,anchorTime=0,blackout=false,nextTick=null,transportRevision=null;
    const audioContext={state:'running',currentTime:0},voices=new Set(),hits=[];
    function soundHit(track,hit,t){hits.push(t);}
    ${production}
    for(ms=0;ms<seconds*3*1000;ms+=25){
      audioContext.currentTime=ms/1000;
      if(ms%150===0)adoptTransport({beat:start+mod(ms/1000*2,length),playing:true,enabled:true,transport_revision:1});
      scheduler();
    }
    // All three cycles contain a full set of hits; no duplicates at a wrap.
    for(let i=0;i<3;i++){
      const count=hits.filter(t=>t>=i*seconds-.00001 && t<(i+1)*seconds-.00001).length;
      assert.equal(count,length*4,'complete cycle '+i);
    }
    for(let i=1;i<hits.length;i++)assert(hits[i]-hits[i-1]>.11,'duplicate scheduled hit');
    // A short seek must invalidate lookahead even below the drift threshold.
    let cancelled=0;voices.add({stop(){cancelled++;}});
    adoptTransport({beat:normalizeBeat(rawBeat()-.25),playing:true,enabled:true,transport_revision:2});
    assert.equal(cancelled,1);assert.equal(nextTick,null);
    scheduler();assert.notEqual(nextTick,null);
    adoptTransport({beat:nowBeat(),playing:false,enabled:true,transport_revision:2});
    assert.equal(nextTick,null);
    const before=hits.length;scheduler();assert.equal(hits.length,before);
    adoptTransport({beat:nowBeat(),playing:true,enabled:true,transport_revision:2});
    scheduler();assert(hits.length>before,'resume schedules again');
  `, context);
}
console.log('tracker scheduler: looped patterns, whole-score wrap, short seek and pause/resume passed');
