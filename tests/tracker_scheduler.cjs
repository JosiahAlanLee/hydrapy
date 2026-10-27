// Run the production transport and audio scheduler with a virtual clock.
// This covers minutes of playback without a browser's wall-clock wait or audio device.
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const {pathToFileURL} = require('node:url');
const source = fs.readFileSync(process.argv[2], 'utf8');
function functions(from, to) {
  const start = source.indexOf(`function ${from}(`), end = source.indexOf(`function ${to}(`, start);
  assert(start >= 0 && end > start);
  return source.slice(start, end);
}
const production = [functions('line', 'valueAt'), functions('stopVoices', 'connectVoice'),
  functions('adoptTransport', 'transport'), functions('scheduler', 'seek')].join('\n');

(async () => {
  const score3 = await import(pathToFileURL(path.join(path.dirname(process.argv[2]), 'score3.js')).href);
  // An older score (looped on three patterns, then the whole song), and a version-3 one whose
  // playhead comes from Studio's segment fields, with a ply that doubles every hit.
  for (const [loop, pattern, v3] of [[true, 0, false], [true, 2, false], [true, 3, false], [false, 0, false], [true, 1, true]]) {
    const context = vm.createContext({assert, loop, pattern, v3, ...score3});
    vm.runInContext(`
      let ms=0;
      const performance={now:()=>ms}, mod=(n,m)=>((n%m)+m)%m;
      const element={hidden:false,setAttribute(){},classList:{toggle(){}},innerHTML:'',textContent:''};
      const $=()=>element;
      const legacy={format:'hydrapy-tracker-1',loop,pattern,bpm:120,key:2,sound:true,mutes:[false,false,false,false],
        modes:{program:'o0',aux:'o1'},bindings:{z:'',roll:'',feedback:''},
        patterns:Array.from({length:4},(_,p)=>({name:'P'+p,audio:[Array.from({length:64},()=>({note:'C-2',vel:100})),
          Array(64).fill(null),Array(64).fill(null),Array(64).fill(null)],keys:{z:[],roll:[],feedback:[]},
          easing:{z:'hold',roll:'hold',feedback:'hold'},scenes:[]}))};
      const state=incoming(legacy);
      if(v3){delete state.legacy;state.patterns[pattern].lanes[0].transforms=[{kind:'ply',n:2}];}
      const per=v3?8:4, start=loop?pattern*16:0, length=loop?16:64, seconds=length/2;
      let playing=true,segTick=0,uAnchor=0,anchorTime=0,blackout=false,audioFrontier=null,transportRevision=null,remote=null;
      const audioContext={state:'running',currentTime:0},voices=new Set(),hits=[];
      function soundHit(track,hit,t){hits.push(t);}
      ${production}
      // Studio: an older score's beat counts a loop from the song's start; a version-3 one sends its segment.
      const reply=(sec,revision,playing=true)=>v3
        ?{start:{tick:0,entry:null},ticks:sec*2*TICKS,beat:mod(sec*2,length),playing,enabled:true,transport_revision:revision}
        :{beat:start+mod(sec*2,length),playing,enabled:true,transport_revision:revision};
      for(ms=0;ms<seconds*3*1000;ms+=25){
        audioContext.currentTime=ms/1000;
        if(ms%150===0)adoptTransport(reply(ms/1000,1));
        scheduler();
      }
      // All three cycles contain a full set of hits; no duplicates at a wrap.
      for(let i=0;i<3;i++){
        const count=hits.filter(t=>t>=i*seconds-.00001 && t<(i+1)*seconds-.00001).length;
        assert.equal(count,length*per,'complete cycle '+i);
      }
      for(let i=1;i<hits.length;i++)assert(hits[i]-hits[i-1]>.5/per-.01,'duplicate scheduled hit');
      // A short seek must invalidate lookahead even below the drift threshold.
      let cancelled=0;voices.add({stop(){cancelled++;}});
      adoptTransport(v3?{...reply(ms/1000,2),start:{tick:480,entry:null},ticks:0}:{beat:start+mod(nowBeat()-.25,length),playing:true,enabled:true,transport_revision:2});
      assert.equal(cancelled,1);assert.equal(audioFrontier,null);
      scheduler();assert.notEqual(audioFrontier,null);
      adoptTransport({...(v3?{start:{tick:segTick,entry:null},ticks:nowTicks()}:{}),beat:start+nowBeat(),playing:false,enabled:true,transport_revision:2});
      assert.equal(audioFrontier,null);
      const before=hits.length;scheduler();assert.equal(hits.length,before);
      adoptTransport({...(v3?{start:{tick:segTick,entry:null},ticks:nowTicks()}:{}),beat:start+nowBeat(),playing:true,enabled:true,transport_revision:2});
      scheduler();assert(hits.length>before,'resume schedules again');
    `, context);
  }
  console.log('tracker scheduler: looped patterns, whole-score wrap, version 3 ply, short seek and pause/resume passed');
})().catch(error => { console.error(error); process.exit(1); });
