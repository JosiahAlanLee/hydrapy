// Same note convention as studio_tracker.note_midi: C-4 = C4 = MIDI 60.
export function noteMidi(note){
  if(note==='AIR')return null;
  const m=/^([A-G])([-#]?)([0-9])$/.exec(note);
  if(m){
    const n=12*(Number(m[3])+1)+{C:0,D:2,E:4,F:5,G:7,A:9,B:11}[m[1]]+(m[2]==='#'?1:0);
    if(n<=127&&!(m[2]==='#'&&'BE'.includes(m[1])))return n;
  }
  throw new Error('Note must be C0–G9 (sharps allowed), or AIR.');
}
export function canonicalNote(note){
  noteMidi(note);
  return note==='AIR'||note.length===3?note:note[0]+'-'+note[1];
}
// The synthesized Breaks kit: C-2 (MIDI 36) is the kick, D-2 (38) the snare, any other note a hat.
// By pitch, not text: C2 is C-2 (a score written outside the editor keeps its spelling).
export function breaksVoice(note){
  const midi=noteMidi(note);
  return midi===36?'kick':midi===38?'snare':'hat';
}
export function noteFrequency(note,key=2){
  const midi=noteMidi(note);
  return midi===null?null:440*2**((midi+key-2-69)/12);
}
