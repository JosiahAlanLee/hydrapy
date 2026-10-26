// Raw machine controls. The external instrument supplies native defaults.
export const SOUND_KEYS = ['machine', ...Array.from({length:8}, (_,i)=>`p${i+1}`)];
export function validateLocks(value, {machine=true}={}) {
  if (!value || typeof value !== 'object' || Array.isArray(value)) throw new Error('Sound controls must be an object.');
  for (const [key, raw] of Object.entries(value)) {
    if (!SOUND_KEYS.includes(key) || (!machine && key==='machine') ||
        !Number.isInteger(raw) || raw<0 || raw>(key==='machine'?191:127)) throw new Error('Invalid sound control: '+key);
  }
  return value;
}
export function ensureSounds(score) {
  score.format='hydrapy-tracker-2';
  score.sounds ??= Array.from({length:4},()=>({machine:null,params:{}}));
}
export function resolveSound(score, pattern, lane, row) {
  const result=structuredClone(score.sounds?.[lane]??{machine:null,params:{}});
  for (const [key,value] of Object.entries(score.patterns[pattern].audio[lane][row]?.locks??{})) {
    if (key==='machine') result.machine=value; else result.params[key]=value;
  }
  return result;
}
