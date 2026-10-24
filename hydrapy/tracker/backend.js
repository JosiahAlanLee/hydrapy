// Python owns automation and rendering. The browser sends edits, never per-frame controls.
const headers = {'X-Hydrapy-Token': location.hash.slice(1), 'Content-Type': 'application/json'};
export async function api(name, data, signal) {
  const response = await fetch('/api/' + name, {
    method: data === undefined ? 'GET' : 'POST', headers, signal: signal || AbortSignal.timeout(5000),
    body: data === undefined ? undefined : JSON.stringify(data),
  });
  const result = await response.json();
  if (!response.ok) throw new Error(result.error || response.statusText);
  return result;
}
const delay = ms => new Promise(resolve => setTimeout(resolve, ms));
export class PreviewStream {
  constructor(bus) {
    Object.assign(this, {bus, output:'', generation:-1, revision:0, image:null, stopped:false,
      stats:{frames:0, samples:[], error:'', output:''}});
    this.loop();
    document.addEventListener('visibilitychange', () => { this.revision++; this.controller?.abort(); });
    window.addEventListener('pagehide', () => {
      this.stopped = true; this.controller?.abort();
      if (this.url) URL.revokeObjectURL(this.url);
    });
  }
  select(output, generation) {
    if (this.output === output && this.generation === generation) return;
    this.output = output; this.generation = generation; this.revision++;
    this.after = -1; this.image = null; this.controller?.abort();
  }
  async loop() {
    while (!this.stopped) {
      if (!this.output || document.hidden) { await delay(100); continue; }
      const revision = this.revision, output = this.output;
      const controller = this.controller = new AbortController();
      let url;
      try {
        const started = performance.now();
        const response = await fetch(`/api/frame?bus=${this.bus}&output=${output}&after=${this.after ?? -1}`,
          {headers, signal:controller.signal});
        if (response.status === 204) { await delay(40); continue; }
        if (!response.ok) throw new Error(`Preview HTTP ${response.status}`);
        const received = performance.now();
        const blob = await response.blob(), transferred = performance.now();
        url = URL.createObjectURL(blob);
        const img = new Image(); img.src = url;
        await img.decode();
        const decoded = performance.now();
        await new Promise(requestAnimationFrame);
        if (revision !== this.revision || document.hidden || this.stopped) continue;
        const previous = this.url;
        this.image = img; this.url = url; url = null;
        if (previous) URL.revokeObjectURL(previous);
        this.after = Number(response.headers.get('X-Preview-Sequence'));
        const timings = {};
        for (const item of (response.headers.get('Server-Timing') || '').split(',')) {
          const match = item.trim().match(/^([\w]+);dur=([\d.]+)$/);
          if (match) timings[match[1]] = Number(match[2]);
        }
        this.stats.frames++; this.stats.output = output; this.stats.error = '';
        this.stats.samples.push({...timings, output, sequence:this.after,
          request_ms:received-started, transfer_ms:transferred-received,
          decode_ms:decoded-transferred, display_schedule_ms:performance.now()-decoded,
          frame_age_estimate_ms:Number(response.headers.get('X-Preview-Age-Ms'))+performance.now()-received});
        if (this.stats.samples.length > 300) this.stats.samples.shift();
      } catch (error) {
        if (error.name !== 'AbortError') { this.stats.error = error.message; await delay(500); }
      } finally { if (url) URL.revokeObjectURL(url); }
    }
  }
  draw(canvas, blackout = false) {
    const context = canvas.getContext('2d'), rect = canvas.getBoundingClientRect();
    const width = Math.max(1, Math.round(rect.width * Math.min(devicePixelRatio, 2)));
    const height = Math.max(1, Math.round(rect.height * Math.min(devicePixelRatio, 2)));
    if (canvas.width !== width || canvas.height !== height) { canvas.width = width; canvas.height = height; }
    context.fillStyle = '#000'; context.fillRect(0, 0, width, height);
    if (!blackout && this.image) {
      const scale = Math.min(width / this.image.width, height / this.image.height);
      const w = this.image.width * scale, h = this.image.height * scale;
      context.drawImage(this.image, (width-w)/2, (height-h)/2, w, h);
    }
  }
}
