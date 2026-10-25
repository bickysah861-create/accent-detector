const $ = s => document.querySelector(s);
const el = (t, c, h) => { const n = document.createElement(t);
  if (c) n.className = c; if (h !== undefined) n.innerHTML = h; return n; };
const esc = s => (s || '').replace(/[&<>]/g, m => ({ '&':'&amp;','<':'&lt;','>':'&gt;' }[m]));
const col = v => v >= 80 ? 'var(--ok)' : v >= 60 ? 'var(--warn)' : 'var(--bad)';

let CID = null, CONVOS = [], rec = null, chunks = [], stream = null,
    actx = null, analyser = null, raf = 0, t0 = 0, tick = 0, busy = false;

/* ------------------------------------------------------------- convos */
async function loadConvos(){
  CONVOS = await (await fetch('/api/convos')).json();
  const box = $('#convos'); box.innerHTML = '';
  CONVOS.forEach(c => {
    const d = el('div', 'cv' + (c.id === CID ? ' on' : ''));
    d.innerHTML = `<span class="nm">${esc(c.title)}</span>
                   <span class="n">${c.turns}</span>
                   <button class="x" title="delete">×</button>`;
    d.onclick = e => { if (e.target.classList.contains('x')) return; open(c.id); };
    d.querySelector('.x').onclick = async e => {
      e.stopPropagation();
      await fetch('/api/convos/' + c.id, { method: 'DELETE' });
      if (CID === c.id){ CID = null; $('#stream').innerHTML = ''; showEmpty(); }
      loadConvos();
    };
    box.appendChild(d);
  });
}

async function newConvo(){
  const c = await (await fetch('/api/convos', { method: 'POST' })).json();
  CID = c.id; await loadConvos(); render(c);
  $('#side').classList.remove('open');
}

async function open(id){
  CID = id;
  const c = await (await fetch('/api/convos/' + id)).json();
  await loadConvos(); render(c);
  $('#side').classList.remove('open');
}

function showEmpty(){
  $('#stream').innerHTML = '';
  const e = el('div', 'empty');
  e.innerHTML = `<div class="orb"></div><h2>Speak, and I will listen</h2>
    <p>This session is voice only. Press the microphone, say a few sentences, and you
       will get your accent reading plus a pronunciation and fluency review measured
       from the audio itself.</p>`;
  $('#stream').appendChild(e);
  $('#ctitle').textContent = 'Accent Coach';
}

function render(c){
  $('#ctitle').textContent = c.title || 'New session';
  const s = $('#stream'); s.innerHTML = '';
  if (!c.messages.length){
    s.innerHTML = `<div class="empty"><div class="orb"></div>
      <h2>Ready when you are</h2><p>Press the microphone and speak for a few seconds.</p></div>`;
    return;
  }
  c.messages.forEach(m => m.role === 'user' ? addUser(m) : addBot(m));
  // Open the newest measurement panel. The result is the point of the app, so
  // the latest one should be readable without a click; older turns stay folded
  // to keep the transcript scannable.
  const reports = s.querySelectorAll('.report');
  if (reports.length) reports[reports.length - 1].classList.add('open');
  // Jump, do not animate: opening a long session should land on the newest
  // turn immediately rather than scrolling through the whole history.
  requestAnimationFrame(() => { s.scrollTop = s.scrollHeight; });
}

/* ------------------------------------------------------------ message */
function addUser(m){
  const d = el('div', 'msg user');
  d.innerHTML = `<div class="av">YOU</div><div class="body">
    <div class="who">you spoke</div>
    <div class="said">${esc(m.transcript) || '<i style="color:var(--mut)">no speech detected</i>'}
      <span class="dur">${m.duration}s</span></div></div>`;
  if (m.analysis && m.analysis.ok) {
    const r = report(m.analysis);
    if (m.open) r.classList.add('open');
    d.querySelector('.body').appendChild(r);
  }
  $('#stream').appendChild(d);
}

function addBot(m){
  const d = el('div', 'msg bot');
  const html = esc(m.text).replace(/\*\*(.+?)\*\*/g, '<b>$1</b>').replace(/\n/g, '<br>');
  d.innerHTML = `<div class="av">AI</div><div class="body">
    <div class="who">coach${m.backend ? ' · ' + m.backend : ''}${m.took ? ' · ' + m.took + 's' : ''}</div>
    <div class="text">${html}</div></div>`;
  $('#stream').appendChild(d);
}

function bar(label, v, max){
  const pct = Math.max(0, Math.min(100, (v / (max || 100)) * 100));
  return `<div class="row"><span class="lb">${label}</span>
    <div class="bar"><div class="fill" style="width:${pct}%;background:${col(pct)}"></div></div>
    <span class="vv" style="color:${col(pct)}">${typeof v === 'number' ? v.toFixed(1) : v}</span></div>`;
}

function report(a){
  const r = el('div', 'report');
  const acc = a.accent || {}, p = a.pronunciation || {}, f = a.fluency || {}, pr = a.prosody || {};
  let h = `<div class="rhead"><span class="chev">▶</span>
      <span class="cap">measurements from this recording</span>
      <b style="color:${col(a.overall)}">${a.overall}</b></div><div class="rbody">`;

  h += `<div class="scores">
    <div class="sc"><label>overall</label><b style="color:${col(a.overall)}">${a.overall}</b></div>
    <div class="sc"><label>pronunciation</label><b style="color:${col(p.score||0)}">${p.score ?? 'n/a'}</b></div>
    <div class="sc"><label>fluency</label><b style="color:${col(f.fluency||0)}">${f.fluency ?? 'n/a'}</b></div>
    <div class="sc"><label>clarity</label><b style="color:${col(a.clarity||0)}">${a.clarity}</b></div></div>`;

  if (acc.available){
    h += `<div class="sec"><h4>accent profile</h4>`;
    acc.top.forEach(t => { h += bar(t.accent, +(t.prob * 100).toFixed(1)); });
    h += `<div class="kv" style="margin-top:8px">
      <span>certainty <b>${acc.certainty}</b></span>
      <span>margin <b>${(acc.margin*100).toFixed(1)}%</b></span>
      <span>native variety <b>${(acc.native_variety_prob*100).toFixed(1)}%</b></span>
      <span>windows <b>${acc.windows}</b></span></div></div>`;
  } else {
    h += `<div class="sec"><h4>accent profile</h4>
      <div class="kv"><span>${esc(acc.reason || 'unavailable')}</span></div></div>`;
  }

  if (p.weak_groups && p.weak_groups.length){
    h += `<div class="sec"><h4>pronunciation by sound group</h4>`;
    p.weak_groups.forEach(g => { h += bar(g.group, g.score); });
    h += `</div>`;
  }
  if (p.weak_words && p.weak_words.length){
    h += `<div class="sec"><h4>words to repeat</h4><div class="chips">` +
      p.weak_words.map(w => `<span class="chip">${esc(w.word)}
        <b style="color:${col(w.score)}">${w.score}</b></span>`).join('') + `</div></div>`;
  }
  if (f.components){
    h += `<div class="sec"><h4>fluency components</h4>`;
    Object.entries(f.components).forEach(([k, v]) => { h += bar(k, v); });
    h += `</div>`;
  }
  h += `<div class="sec"><h4>signal measurements</h4><div class="kv">
    <span>duration <b>${pr.duration_s}s</b></span>
    <span>speaking time <b>${pr.speech_time_s}s</b></span>
    <span>words per minute <b>${pr.words_per_min}</b></span>
    <span>articulation <b>${pr.articulation_syl_per_s} syl/s</b></span>
    <span>pauses <b>${pr.pause_count}</b></span>
    <span>longest pause <b>${pr.longest_pause_s}s</b></span>
    <span>pause ratio <b>${(pr.pause_ratio*100).toFixed(0)}%</b></span>
    <span>median pitch <b>${pr.f0_mean} Hz</b></span>
    <span>pitch range <b>${pr.f0_range_semitones} st</b></span></div></div></div>`;
  r.innerHTML = h;
  r.querySelector('.rhead').onclick = () => r.classList.toggle('open');
  return r;
}

/* ------------------------------------------------------------ recorder */
function drawViz(){
  const cv = $('#viz'), ctx = cv.getContext('2d');
  const d = window.devicePixelRatio || 1, w = cv.clientWidth;
  cv.width = w * d; cv.height = 44 * d; ctx.setTransform(d, 0, 0, d, 0, 0);
  const buf = new Uint8Array(analyser.frequencyBinCount);
  const loop = () => {
    raf = requestAnimationFrame(loop);
    analyser.getByteFrequencyData(buf);
    ctx.clearRect(0, 0, w, 44);
    const n = 52, step = Math.floor(buf.length / n);
    for (let i = 0; i < n; i++){
      const v = buf[i * step] / 255, h = Math.max(2, v * 40);
      const x = i * (w / n), bw = (w / n) - 2.5;
      const g = ctx.createLinearGradient(0, 44 - h, 0, 44);
      g.addColorStop(0, '#7b61ff'); g.addColorStop(1, '#3aa0ff');
      ctx.fillStyle = g;
      ctx.fillRect(x, (44 - h) / 2, bw, h);
    }
  };
  loop();
}

async function startRec(){
  if (!CID) await newConvo();
  try{
    stream = await navigator.mediaDevices.getUserMedia({
      audio: { echoCancellation: true, noiseSuppression: true, autoGainControl: true } });
  }catch(e){
    $('#state').textContent = 'microphone permission denied'; return;
  }
  actx = new (window.AudioContext || window.webkitAudioContext)();
  const src = actx.createMediaStreamSource(stream);
  analyser = actx.createAnalyser(); analyser.fftSize = 256; src.connect(analyser);
  $('#viz').classList.add('on'); drawViz();

  chunks = [];
  const mime = MediaRecorder.isTypeSupported('audio/webm;codecs=opus')
    ? 'audio/webm;codecs=opus' : (MediaRecorder.isTypeSupported('audio/webm') ? 'audio/webm' : '');
  rec = new MediaRecorder(stream, mime ? { mimeType: mime } : undefined);
  rec.ondataavailable = e => { if (e.data.size) chunks.push(e.data); };
  rec.onstop = send;
  rec.start();
  t0 = Date.now();
  tick = setInterval(() => {
    const s = (Date.now() - t0) / 1000;
    $('#timer').textContent = s.toFixed(1) + 's';
    if (s > 60) stopRec();
  }, 100);
  $('#mic').classList.add('rec');
  $('#state').textContent = 'listening, press again to finish';
}

function stopRec(){
  clearInterval(tick);
  if (rec && rec.state !== 'inactive') rec.stop();
  if (stream) stream.getTracks().forEach(t => t.stop());
  cancelAnimationFrame(raf);
  if (actx) actx.close().catch(()=>{});
  $('#viz').classList.remove('on');
  $('#mic').classList.remove('rec');
  $('#timer').textContent = '';
}

async function send(){
  const secs = (Date.now() - t0) / 1000;
  const blob = new Blob(chunks, { type: chunks[0]?.type || 'audio/webm' });
  if (secs < 1.0 || blob.size < 1500){
    $('#state').textContent = 'that was too short, try a few seconds more'; return;
  }
  busy = true; $('#mic').disabled = true;
  $('#state').textContent = 'analysing your speech';
  const e = $('#stream').querySelector('.empty'); if (e) e.remove();

  const d = el('div', 'msg bot');
  d.innerHTML = `<div class="av">AI</div><div class="body"><div class="who">coach</div>
    <div class="think"><i></i><i></i><i></i></div></div>`;
  $('#stream').appendChild(d);
  $('#stream').scrollTop = $('#stream').scrollHeight;

  try{
    const fd = new FormData();
    fd.append('cid', CID); fd.append('audio', blob, 'a.webm');
    const r = await fetch('/api/chat', { method: 'POST', body: fd });
    if (!r.ok) throw new Error((await r.text()).slice(0, 200));
    const j = await r.json();
    d.remove();
    addUser({ transcript: j.analysis.transcript || '', duration: +secs.toFixed(2),
              analysis: j.analysis, open: true });
    addBot({ text: j.reply, backend: j.backend, took: j.took });
    $('#ctitle').textContent = j.title;
    loadConvos();
  }catch(err){
    d.querySelector('.body').innerHTML =
      `<div class="who">coach</div><div class="err">${esc(String(err.message || err))}</div>`;
  }finally{
    busy = false; $('#mic').disabled = false;
    $('#state').textContent = 'press to speak';
    $('#stream').scrollTo({ top: $('#stream').scrollHeight, behavior: 'smooth' });
  }
}

/* ---------------------------------------------------------------- init */
$('#mic').onclick = () => {
  if (busy) return;
  rec && rec.state === 'recording' ? stopRec() : startRec();
};
$('#newbtn').onclick = newConvo;
$('#burger').onclick = () => $('#side').classList.toggle('open');
document.addEventListener('keydown', e => {
  if (e.code === 'Space' && e.target === document.body){
    e.preventDefault(); $('#mic').click();
  }
});

(async () => {
  await loadConvos();
  try{
    const m = await (await fetch('/api/model_status')).json();
    const t = m.metrics?.test;
    $('#mstat').textContent = m.accent_model
      ? (t ? `accent model ready · test macro F1 ${t.macro_f1.toFixed(3)}` : 'accent model ready')
      : 'accent model not trained yet';
  }catch(e){ $('#mstat').textContent = 'model status unavailable'; }
  if (CONVOS.length) open(CONVOS[0].id); else showEmpty();
})();
