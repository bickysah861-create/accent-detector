const $ = id => document.getElementById(id);
const CSS = getComputedStyle(document.documentElement);
const C = n => CSS.getPropertyValue(n).trim();

/* high dpi canvas sizing */
function fit(cv){
  const r = cv.getBoundingClientRect(), d = window.devicePixelRatio || 1;
  cv.width = Math.max(1, r.width * d);
  cv.height = Math.max(1, (cv.getAttribute('height') | 0) * d);
  const x = cv.getContext('2d');
  x.setTransform(d, 0, 0, d, 0, 0);
  return { x, w: r.width, h: cv.getAttribute('height') | 0 };
}
const lerp = (a, b, t) => a + (b - a) * t;

/* ---------------------------------------------------- learning curve */
function drawCurve(st){
  const { x, w, h } = fit($('curve'));
  x.clearRect(0, 0, w, h);
  const H = st.history || [];
  const pad = { l: 44, r: 46, t: 14, b: 24 };
  const iw = w - pad.l - pad.r, ih = h - pad.t - pad.b;
  x.strokeStyle = '#18243a'; x.lineWidth = 1; x.font = '10px ui-monospace';
  for (let i = 0; i <= 4; i++){
    const yy = pad.t + ih * i / 4;
    x.beginPath(); x.moveTo(pad.l, yy); x.lineTo(pad.l + iw, yy); x.stroke();
    x.fillStyle = '#5c6a82'; x.textAlign = 'right';
    x.fillText((1 - i / 4).toFixed(2), pad.l - 7, yy + 3);
  }
  if (!H.length){
    x.fillStyle = '#5c6a82'; x.textAlign = 'center';
    x.fillText('waiting for first epoch', w / 2, h / 2); return;
  }
  const n = H.length, maxL = Math.max(...H.map(d => d.loss), 0.001);
  const X = i => pad.l + (n === 1 ? iw / 2 : iw * i / (n - 1));

  const line = (vals, col, fill) => {
    x.beginPath();
    vals.forEach((v, i) => { const yy = pad.t + ih * (1 - v);
      i ? x.lineTo(X(i), yy) : x.moveTo(X(i), yy); });
    x.strokeStyle = col; x.lineWidth = 2; x.lineJoin = 'round'; x.stroke();
    if (fill){
      x.lineTo(X(n - 1), pad.t + ih); x.lineTo(X(0), pad.t + ih); x.closePath();
      const g = x.createLinearGradient(0, pad.t, 0, pad.t + ih);
      g.addColorStop(0, col + '33'); g.addColorStop(1, col + '00');
      x.fillStyle = g; x.fill();
    }
  };
  const maxLR = Math.max(...H.map(d => d.lr), 1e-9);
  line(H.map(d => d.lr / maxLR * 0.28), C('--acc2'), false);
  line(H.map(d => Math.min(d.loss / maxL, 1)), C('--warn'), false);
  line(H.map(d => d.val_macro_f1), C('--acc'), true);

  const last = H[n - 1];
  x.fillStyle = C('--acc'); x.textAlign = 'left'; x.font = '600 11px ui-monospace';
  x.fillText(last.val_macro_f1.toFixed(3), pad.l + iw + 6, pad.t + ih * (1 - last.val_macro_f1) + 3);
  x.fillStyle = C('--warn');
  x.fillText(last.loss.toFixed(3), pad.l + iw + 6, pad.t + ih * (1 - Math.min(last.loss / maxL, 1)) + 3);
  x.fillStyle = '#5c6a82'; x.textAlign = 'center'; x.font = '10px ui-monospace';
  x.fillText('epoch ' + last.epoch, pad.l + iw / 2, h - 6);
}

/* ------------------------------------------- SSL layer reliance bars */
function drawLayers(st){
  const box = $('layers'), lw = st.layer_weights || [], names = st.ssl_layers || [];
  if (!lw.length){
    box.innerHTML = '<p class="hint">The winning model is a linear probe on all four '
      + 'cached layers at once, so it learns no explicit layer mix. This panel fills in '
      + 'whenever an MLP head is training.</p>';
    return;
  }
  if (box.children.length !== lw.length){
    box.innerHTML = names.map(n =>
      `<div class="lrow"><span class="nm">${n}</span>
       <div class="lbar"><div class="lfill"></div></div><span class="pc"></span></div>`).join('');
  }
  const mx = Math.max(...lw);
  [...box.children].forEach((row, i) => {
    row.querySelector('.lfill').style.width = (lw[i] / mx * 100).toFixed(1) + '%';
    row.querySelector('.pc').textContent = (lw[i] * 100).toFixed(1) + '%';
  });
}

/* ------------------------------------------ live network behaviour */
let tphase = 0;
function drawNet(st){
  const { x, w, h } = fit($('net'));
  x.clearRect(0, 0, w, h);
  const acts = st.act_stats || {}, gn = st.grad_norms || {};
  const names = Object.keys(acts);
  const cols = [
    { k: 'input',      label: 'embedding\n6144', n: 8, act: null },
    ...names.map(k => ({ k, label: k, n: 7, act: acts[k] })),
    { k: 'out',        label: 'accent\n23', n: 6, act: null },
  ];
  if (!names.length){
    x.fillStyle = '#5c6a82'; x.textAlign = 'center';
    x.font = '11px ui-monospace'; x.fillText('waiting for first forward pass', w / 2, h / 2); return;
  }
  const pad = 56, span = (w - pad * 2) / (cols.length - 1);
  const gmax = Math.max(...Object.values(gn), 1e-9);
  // Probe names and parameter names differ, so map them explicitly rather than
  // by string surgery: mlp_block1 is net.0, mlp_block2 is net.3, and the
  // classifier is `out`. Guessing here silently drew uniform edges.
  const GPREFIX = {
    layer_mix: ['layer_logits', 'norm.'],
    mlp_block1: ['net.0.'],
    mlp_block2: ['net.3.'],
    classifier: ['out.'],
  };
  const gForCol = i => {
    const pre = GPREFIX[cols[i].k];
    if (!pre) return gmax * 0.15;
    const hit = Object.keys(gn).filter(n => pre.some(p => n.startsWith(p)));
    return hit.length ? Math.max(...hit.map(n => gn[n])) : gmax * 0.15;
  };
  const pos = cols.map((c, i) => {
    const cx = pad + span * i, gap = Math.min(26, (h - 86) / c.n);
    const y0 = h / 2 - gap * (c.n - 1) / 2;
    return { ...c, cx, pts: Array.from({ length: c.n }, (_, j) => ({ x: cx, y: y0 + gap * j })) };
  });

  /* edges, thickness by gradient magnitude, animated pulse along flow */
  for (let i = 0; i < pos.length - 1; i++){
    const g = gForCol(i + 1) / gmax;
    x.lineWidth = lerp(0.25, 2.3, Math.min(g, 1));
    pos[i].pts.forEach(a => pos[i + 1].pts.forEach(b => {
      const grad = x.createLinearGradient(a.x, a.y, b.x, b.y);
      grad.addColorStop(0, 'rgba(55,182,255,.05)');
      grad.addColorStop(0.5, `rgba(55,182,255,${lerp(.07, .4, Math.min(g, 1))})`);
      grad.addColorStop(1, 'rgba(124,92,255,.05)');
      x.strokeStyle = grad;
      x.beginPath(); x.moveTo(a.x + 8, a.y); x.lineTo(b.x - 8, b.y); x.stroke();
    }));
    /* travelling activation pulse */
    const t = (tphase + i * 0.17) % 1;
    const a = pos[i].pts[i % pos[i].pts.length], b = pos[i + 1].pts[(i + 1) % pos[i + 1].pts.length];
    x.beginPath();
    x.arc(lerp(a.x + 8, b.x - 8, t), lerp(a.y, b.y, t), 2.2, 0, 7);
    x.fillStyle = 'rgba(46,230,168,.85)'; x.fill();
  }

  /* nodes: brightness by activation magnitude, red ring when units go quiet */
  pos.forEach(c => {
    const a = c.act;
    const mag = a ? Math.min(Math.abs(a.mean) + a.std, 2) / 2 : 0.5;
    const dead = a ? a.dead_frac : 0;
    c.pts.forEach(p => {
      const r = 6.5;
      const g = x.createRadialGradient(p.x, p.y, 0, p.x, p.y, r * 2.4);
      g.addColorStop(0, `rgba(46,230,168,${lerp(.12, .95, mag)})`);
      g.addColorStop(1, 'rgba(46,230,168,0)');
      x.fillStyle = g; x.beginPath(); x.arc(p.x, p.y, r * 2.4, 0, 7); x.fill();
      x.beginPath(); x.arc(p.x, p.y, r, 0, 7);
      x.fillStyle = `rgba(20,40,58,.95)`; x.fill();
      x.lineWidth = 1.6;
      x.strokeStyle = dead > 0.45 ? C('--bad') : `rgba(46,230,168,${lerp(.3, 1, mag)})`;
      x.stroke();
    });
    x.fillStyle = '#8d9ab2'; x.font = '10px ui-monospace'; x.textAlign = 'center';
    c.label.split('\n').forEach((ln, k) =>
      x.fillText(ln, c.cx, h - 20 + k * 11));
  });
  tphase = (tphase + 0.012) % 1;
}

/* ------------------------------------------------- module stat table */
function drawMod(st){
  const a = st.act_stats || {}, gn = st.grad_norms || {};
  const rows = Object.entries(a).map(([k, v]) =>
    `<tr><td>${k}</td><td>μ ${v.mean.toFixed(2)} σ ${v.std.toFixed(2)}</td>
     <td>${(v.dead_frac * 100).toFixed(0)}% idle</td></tr>`);
  const g = Object.entries(gn).slice(0, 6).map(([k, v]) =>
    `<tr><td>∇ ${k.replace('net.', 'mlp.')}</td><td></td><td>${v.toFixed(4)}</td></tr>`);
  $('modstats').querySelector('tbody').innerHTML = rows.join('') + g.join('');
}

/* ------------------------------------------------------ per class F1 */
function drawClasses(st){
  const f1 = st.per_class_f1 || [], names = st.accents || [];
  const box = $('classes');
  if (!f1.length){ box.innerHTML = '<p class="hint">available after the first validation pass</p>'; return; }
  const order = f1.map((v, i) => [v, i]).sort((a, b) => b[0] - a[0]);
  box.innerHTML = order.map(([v, i]) => {
    const col = v > .7 ? C('--ok') : v > .45 ? C('--warn') : C('--bad');
    return `<div class="crow"><span class="cn">${names[i]}</span>
      <div class="cbar"><div class="cfill" style="width:${(v*100).toFixed(1)}%;background:${col}"></div></div>
      <span class="cv" style="color:${col}">${v.toFixed(2)}</span></div>`;
  }).join('');
}

/* ------------------------------------------------- confusion matrix */
function drawCM(st){
  const { x, w, h } = fit($('cm'));
  x.clearRect(0, 0, w, h);
  const M = st.confusion || [];
  if (!M.length){ x.fillStyle = '#5c6a82'; x.textAlign = 'center'; x.font = '11px ui-monospace';
    x.fillText('awaiting validation', w / 2, h / 2); return; }
  const n = M.length, s = Math.min((w - 10) / n, (h - 10) / n);
  const ox = (w - s * n) / 2, oy = (h - s * n) / 2;
  M.forEach((row, i) => {
    const tot = row.reduce((a, b) => a + b, 0) || 1;
    row.forEach((v, j) => {
      const f = v / tot;
      x.fillStyle = i === j
        ? `rgba(46,230,168,${0.10 + f * 0.9})`
        : `rgba(255,92,114,${f * 0.92})`;
      x.fillRect(ox + j * s, oy + i * s, Math.max(s - .6, 1), Math.max(s - .6, 1));
    });
  });
  x.strokeStyle = '#1d2a41'; x.lineWidth = 1; x.strokeRect(ox, oy, s * n, s * n);
}

/* ------------------------------------------------------- step losses */
function drawSteps(st){
  const { x, w, h } = fit($('steps'));
  x.clearRect(0, 0, w, h);
  const S = st.step_loss || [];
  if (S.length < 2){ x.fillStyle = '#5c6a82'; x.textAlign = 'center'; x.font = '11px ui-monospace';
    x.fillText('no steps yet', w / 2, h / 2); return; }
  const mn = Math.min(...S), mx = Math.max(...S), rng = (mx - mn) || 1;
  x.beginPath();
  S.forEach((v, i) => {
    const px = 6 + (w - 12) * i / (S.length - 1);
    const py = 10 + (h - 26) * (1 - (v - mn) / rng);
    i ? x.lineTo(px, py) : x.moveTo(px, py);
  });
  x.strokeStyle = C('--warn'); x.lineWidth = 1.4; x.stroke();
  x.fillStyle = '#5c6a82'; x.font = '10px ui-monospace'; x.textAlign = 'left';
  x.fillText('min ' + mn.toFixed(3), 6, h - 4);
  x.textAlign = 'right'; x.fillText('max ' + mx.toFixed(3), w - 6, h - 4);
}

function tbl(id, rows){ $(id).querySelector('tbody').innerHTML = rows.join(''); }

/* ------------------------------------------------------------ render */
let prev = {};
function render(st){
  const dot = $('dot');
  dot.className = 'dot' + (st.status === 'done' ? ' done' : st.status === 'running' || st.status === 'training' ? ' live' : '');
  $('phase').textContent = st.phase || st.status || 'idle';
  const e = st.elapsed_s || 0;
  $('clock').textContent = `${String(Math.floor(e / 60)).padStart(2,'0')}:${String(Math.floor(e % 60)).padStart(2,'0')}`;
  const H = st.history || [], last = H[H.length - 1];
  const pct = st.total_epochs ? Math.min(100, (st.epoch / st.total_epochs) * 100) : 0;
  $('prog').style.width = pct + '%';

  const set = (id, v, d) => {
    $(id).textContent = v;
    if (d !== undefined && $(d)) {
      const p = prev[id], nv = parseFloat(v);
      if (p !== undefined && !isNaN(nv)) {
        const diff = nv - p;
        $(d).textContent = (diff >= 0 ? '+' : '') + diff.toFixed(3);
        $(d).style.color = diff >= 0 ? C('--ok') : C('--bad');
      }
      prev[id] = nv;
    }
  };
  if (last){
    set('t_f1', last.val_macro_f1.toFixed(3), 'd_f1');
    set('t_acc', last.val_acc.toFixed(3), 'd_acc');
    set('t_bal', last.val_bal_acc.toFixed(3), 'd_bal');
    set('t_loss', last.loss.toFixed(3), 'd_loss');
  }
  $('t_best').textContent = (st.best?.val_macro_f1 ?? 0).toFixed(3);
  $('t_ep').textContent = st.epoch || 0;
  $('d_ep').textContent = 'of ' + (st.total_epochs || 0);
  $('msg').textContent = st.message || '';

  drawCurve(st); drawLayers(st); drawMod(st); drawClasses(st); drawCM(st); drawSteps(st);
  tbl('trials', (st.trials || []).map((t, i) =>
    `<tr><td>${t.cfg.arch} h${t.cfg.hidden} d${t.cfg.dropout} mix${t.cfg.mixup}</td>
     <td>${t.secs}s</td><td>${t.val_macro_f1.toFixed(4)}</td></tr>`));
  tbl('seeds', (st.seeds || []).map(s =>
    `<tr><td>seed ${s.seed}</td><td>val ${s.val_macro_f1.toFixed(4)}</td>
     <td>test ${s.test_macro_f1.toFixed(4)}</td></tr>`));
}

let STATE = {};
async function poll(){
  try{
    const r = await fetch('/api/train_state?t=' + Date.now());
    if (r.ok) STATE = await r.json();
  }catch(e){}
  render(STATE);
}
setInterval(poll, 1000);
poll();
/* keep the network animation smooth between polls */
setInterval(() => { if (Object.keys(STATE).length) drawNet(STATE); }, 40);
window.addEventListener('resize', () => render(STATE));
