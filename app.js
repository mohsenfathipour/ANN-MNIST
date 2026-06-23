// ── constants ──────────────────────────────────────────────────────────────
const GRID      = 28;
const CELL      = 10;
const SZ_HIDDEN = 100;
const SHOW_OUT  = 10;
const SHOW_HID  = 35;
const SHOW_INP  = 40;

const weightsHidden = window.MNIST_WEIGHTS.weightsHidden;
const weightsOutput = window.MNIST_WEIGHTS.weightsOutput;

// ── canvas refs (draw + processed only) ───────────────────────────────────
const drawCanvas      = document.getElementById('drawCanvas');
const drawCtx         = drawCanvas.getContext('2d');
const processedCanvas = document.getElementById('processedCanvas');
const processedCtx    = processedCanvas.getContext('2d');
const clearBtn        = document.getElementById('clearBtn');

// ── state ──────────────────────────────────────────────────────────────────
let pixels        = new Float32Array(GRID * GRID);
let layerHidden   = new Float32Array(SZ_HIDDEN + 1);
let layerOutput   = new Float32Array(10);
let currentPooled = null;
let bestIdx       = -1;
let isDrawing     = false;
let lastCell      = null;

// ── draw canvas ────────────────────────────────────────────────────────────
function renderDrawCanvas() {
  for (let y = 0; y < GRID; y++) {
    for (let x = 0; x < GRID; x++) {
      const v = pixels[y * GRID + x];
      const c = Math.round(255 - v * 255);
      drawCtx.fillStyle = `rgb(${c},${c},${c})`;
      drawCtx.fillRect(x * CELL, y * CELL, CELL, CELL);
    }
  }
  drawCtx.strokeStyle = 'rgba(40,40,65,0.4)';
  drawCtx.lineWidth = 0.5;
  for (let i = 0; i <= GRID; i++) {
    drawCtx.beginPath(); drawCtx.moveTo(i * CELL, 0); drawCtx.lineTo(i * CELL, GRID * CELL); drawCtx.stroke();
    drawCtx.beginPath(); drawCtx.moveTo(0, i * CELL); drawCtx.lineTo(GRID * CELL, i * CELL); drawCtx.stroke();
  }
}

function renderProcessedCanvas(pooledData) {
  if (!pooledData) { processedCtx.clearRect(0, 0, 14, 14); return; }
  for (let y = 0; y < 14; y++) {
    for (let x = 0; x < 14; x++) {
      const v = pooledData[y * 14 + x];
      const c = Math.round(v * 255);
      processedCtx.fillStyle = `rgb(${c},${c},${c})`;
      processedCtx.fillRect(x, y, 1, 1);
    }
  }
}

function paintAtCell(cx, cy) {
  const R = 1.3;
  for (let dy = -2; dy <= 2; dy++) {
    for (let dx = -2; dx <= 2; dx++) {
      const nx = cx + dx, ny = cy + dy;
      if (nx < 0 || nx >= GRID || ny < 0 || ny >= GRID) continue;
      const dist = Math.sqrt(dx * dx + dy * dy);
      const ink  = Math.max(0, 1 - dist / R);
      if (ink <= 0) continue;
      pixels[ny * GRID + nx] = Math.min(1, pixels[ny * GRID + nx] + ink * 0.72);
    }
  }
}

function paintLine(x0, y0, x1, y1) {
  const steps = Math.max(Math.abs(x1 - x0), Math.abs(y1 - y0), 1);
  for (let s = 0; s <= steps; s++) {
    const t = s / steps;
    paintAtCell(Math.round(x0 + t * (x1 - x0)), Math.round(y0 + t * (y1 - y0)));
  }
}

function getCell(e) {
  const rect   = drawCanvas.getBoundingClientRect();
  const scaleX = drawCanvas.width  / rect.width;
  const scaleY = drawCanvas.height / rect.height;
  const cx     = e.touches ? e.touches[0].clientX : e.clientX;
  const cy     = e.touches ? e.touches[0].clientY : e.clientY;
  return {
    x: Math.floor((cx - rect.left) * scaleX / CELL),
    y: Math.floor((cy - rect.top)  * scaleY / CELL)
  };
}

// ── events ─────────────────────────────────────────────────────────────────
drawCanvas.addEventListener('mousedown',  e => { e.preventDefault(); isDrawing = true; lastCell = getCell(e); paintAtCell(lastCell.x, lastCell.y); renderDrawCanvas(); });
drawCanvas.addEventListener('mousemove',  e => { if (!isDrawing) return; const c = getCell(e); if (lastCell) paintLine(lastCell.x, lastCell.y, c.x, c.y); lastCell = c; renderDrawCanvas(); });
window.addEventListener('mouseup',        () => { if (!isDrawing) return; isDrawing = false; lastCell = null; predictAndUpdate(); });
drawCanvas.addEventListener('touchstart', e => { e.preventDefault(); isDrawing = true; lastCell = getCell(e); paintAtCell(lastCell.x, lastCell.y); renderDrawCanvas(); }, { passive: false });
drawCanvas.addEventListener('touchmove',  e => { e.preventDefault(); if (!isDrawing) return; const c = getCell(e); if (lastCell) paintLine(lastCell.x, lastCell.y, c.x, c.y); lastCell = c; renderDrawCanvas(); }, { passive: false });
window.addEventListener('touchend',       () => { if (!isDrawing) return; isDrawing = false; lastCell = null; predictAndUpdate(); });

clearBtn.addEventListener('click', () => {
  pixels.fill(0); layerHidden.fill(0); layerOutput.fill(0);
  currentPooled = null; bestIdx = -1;
  renderDrawCanvas(); renderProcessedCanvas(null);
  resetNetSVG();
  updatePredPanel(); updateBars(false); updateStats();
});

// ── preprocessing ──────────────────────────────────────────────────────────
function cropDigit(input) {
  let left = 0, right = 27, top = 0, bottom = 27, found;
  for (let x = 0; x < 28; x++) { found = false; for (let y = 0; y < 28; y++) if (input[y*28+x] > 0.05) found = true; if (found) { left = x; break; } }
  for (let x = 27; x >= 0; x--) { found = false; for (let y = 0; y < 28; y++) if (input[y*28+x] > 0.05) found = true; if (found) { right = x; break; } }
  for (let y = 0; y < 28; y++) { found = false; for (let x = 0; x < 28; x++) if (input[y*28+x] > 0.05) found = true; if (found) { top = y; break; } }
  for (let y = 27; y >= 0; y--) { found = false; for (let x = 0; x < 28; x++) if (input[y*28+x] > 0.05) found = true; if (found) { bottom = y; break; } }

  const h = right - left + 1, w = bottom - top + 1, norm = 20.0;
  let s = norm / h; if (norm / w < s) s = norm / w;
  const w2 = Math.floor(w*s), h2 = Math.floor(h*s);
  const w3 = Math.round((28-w2)/2), h3 = Math.round((28-h2)/2);
  const out = new Float32Array(784);
  for (let j = 0; j < w2; j++)
    for (let i = 0; i < h2; i++) {
      const x = Math.floor(i/s)+left, y = Math.floor(j/s)+top;
      out[(j+w3)*28+(i+h3)] = input[x+y*28];
    }
  return out;
}

function avgPool2x2(input) {
  const p = new Float32Array(197);
  for (let i = 0; i < 14; i++)
    for (let j = 0; j < 14; j++)
      p[i*14+j] = (input[(2*i)*28+(2*j)] + input[(2*i)*28+(2*j+1)] +
                   input[(2*i+1)*28+(2*j)] + input[(2*i+1)*28+(2*j+1)]) / 4;
  p[196] = 1.0;
  return p;
}

// ── predict ────────────────────────────────────────────────────────────────
function predictAndUpdate() {
  const totalInk = pixels.reduce((a, b) => a + b, 0);
  if (totalInk < 2) {
    layerHidden.fill(0); layerOutput.fill(0); currentPooled = null; bestIdx = -1;
    renderProcessedCanvas(null);
    resetNetSVG();
    updatePredPanel(); updateBars(false); updateStats();
    return;
  }

  const centered = cropDigit(pixels);
  const pooled   = avgPool2x2(centered);
  currentPooled  = pooled;
  renderProcessedCanvas(pooled);

  // Hidden layer — ReLU
  for (let i = 0; i < SZ_HIDDEN; i++) {
    let sum = 0;
    for (let j = 0; j < 197; j++) sum += pooled[j] * weightsHidden[i*197+j];
    layerHidden[i] = sum > 0 ? sum : 0;
  }
  layerHidden[SZ_HIDDEN] = 1.0;

  // Output layer — Softmax
  let esum = 0;
  for (let i = 0; i < 10; i++) {
    let sum = 0;
    for (let j = 0; j <= SZ_HIDDEN; j++) sum += layerHidden[j] * weightsOutput[i*101+j];
    layerOutput[i] = Math.exp(sum);
    esum += layerOutput[i];
  }
  for (let i = 0; i < 10; i++) layerOutput[i] /= esum;

  bestIdx = 0;
  for (let i = 1; i < 10; i++) if (layerOutput[i] > layerOutput[bestIdx]) bestIdx = i;

  animateForwardPass();
  updatePredPanel();
  updateBars(true);
  updateStats();
}

// ── UI helpers ─────────────────────────────────────────────────────────────
function updatePredPanel() {
  const badge  = document.getElementById('pred-badge');
  const confEl = document.getElementById('pred-conf');
  const descEl = document.getElementById('pred-desc');
  if (!badge) return;

  if (bestIdx < 0) {
    badge.textContent = '?';
    badge.classList.add('empty');
    confEl.textContent = 'Awaiting input';
    descEl.textContent = 'Draw a digit on the canvas to trigger forward propagation.';
    return;
  }

  const pct = Math.round(layerOutput[bestIdx] * 100);
  badge.textContent = bestIdx;
  badge.classList.remove('empty');
  confEl.innerHTML = `Digit <em>${bestIdx}</em> · ${pct}% confidence`;
  descEl.textContent = `Output node ${bestIdx} received the highest Softmax probability during forward pass.`;
}

function initBars() {
  const list = document.getElementById('bars-list');
  if (!list) return;
  list.innerHTML = '';
  for (let i = 0; i < 10; i++) {
    list.insertAdjacentHTML('beforeend', `
      <div class="bar-row" id="bar-row-${i}">
        <span class="bar-digit">${i}</span>
        <div class="bar-track"><div class="bar-fill" id="bar-fill-${i}"></div></div>
        <span class="bar-pct" id="bar-pct-${i}">0%</span>
      </div>
    `);
  }
}

function updateBars(active) {
  const empty = document.getElementById('bars-empty');
  const list  = document.getElementById('bars-list');
  if (!list) return;

  if (!active) { if (empty) empty.style.display = ''; list.style.display = 'none'; return; }
  if (empty) empty.style.display = 'none';
  list.style.display = '';

  for (let i = 0; i < 10; i++) {
    const pct  = (layerOutput[i] || 0) * 100;
    const row  = document.getElementById(`bar-row-${i}`);
    const fill = document.getElementById(`bar-fill-${i}`);
    const lbl  = document.getElementById(`bar-pct-${i}`);
    if (row)  row.classList.toggle('winner', i === bestIdx);
    if (fill) fill.style.width = pct.toFixed(1) + '%';
    if (lbl)  lbl.textContent  = pct < 0.5 ? '<1%' : Math.round(pct) + '%';
  }
}

function updateStats() {
  const activeEl = document.getElementById('stat-active');
  const maxEl    = document.getElementById('stat-maxact');
  const confEl   = document.getElementById('stat-conf');

  if (bestIdx < 0) { [activeEl, maxEl, confEl].forEach(el => { if (el) el.textContent = '—'; }); return; }

  const hiddenArr = Array.from(layerHidden.slice(0, SZ_HIDDEN));
  const active    = hiddenArr.filter(v => v > 0).length;
  const maxAct    = Math.max(...hiddenArr);
  const conf      = Math.round(layerOutput[bestIdx] * 100);

  if (activeEl) activeEl.textContent = `${active}/100`;
  if (maxEl)    maxEl.textContent    = maxAct.toFixed(2);
  if (confEl)   confEl.textContent   = `${conf}%`;
}

// ── SVG network visualization ──────────────────────────────────────────────
const SVG_NS = 'http://www.w3.org/2000/svg';
const NET_W  = 580;
const NET_H  = 340;
const N_PAD  = 35;
const N_UW   = NET_W - N_PAD * 2;
const Y_INP  = 68;
const Y_HID  = 190;
const Y_OUT  = 310;
const R_INP  = 4.5;
const R_HID  = 5.5;
const R_OUT  = 11;
const DIM_OP   = 0.12;
const DIM_FILL = '#b8b8d8'; // dim node fill for light background

function xOf(i, n) { return N_PAD + (i + 0.5) * (N_UW / n); }

function svgEl(tag, attrs = {}) {
  const el = document.createElementNS(SVG_NS, tag);
  Object.entries(attrs).forEach(([k, v]) => el.setAttribute(k, v));
  return el;
}

// Element registries
let inputNodeEls   = [];
let hiddenNodeEls  = [];
let outputNodeEls  = [];
let outputGlowEls  = [];
let outputLabelEls = [];
let ihLineData     = []; // { el, maxOp }
let hoLineData     = []; // { el, maxOp }

function initNetSVG() {
  const svg = document.getElementById('netSvg');
  svg.innerHTML = '';
  inputNodeEls = []; hiddenNodeEls = []; outputNodeEls = [];
  outputGlowEls = []; outputLabelEls = []; ihLineData = []; hoLineData = [];

  // ── Filters ──────────────────────────────────────────────────────────────
  const defs = svg.appendChild(svgEl('defs'));

  // Glow filter for active nodes
  const fGlow = defs.appendChild(svgEl('filter', { id: 'glow', x: '-60%', y: '-60%', width: '220%', height: '220%' }));
  fGlow.appendChild(svgEl('feGaussianBlur', { stdDeviation: '5', result: 'blur' }));
  const fm1 = fGlow.appendChild(svgEl('feMerge'));
  fm1.appendChild(svgEl('feMergeNode', { in: 'blur' }));
  fm1.appendChild(svgEl('feMergeNode', { in: 'SourceGraphic' }));

  // Soft glow for connections during flash
  const fConnGlow = defs.appendChild(svgEl('filter', { id: 'conn-glow', x: '-20%', y: '-400%', width: '140%', height: '900%' }));
  fConnGlow.appendChild(svgEl('feGaussianBlur', { stdDeviation: '1.5', result: 'blur' }));
  const fm2 = fConnGlow.appendChild(svgEl('feMerge'));
  fm2.appendChild(svgEl('feMergeNode', { in: 'blur' }));
  fm2.appendChild(svgEl('feMergeNode', { in: 'SourceGraphic' }));

  // ── Layer Groups (z-order: connections → nodes → labels) ─────────────────
  const gIhConn  = svg.appendChild(svgEl('g', { id: 'g-conn-ih' }));
  const gHoConn  = svg.appendChild(svgEl('g', { id: 'g-conn-ho' }));
  const gInp     = svg.appendChild(svgEl('g', { id: 'g-inp' }));
  const gHid     = svg.appendChild(svgEl('g', { id: 'g-hid' }));
  const gOutGlow = svg.appendChild(svgEl('g', { id: 'g-out-glow' }));
  const gOut     = svg.appendChild(svgEl('g', { id: 'g-out' }));
  const gLabels  = svg.appendChild(svgEl('g', { id: 'g-labels' }));

  // ── Layer Labels ──────────────────────────────────────────────────────────
  const MONO = "'JetBrains Mono', monospace";
  function addLayerLabel(text, y) {
    const t = svgEl('text', { x: N_PAD, y: y - 24, 'font-family': MONO, 'font-size': '8', fill: '#6060a0' });
    t.textContent = text;
    gLabels.appendChild(t);
  }
  addLayerLabel('INPUT LAYER  ·  14×14 avg-pool  →  196 nodes + bias', Y_INP);
  addLayerLabel('HIDDEN LAYER  ·  100 neurons  ·  ReLU: f(x) = max(0, x)', Y_HID);
  addLayerLabel('OUTPUT LAYER  ·  10 classes  ·  Softmax: σ(zᵢ) = eᶻⁱ / Σeᶻʲ', Y_OUT);

  // ── Connections: Input → Hidden ───────────────────────────────────────────
  for (let h = 0; h < SHOW_HID; h++) {
    for (let inp = 0; inp < SHOW_INP; inp++) {
      const hIdx = Math.floor(h   * SZ_HIDDEN / SHOW_HID);
      const iIdx = Math.floor(inp * 196       / SHOW_INP);
      const w    = weightsHidden[hIdx * 197 + iIdx];
      if (Math.abs(w) < 0.12) continue;

      const line = svgEl('line', {
        x1: xOf(inp, SHOW_INP), y1: Y_INP + R_INP + 1,
        x2: xOf(h,   SHOW_HID), y2: Y_HID - R_HID - 1,
        stroke: w > 0 ? '#a855f7' : '#10b981',
        'stroke-width': 0.5,
        opacity: 0
      });
      gIhConn.appendChild(line);
      ihLineData.push({ el: line, maxOp: Math.min(Math.abs(w) * 0.85, 0.5) });
    }
  }

  // ── Connections: Hidden → Output ──────────────────────────────────────────
  for (let o = 0; o < SHOW_OUT; o++) {
    for (let h = 0; h < SHOW_HID; h++) {
      const hIdx = Math.floor(h * SZ_HIDDEN / SHOW_HID);
      const w    = weightsOutput[o * 101 + hIdx];
      if (Math.abs(w) < 0.09) continue;

      const line = svgEl('line', {
        x1: xOf(h, SHOW_HID), y1: Y_HID + R_HID + 1,
        x2: xOf(o, SHOW_OUT), y2: Y_OUT - R_OUT - 1,
        stroke: w > 0 ? '#a855f7' : '#10b981',
        'stroke-width': 0.75,
        opacity: 0
      });
      gHoConn.appendChild(line);
      hoLineData.push({ el: line, maxOp: Math.min(Math.abs(w) * 1.1, 0.65) });
    }
  }

  // ── Input Nodes ───────────────────────────────────────────────────────────
  for (let i = 0; i < SHOW_INP; i++) {
    const circle = svgEl('circle', { cx: xOf(i, SHOW_INP), cy: Y_INP, r: R_INP, fill: DIM_FILL, opacity: DIM_OP });
    gInp.appendChild(circle);
    inputNodeEls.push(circle);
  }

  // ── Hidden Nodes ──────────────────────────────────────────────────────────
  for (let i = 0; i < SHOW_HID; i++) {
    const circle = svgEl('circle', { cx: xOf(i, SHOW_HID), cy: Y_HID, r: R_HID, fill: DIM_FILL, opacity: DIM_OP });
    gHid.appendChild(circle);
    hiddenNodeEls.push(circle);
  }

  // ── Output Glow Halos + Nodes + Digit Labels ──────────────────────────────
  for (let i = 0; i < SHOW_OUT; i++) {
    const cx = xOf(i, SHOW_OUT);

    const glow = svgEl('circle', { cx, cy: Y_OUT, r: R_OUT + 10, fill: '#6366f1', opacity: 0 });
    gOutGlow.appendChild(glow);
    outputGlowEls.push(glow);

    const circle = svgEl('circle', { cx, cy: Y_OUT, r: R_OUT, fill: DIM_FILL, opacity: DIM_OP });
    gOut.appendChild(circle);
    outputNodeEls.push(circle);

    const label = svgEl('text', {
      x: cx, y: Y_OUT + R_OUT + 16,
      'text-anchor': 'middle',
      'font-family': MONO,
      'font-size': '10',
      fill: '#5a5a80'
    });
    label.textContent = String(i);
    gLabels.appendChild(label);
    outputLabelEls.push(label);
  }
}

// ── compute target fill + opacity from activations ─────────────────────────
function computeTargets() {
  inputNodeEls.forEach((el, i) => {
    const iIdx = Math.floor(i * 196 / SHOW_INP);
    const v    = currentPooled ? currentPooled[iIdx] : 0;
    const active = v > 0.01;
    el._fill = active ? '#6366f1' : DIM_FILL;
    el._op   = active ? Math.min(v * 1.7, 0.95) : DIM_OP;
  });

  hiddenNodeEls.forEach((el, i) => {
    const hIdx = Math.floor(i * SZ_HIDDEN / SHOW_HID);
    const v    = layerHidden[hIdx] || 0;
    const active = v > 0.05;
    el._fill = active ? '#a855f7' : DIM_FILL;
    el._op   = active ? Math.min(v / 2.5, 0.92) : DIM_OP;
  });

  outputNodeEls.forEach((el, i) => {
    const v      = layerOutput[i] || 0;
    el._fill     = '#6366f1';
    el._op       = i === bestIdx ? 1.0 : Math.max(Math.min(v * 5, 0.5), DIM_OP);
  });
}

// ── GSAP forward-pass animation ────────────────────────────────────────────
function animateForwardPass() {
  computeTargets();

  const ihEls = ihLineData.map(d => d.el);
  const hoEls = hoLineData.map(d => d.el);
  const tl    = gsap.timeline({ defaults: { ease: 'power2.out' } });

  // ── 0. Reset to dim/hidden ──────────────────────────────────────────────
  tl.set(inputNodeEls,   { opacity: DIM_OP, attr: { fill: DIM_FILL } })
    .set(hiddenNodeEls,  { opacity: DIM_OP, attr: { fill: DIM_FILL } })
    .set(outputNodeEls,  { opacity: DIM_OP, attr: { fill: DIM_FILL, r: R_OUT } })
    .set(outputGlowEls,  { opacity: 0, attr: { r: R_OUT + 10 } })
    .set(ihEls, { opacity: 0 })
    .set(hoEls, { opacity: 0 });

  // ── 1. Input layer activates (left → right wave) ─────────────────────────
  tl.to(inputNodeEls, {
    opacity: (i, el) => el._op,
    attr: { fill: (i, el) => el._fill },
    duration: 0.5,
    stagger: { amount: 0.4, from: 'start' }
  });

  // ── 2. Input→Hidden connections: bright flash then settle ─────────────────
  tl.to(ihEls, {
    opacity: 0.75,
    duration: 0.22,
    stagger: { amount: 0.14, from: 'start' }
  }, '+=0.06')
  .to(ihEls, {
    opacity: (i) => ihLineData[i].maxOp,
    duration: 0.55
  });

  // ── 3. Hidden layer activates (left → right wave) ─────────────────────────
  tl.to(hiddenNodeEls, {
    opacity: (i, el) => el._op,
    attr: { fill: (i, el) => el._fill },
    duration: 0.45,
    stagger: { amount: 0.3, from: 'start' }
  }, '-=0.35');

  // ── 4. Hidden→Output connections: bright flash then settle ────────────────
  tl.to(hoEls, {
    opacity: 0.8,
    duration: 0.2,
    stagger: { amount: 0.1, from: 'start' }
  }, '+=0.05')
  .to(hoEls, {
    opacity: (i) => hoLineData[i].maxOp,
    duration: 0.5
  });

  // ── 5. Output layer activates ────────────────────────────────────────────
  tl.to(outputNodeEls, {
    opacity: (i, el) => el._op,
    attr: { fill: (i, el) => el._fill },
    duration: 0.35,
    stagger: 0.04
  }, '-=0.3');

  // ── 6. Winner: glow halo + node bounce + label highlight ─────────────────
  const wNode  = outputNodeEls[bestIdx];
  const wGlow  = outputGlowEls[bestIdx];
  const wLabel = outputLabelEls[bestIdx];

  // Glow halo expands then contracts
  tl.to(wGlow, { opacity: 0.22, attr: { r: R_OUT + 16 }, duration: 0.3, ease: 'power2.out' }, '+=0.02')
    .to(wGlow, { opacity: 0.1,  attr: { r: R_OUT + 8  }, duration: 0.8, ease: 'elastic.out(1, 0.4)' });

  // Node bounces
  tl.to(wNode, { attr: { r: R_OUT + 5 }, duration: 0.28, ease: 'back.out(2.5)' }, '<-=0.5')
    .to(wNode, { attr: { r: R_OUT + 1 }, duration: 0.65, ease: 'elastic.out(1, 0.5)' });

  // Label turns dark + bold (immediate, via callback)
  tl.call(() => {
    wLabel.setAttribute('fill', '#1a1a30');
    wLabel.setAttribute('font-weight', '700');
    wLabel.setAttribute('font-size', '11');
  }, null, '<-=0.7');
}

// ── reset SVG to idle state ────────────────────────────────────────────────
function resetNetSVG() {
  if (!inputNodeEls.length) return;

  const allNodes = [...inputNodeEls, ...hiddenNodeEls, ...outputNodeEls];
  const allConns = [...ihLineData.map(d => d.el), ...hoLineData.map(d => d.el)];

  gsap.killTweensOf([...allNodes, ...allConns, ...outputGlowEls, ...outputNodeEls]);

  gsap.to(allNodes,      { opacity: DIM_OP, attr: { fill: DIM_FILL }, duration: 0.4 });
  gsap.to(allConns,      { opacity: 0, duration: 0.35 });
  gsap.to(outputGlowEls, { opacity: 0, duration: 0.3 });
  gsap.to(outputNodeEls, { attr: { r: R_OUT }, duration: 0.4 });

  outputLabelEls.forEach(el => {
    el.setAttribute('fill', '#5a5a80');
    el.setAttribute('font-weight', '400');
    el.setAttribute('font-size', '10');
  });
}

// ── init ───────────────────────────────────────────────────────────────────
initBars();
renderDrawCanvas();
initNetSVG();
