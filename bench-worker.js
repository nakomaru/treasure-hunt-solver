// Persistent worker — mirror of index.html WORKER_SRC, adapted for worker_threads.
// Listens for multiple jobs; each job is a {state, slots, workerIdx, workerCount, seed}.
// Posts back {type:'progress',...} during the solve and {type:'done',...} when finished.
// Progress messages do NOT include accumAny (we don't need the heatmap for ETA work,
// which saves IPC bandwidth and matches what the predictor would actually consume).
'use strict';
const { parentPort } = require('worker_threads');
const { performance } = require('perf_hooks');

const W = 9, H = 5, NCELLS = 45;

function orientations(sh){ const p = sh.split('x'), a = +p[0], b = +p[1]; return a === b ? [[a,b]] : [[a,b],[b,a]]; }
function shapeArea(sh){ const p = sh.split('x'); return p[0]*p[1]; }
function placementsFor(sh, missLo, missHi){
  const seen = {}, out = [];
  for (const [w,h] of orientations(sh)){
    for (let r = 0; r <= H-h; r++) for (let c = 0; c <= W-w; c++){
      let lo = 0, hi = 0; const cs = [];
      for (let dr = 0; dr < h; dr++) for (let dc = 0; dc < w; dc++){
        const idx = (r+dr)*W + (c+dc); cs.push(idx);
        if (idx < 32) lo |= (1<<idx); else hi |= (1<<(idx-32));
      }
      const key = lo + '_' + hi;
      if (seen[key]) continue;
      seen[key] = 1;
      if ((lo & missLo) || (hi & missHi)) continue;
      out.push({ lo, hi, cells: cs });
    }
  }
  return out;
}
function mulberry32(a){ return function(){ a |= 0; a = a + 0x6D2B79F5 | 0; let t = Math.imul(a ^ a >>> 15, 1 | a); t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t; return ((t ^ t >>> 14) >>> 0) / 4294967296; }; }

function runJob(ev){
  const state = ev.state, slots = ev.slots;
  const workerIdx = ev.workerIdx | 0, workerCount = ev.workerCount | 0, seed = ev.seed | 0;
  const rnd = mulberry32(seed);
  let missLo = 0, missHi = 0, hitLo = 0, hitHi = 0;
  for (let i = 0; i < NCELLS; i++){
    const bL = i < 32 ? (1<<i) : 0, bH = i >= 32 ? (1<<(i-32)) : 0;
    if (state[i] === 1 || state[i] === 3){ missLo |= bL; missHi |= bH; }
    if (state[i] === 2){ hitLo |= bL; hitHi |= bH; }
  }
  const gm = {}, order = [];
  for (let i = 0; i < slots.length; i++){
    const s = slots[i]; if (s.count === 0) continue;
    if (gm[s.shape]) gm[s.shape].count += s.count;
    else { gm[s.shape] = { shape: s.shape, count: s.count }; order.push(gm[s.shape]); }
  }
  const groups = order.sort((a,b) => shapeArea(b.shape) - shapeArea(a.shape));
  const t0 = performance.now();
  let leaf = 0, nodes = 0;
  let branchCount = 0, branchNodesSum = 0, rangeLen = 0;
  function done(total){
    parentPort.postMessage({ type: 'done', total, leaf, ms: performance.now() - t0,
      workerIdx, nodes, branchCount, branchNodesSum, rangeLen });
  }
  if (groups.length === 0){ done(0); return; }
  for (const g of groups){
    g.placements = placementsFor(g.shape, missLo, missHi);
    for (let sh = g.placements.length - 1; sh > 0; sh--){
      const sj = (rnd() * (sh+1)) | 0, st = g.placements[sh];
      g.placements[sh] = g.placements[sj]; g.placements[sj] = st;
    }
    let oL = 0, oH = 0;
    for (const p of g.placements){ oL |= p.lo; oH |= p.hi; }
    g.orLo = oL; g.orHi = oH;
  }
  const inst = [];
  for (const g of groups) for (let k = 0; k < g.count; k++) inst.push(g);
  const N = inst.length;
  if (N === 0){ done(workerIdx === 0 ? 1 : 0); return; }
  for (const g of groups) if (g.placements.length === 0){ done(0); return; }
  const sameAsPrev = [], plList = [];
  for (let i = 0; i < N; i++){ sameAsPrev.push(i > 0 && inst[i-1] === inst[i]); plList.push(inst[i].placements); }
  const sufLo = new Array(N+1), sufHi = new Array(N+1);
  sufLo[N] = 0; sufHi[N] = 0;
  for (let i = N-1; i >= 0; i--){ sufLo[i] = sufLo[i+1] | inst[i].orLo; sufHi[i] = sufHi[i+1] | inst[i].orHi; }
  if ((hitLo & ~sufLo[0]) || (hitHi & ~sufHi[0])){ done(0); return; }
  const rootLen = plList[0].length;
  const rootStart = Math.floor(rootLen * workerIdx / workerCount);
  const rootEnd   = Math.floor(rootLen * (workerIdx+1) / workerCount);
  rangeLen = rootEnd - rootStart;
  if (rangeLen <= 0){ done(0); return; }

  let lastPost = t0;
  let rootIdx = rootStart, lvl2Len = 1, lvl2Idx = 0;
  let nodesAtBranchStart = 0;
  function rec(i, occLo, occHi, uncLo, uncHi, minIdx){
    if (i === N){ if (uncLo === 0 && uncHi === 0){ leaf++; return 1; } return 0; }
    if ((uncLo & ~sufLo[i]) || (uncHi & ~sufHi[i])) return 0;
    if ((++nodes & 131071) === 0){
      const now = performance.now();
      const threshold = (now - t0 < 3000) ? 50 : 500;
      if (now - lastPost > threshold){
        const pctBranch = (rootIdx - rootStart + lvl2Idx/lvl2Len) / rangeLen;
        parentPort.postMessage({ type: 'progress', leaf, pct: pctBranch,
          elapsed: now - t0, workerIdx, nodes, branchCount, branchNodesSum, rangeLen });
        lastPost = now;
      }
    }
    const pls = plList[i];
    const lo = i === 0 ? rootStart : (sameAsPrev[i] ? minIdx : 0);
    const hi = i === 0 ? rootEnd   : pls.length;
    let count = 0;
    for (let idx = lo; idx < hi; idx++){
      if (i === 0){
        if (idx > rootStart){
          branchNodesSum += (nodes - nodesAtBranchStart);
          branchCount++;
        }
        nodesAtBranchStart = nodes;
        rootIdx = idx;
      } else if (i === 1){ lvl2Idx = idx - lo; lvl2Len = hi - lo; }
      const p = pls[idx];
      if ((occLo & p.lo) | (occHi & p.hi)) continue;
      const sub = rec(i+1, occLo | p.lo, occHi | p.hi, uncLo & ~p.lo, uncHi & ~p.hi, idx+1);
      if (sub > 0) count += sub;
    }
    return count;
  }
  const total = rec(0, 0, 0, hitLo, hitHi, 0);
  if (rangeLen > 0){
    branchNodesSum += (nodes - nodesAtBranchStart);
    branchCount++;
  }
  parentPort.postMessage({ type: 'done', total, leaf, ms: performance.now() - t0,
    workerIdx, nodes, branchCount, branchNodesSum, rangeLen });
}

parentPort.on('message', ev => {
  if (ev && ev.type === 'shutdown'){ process.exit(0); }
  try { runJob(ev); }
  catch (e){ parentPort.postMessage({ type: 'error', message: String(e && e.stack || e) }); }
});
