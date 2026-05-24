// ETA-prediction benchmark for the treasure-hunt solver.
//
// Drives the real index.html engine (pooled worker_threads, partition-by-root-branch,
// same heuristics, same progress cadence) over a wide variety of randomly generated
// boards, records per-solve features + full progress trajectory + final wall time,
// then evaluates a battery of "prior" (predict before solve) and "live" (predict
// during solve) ETA strategies against each other.
//
// Usage:  node bench.js [budget_minutes]
//   Defaults to 30 min. Long solves are not aborted — if a board is going to take
//   8 minutes, we let it run. Budget check happens between boards.
//
// Outputs:
//   bench-results.jsonl   one JSON record per finished solve (streamed; safe to tail
//                         while running, and the report step can be re-run alone).
//   bench-report.txt      final ranked tables.
//
// To re-analyze without rerunning the solves, pass --report-only:
//   node bench.js --report-only
'use strict';

const { Worker } = require('worker_threads');
const { performance } = require('perf_hooks');
const fs = require('fs');
const path = require('path');
const os = require('os');

const W = 9, H = 5, NCELLS = 45;
const SHAPES = ['1x2','1x3','1x4','2x2','2x3','2x4','3x3'];
const POOL_SIZE = Math.max(1, Math.min(16, (os.cpus().length || 4) - 1));
const RESULTS_PATH = path.join(__dirname, 'bench-results.jsonl');
const REPORT_PATH  = path.join(__dirname, 'bench-report.txt');
const WORKER_FILE  = path.join(__dirname, 'bench-worker.js');

const REPORT_ONLY = process.argv.includes('--report-only');
const SHRINK_ONLY = process.argv.includes('--shrink');
const SHRINK_BENCH = process.argv.includes('--shrink-bench');
const FULL_SWEEP   = process.argv.includes('--full-sweep');
const FEATURE_AUDIT = process.argv.includes('--feature-audit');
// Bayesian-shrinkage pseudocounts for the binned tables. `α=0` recovers the
// raw cell mean (and hard-cutoff fallback when n < threshold); large α pulls
// every cell toward the pooled-global estimate. `α=20` ≈ "need 20 samples to
// outweigh the global prior." See BENCH_MEMO.md §"Bayesian shrinkage".
function parseAlpha(flag, dflt){
  const a = process.argv.find(s => s.startsWith(flag + '='));
  return a ? parseFloat(a.slice(flag.length + 1)) : dflt;
}
const ALPHA_DEFAULT = parseAlpha('--alpha', 20);
const ALPHA_BIAS  = parseAlpha('--alpha-bias',  ALPHA_DEFAULT);
const ALPHA_VAR   = parseAlpha('--alpha-var',   ALPHA_DEFAULT);
const ALPHA_CALIB = parseAlpha('--alpha-calib', ALPHA_DEFAULT);
// Minimum finalMs for a solve to count in the final EVALUATION. The user only
// cares about predictions for solves that actually take a noticeable amount of
// time. Default 3s. Override with --min-eval-ms=N. Solves below this threshold
// still inform the predictors (they're useful training data) but they don't
// contribute to the scoring metrics. Override:
//   node bench.js --report-only --min-eval-ms=5000
const EVAL_MIN_MS = (() => {
  const a = process.argv.find(s => s.startsWith('--min-eval-ms='));
  return a ? parseFloat(a.slice('--min-eval-ms='.length)) : 3000;
})();
const BUDGET_MIN = (() => {
  for (const a of process.argv.slice(2)) {
    const n = parseFloat(a);
    if (!isNaN(n) && n > 0) return n;
  }
  return 30;
})();
const BUDGET_MS = BUDGET_MIN * 60 * 1000;

// ----------------------------------------------------------------------------
// Feature computation — superset of what's in index.html, plus a couple extras.
// ----------------------------------------------------------------------------
function orientations(sh){ const p = sh.split('x'), a = +p[0], b = +p[1]; return a === b ? [[a,b]] : [[a,b],[b,a]]; }
function shapeArea(sh){ const p = sh.split('x'); return p[0]*p[1]; }

function placementCountFor(sh, missLo, missHi){
  let n = 0; const seen = Object.create(null);
  for (const [w, h] of orientations(sh)){
    for (let r = 0; r <= H - h; r++)
      for (let c = 0; c <= W - w; c++){
        let lo = 0, hi = 0;
        for (let dr = 0; dr < h; dr++)
          for (let dc = 0; dc < w; dc++){
            const idx = (r+dr)*W + (c+dc);
            if (idx < 32) lo |= (1 << idx); else hi |= (1 << (idx-32));
          }
        const key = lo + '_' + hi;
        if (seen[key]) continue;
        seen[key] = 1;
        if ((lo & missLo) || (hi & missHi)) continue;
        n++;
      }
  }
  return n;
}

function logFactorial(k){ let s = 0; for (let i = 2; i <= k; i++) s += Math.log(i); return s; }

// Count connected components in the OPEN region (4-adjacent) and the bounding-box
// area of open cells. Compact / single-component boards prune differently than
// scattered ones. Cheap to compute over 45 cells.
function openGeometry(stateArr){
  const open = stateArr.map(s => s === 0 || s === 2);   // hit cells are still "open" for the solver
  const seen = new Array(NCELLS).fill(false);
  let comps = 0, minR = H, maxR = -1, minC = W, maxC = -1, nOpen = 0;
  for (let i = 0; i < NCELLS; i++){
    if (!open[i]) continue;
    nOpen++;
    const r = (i / W) | 0, c = i % W;
    if (r < minR) minR = r; if (r > maxR) maxR = r;
    if (c < minC) minC = c; if (c > maxC) maxC = c;
    if (seen[i]) continue;
    comps++;
    // BFS
    const stack = [i]; seen[i] = true;
    while (stack.length){
      const j = stack.pop(); const jr = (j / W) | 0, jc = j % W;
      for (const [dr, dc] of [[-1,0],[1,0],[0,-1],[0,1]]){
        const nr = jr + dr, nc = jc + dc;
        if (nr < 0 || nr >= H || nc < 0 || nc >= W) continue;
        const k = nr * W + nc;
        if (open[k] && !seen[k]){ seen[k] = true; stack.push(k); }
      }
    }
  }
  const bboxArea = nOpen > 0 ? (maxR - minR + 1) * (maxC - minC + 1) : 0;
  return { components: nOpen > 0 ? comps : 1, bboxArea };
}

function featuresFor(stateArr, slotArr){
  let missLo = 0, missHi = 0, nMiss = 0, nHit = 0, nOpen = 0;
  for (let i = 0; i < NCELLS; i++){
    const s = stateArr[i];
    const bL = i < 32 ? (1 << i) : 0, bH = i >= 32 ? (1 << (i-32)) : 0;
    if (s === 1 || s === 3){ missLo |= bL; missHi |= bH; nMiss++; }
    else if (s === 2){ nHit++; }
    else { nOpen++; }
  }
  const grouped = {};
  for (const s of slotArr){
    if (s.count <= 0) continue;
    grouped[s.shape] = (grouped[s.shape] || 0) + s.count;
  }
  let logProd = 0, sumPlacements = 0, nPrizes = 0, prizeArea = 0, infeasible = false;
  let topArea = -1, topPlacements = 0, topShapeCount = 0;
  let minPlacements = Infinity, maxPlacements = 0;
  let sumAreaSq = 0;
  let logFactSum = 0;                                // Σ log(kᵢ!) — sameAsPrev duplicate reduction
  const perShape = [];
  for (const sh of Object.keys(grouped)){
    const k = grouped[sh];
    const np = placementCountFor(sh, missLo, missHi);
    if (np <= 0){ infeasible = true; break; }
    const area = shapeArea(sh);
    if (area > topArea){ topArea = area; topPlacements = np; topShapeCount = k; }
    prizeArea += k * area;
    sumPlacements += k * np;
    logProd += k * Math.log(np);
    logFactSum += logFactorial(k);
    nPrizes += k;
    minPlacements = Math.min(minPlacements, np);
    maxPlacements = Math.max(maxPlacements, np);
    sumAreaSq += k * area * area;
    perShape.push({ sh, k, np, area });
  }
  if (infeasible) return { infeasible: true };
  const logChoose = (n, k) => {
    if (k < 0 || k > n) return -Infinity;
    let s = 0;
    for (let i = 1; i <= k; i++) s += Math.log(n - k + i) - Math.log(i);
    return s;
  };
  const meanArea = nPrizes ? prizeArea / nPrizes : 0;
  const varArea = nPrizes ? sumAreaSq / nPrizes - meanArea * meanArea : 0;
  const geom = openGeometry(stateArr);
  return {
    infeasible: false,
    logProd,                                              // Σ kᵢ·log(|placementsᵢ|)  — current main signal
    logSum: Math.log(Math.max(1, sumPlacements)),         // log(Σ kᵢ·|placementsᵢ|)
    logTopRange: topPlacements > 0 ? Math.log(topPlacements) : 0,  // current "totalRange"
    logMinPl: minPlacements < Infinity ? Math.log(minPlacements) : 0,
    logMaxPl: maxPlacements > 0 ? Math.log(maxPlacements) : 0,
    logChooseFree: logChoose(nOpen, prizeArea),           // log C(open, area)
    nPrizes,                                              // sum of all counts
    nOpen, nMiss, nHit,
    prizeArea,
    slack: nOpen - prizeArea,
    density: nOpen > 0 ? prizeArea / nOpen : 0,
    topArea, topShapeCount,
    nDistinctShapes: Object.keys(grouped).length,
    varArea,
    // logProd-style with the SPLIT into largest-shape vs rest (largest is the
    // recursion root, so its contribution to time scales differently than the rest)
    logProdTop: topPlacements > 0 ? topShapeCount * Math.log(topPlacements) : 0,
    logProdRest: 0,                                       // filled below
    // NEW: duplicate-corrected combinatoric size. The solver's sameAsPrev
    // optimization makes each extra copy of an identical shape cost ~k!
    // less, so the "effective" search space is logProd − Σ log(kᵢ!).
    logFactSum,
    logProdAdj: logProd - logFactSum,
    // NEW: geometric mean placements per piece — the effective branching factor
    // of the recursion, independent of total piece count
    effBranching: nPrizes > 0 ? logProd / nPrizes : 0,
    // NEW: open-region geometry. Disconnected regions / cramped bbox = more pruning
    openComponents: geom.components,
    logBboxArea: geom.bboxArea > 0 ? Math.log(geom.bboxArea) : 0,
    bboxFill: geom.bboxArea > 0 ? nOpen / geom.bboxArea : 0,
  };
}
// fill logProdRest = logProd - logProdTop
function patchFeatures(f){ if (!f.infeasible) f.logProdRest = f.logProd - f.logProdTop; return f; }

const FEATURE_KEYS = [
  'logProd', 'logSum', 'logTopRange', 'logMinPl', 'logMaxPl', 'logChooseFree',
  'nPrizes', 'nOpen', 'nMiss', 'nHit', 'prizeArea', 'slack', 'density',
  'topArea', 'topShapeCount', 'nDistinctShapes', 'varArea',
  'logProdTop', 'logProdRest',
  // new features
  'logFactSum', 'logProdAdj', 'effBranching',
  'openComponents', 'logBboxArea', 'bboxFill',
];

// Lean shipping set: top-10 by standardized-ridge |coef|. Matches full-25 LOO
// absLog to 3 decimals (0.1350 vs 0.1347) but lets ridge fit from ~12 user
// solves instead of 27. The 4 zero-variance features (nHit, nDistinctShapes,
// openComponents, logBboxArea) and a long tail of marginal-or-noisy features
// are excluded.
const FEATURE_KEYS_LEAN = [
  'logProdAdj', 'logProd', 'logSum', 'logFactSum', 'topShapeCount',
  'density', 'logChooseFree', 'prizeArea', 'slack', 'logProdRest',
];

// ----------------------------------------------------------------------------
// Random board generator — broader distribution than the app's randomSlots()
// so we sweep the full difficulty range from <100ms to multi-minute.
// ----------------------------------------------------------------------------
function randInt(n){ return Math.floor(Math.random() * n); }
function randSlots(){
  // Always 3 distinct shapes (matches the app), total piece count in [5..9]
  // (per the user: that's the regime they actually solve in).
  const shapes = [...SHAPES].sort(() => Math.random() - 0.5).slice(0, 3);
  const targetPieces = 5 + randInt(5);   // 5..9
  // distribute targetPieces across 3 shapes, each at least 1
  const counts = [1, 1, 1];
  for (let extra = targetPieces - 3; extra > 0; extra--) counts[randInt(3)]++;
  // cap any single count at 5 (the app's max)
  for (let i = 0; i < 3; i++) if (counts[i] > 5){ const over = counts[i] - 5; counts[i] = 5; counts[(i+1)%3] += over; }
  return shapes.map((sh, i) => ({ shape: sh, count: counts[i] }));
}
function randState(){
  // Mostly empty (matches the user's first solve when a new board appears),
  // with mid-game states (some misses) the remainder.
  const state = new Int8Array(NCELLS);
  const r = Math.random();
  if (r < 0.65) return state;        // 65% empty — that's where the long solves live
  // mid-game: 1..12 misses, geometric-ish
  const nMiss = 1 + Math.floor(Math.pow(Math.random(), 2) * 12);
  const idxs = [...Array(NCELLS).keys()].sort(() => Math.random() - 0.5);
  for (let i = 0; i < nMiss; i++) state[idxs[i]] = 1;
  return state;
}
function genBoard(){
  for (let attempt = 0; attempt < 200; attempt++){
    const slots = randSlots();
    if (!slots) continue;
    // total area must fit on the board with some slack
    const totalArea = slots.reduce((a,s) => a + s.count * shapeArea(s.shape), 0);
    if (totalArea > 38) continue;
    const state = randState();
    const stateArr = Array.from(state);
    const feats = patchFeatures(featuresFor(stateArr, slots));
    if (feats.infeasible) continue;
    if (feats.logProd > 55) continue;          // skip likely-multi-hour solves
    return { slots, state: stateArr, features: feats };
  }
  return null;
}

// ----------------------------------------------------------------------------
// Worker pool — persistent. One Worker per slot; we post jobs serially (one
// board at a time, all POOL_SIZE workers participate, like the app).
// ----------------------------------------------------------------------------
class Pool {
  constructor(size){
    this.workers = [];
    this.size = size;
    for (let k = 0; k < size; k++){
      const w = new Worker(WORKER_FILE);
      this.workers.push(w);
    }
  }
  shutdown(){ for (const w of this.workers) w.postMessage({ type: 'shutdown' }); }
  solve(board){
    const { state, slots } = board;
    const seed = (Math.random() * 0x7fffffff) | 0;
    const startedAt = performance.now();
    const pcts = new Array(this.size).fill(0);
    const wNodes = new Array(this.size).fill(0);
    const wBranchCount = new Array(this.size).fill(0);
    const wBranchNodesSum = new Array(this.size).fill(0);
    const wRangeLen = new Array(this.size).fill(0);
    const wMs = new Array(this.size).fill(0);
    const wTotal = new Array(this.size).fill(0);
    const wLeaf = new Array(this.size).fill(0);
    const trajectory = [];
    let finished = 0;
    const handlers = new Array(this.size);
    return new Promise((resolve, reject) => {
      for (let k = 0; k < this.size; k++){
        const handler = (d) => {
          if (d.type === 'error'){ reject(new Error(d.message)); return; }
          wNodes[k] = d.nodes || 0;
          wBranchCount[k] = d.branchCount || 0;
          wBranchNodesSum[k] = d.branchNodesSum || 0;
          wRangeLen[k] = d.rangeLen || 0;
          if (d.type === 'progress'){
            pcts[k] = d.pct;
            const elapsed = performance.now() - startedAt;
            const nps = [];
            for (let j = 0; j < this.size; j++){
              const bc = wBranchCount[j], rl = wRangeLen[j];
              if (bc < 1 || rl <= 0){ nps.push(0); continue; }
              const avg = wBranchNodesSum[j] / bc;
              const est = avg * rl;
              if (est <= 0){ nps.push(0); continue; }
              nps.push(Math.max(0, Math.min(0.9999, wNodes[j] / est)));
            }
            trajectory.push({ e: elapsed, pcts: pcts.slice(), nps });
          } else {   // done
            pcts[k] = 1;
            wMs[k] = d.ms;
            wTotal[k] = d.total;
            wLeaf[k] = d.leaf;
            finished++;
            if (finished === this.size){
              for (let j = 0; j < this.size; j++) this.workers[j].off('message', handlers[j]);
              const wallMs = performance.now() - startedAt;
              // Round trajectory at write time to shrink the JSONL substantially.
              // e to int ms, pcts/nps to 4 decimals — well past what any analysis
              // distinguishes (1 part in 10000 of progress is sub-noise).
              const round4 = x => Math.round(x * 10000) / 10000;
              const trajRounded = trajectory.map(pt => ({
                e: Math.round(pt.e),
                pcts: pt.pcts.map(round4),
                nps: pt.nps.map(round4),
              }));
              resolve({
                wallMs, finalMs: wallMs,
                workerMs: wMs.slice(),
                total: wTotal.reduce((a,b) => a+b, 0),
                leaves: wLeaf.reduce((a,b) => a+b, 0),
                trajectory: trajRounded,
              });
            }
          }
        };
        handlers[k] = handler;
        this.workers[k].on('message', handler);
        this.workers[k].postMessage({
          state, slots, workerIdx: k, workerCount: this.size, seed,
        });
      }
    });
  }
}

// ----------------------------------------------------------------------------
// Prediction methods — prior (offline, before solve) and live (in-flight).
// All operate in log(ms) space because the underlying time spans many orders
// of magnitude; log-linear / log-RMSE is the appropriate metric.
// ----------------------------------------------------------------------------

// ---- PRIOR predictors ----
function prMean(history){
  // baseline: constant geometric mean of finalMs (== exp of mean of log)
  let s = 0; for (const h of history) s += Math.log(h.finalMs);
  const lm = s / history.length;
  return () => Math.exp(lm);
}
function prKernel1(history, key){
  // 1-D Gaussian kernel regression in log(ms); bandwidth = σ / √n.
  const xs = history.map(h => h.features[key]).filter(Number.isFinite);
  if (xs.length < 2) return null;
  const mean = xs.reduce((a,b) => a+b, 0) / xs.length;
  const variance = xs.reduce((a,b) => a + (b-mean)**2, 0) / xs.length;
  const sigma = Math.sqrt(variance);
  if (sigma < 1e-9) return null;
  const h = Math.max(sigma * 0.05, sigma / Math.sqrt(history.length));
  return (target) => {
    const tv = target.features[key];
    if (!Number.isFinite(tv)) return null;
    let wSum = 0, wLog = 0;
    for (let i = 0; i < history.length; i++){
      const x = history[i].features[key];
      if (!Number.isFinite(x)) continue;
      const d = tv - x;
      const w = Math.exp(-0.5 * (d/h)**2);
      wSum += w; wLog += w * Math.log(history[i].finalMs);
    }
    return wSum > 1e-12 ? Math.exp(wLog / wSum) : null;
  };
}
function prLogLinear1(history, key){
  const n = history.length;
  let sx = 0, sy = 0, sxx = 0, sxy = 0, used = 0;
  for (const h of history){
    const x = h.features[key]; if (!Number.isFinite(x)) continue;
    const y = Math.log(h.finalMs);
    sx += x; sy += y; sxx += x*x; sxy += x*y; used++;
  }
  if (used < 2) return null;
  const denom = used * sxx - sx*sx;
  if (Math.abs(denom) < 1e-12) return null;
  const b = (used * sxy - sx * sy) / denom;
  const a = (sy - b * sx) / used;
  return (target) => {
    const tv = target.features[key];
    return Number.isFinite(tv) ? Math.exp(a + b * tv) : null;
  };
}
// multivariate ridge in log space; standardized features
function solveLinear(A, bv){
  const n = A.length;
  const M = A.map((row,i) => [...row, bv[i]]);
  for (let i = 0; i < n; i++){
    let pivot = i;
    for (let j = i+1; j < n; j++) if (Math.abs(M[j][i]) > Math.abs(M[pivot][i])) pivot = j;
    if (Math.abs(M[pivot][i]) < 1e-12) return null;
    [M[i], M[pivot]] = [M[pivot], M[i]];
    for (let j = i+1; j < n; j++){
      const factor = M[j][i] / M[i][i];
      for (let k = i; k <= n; k++) M[j][k] -= factor * M[i][k];
    }
  }
  const x = new Array(n);
  for (let i = n-1; i >= 0; i--){
    let s = M[i][n];
    for (let j = i+1; j < n; j++) s -= M[i][j] * x[j];
    x[i] = s / M[i][i];
  }
  return x;
}
function prRidge(history, keys, lambda){
  const n = history.length, p = keys.length;
  // Note: NO `n < p+2` guard. Ridge with λ>0 is mathematically well-defined at
  // any n≥1 — when data is scarce, the λI term dominates and coefficients
  // shrink toward zero, so the prediction collapses to ybar (the log-mean of
  // training finalMs). That's a sensible cold-start fallback.
  if (n < 1) return null;
  const means = keys.map(k => { let s = 0; for (const h of history) s += h.features[k]; return s / n; });
  const stds  = keys.map((k,j) => { let s = 0; for (const h of history) s += (h.features[k] - means[j])**2; return Math.sqrt(s/n) || 1; });
  const X = history.map(h => keys.map((k,j) => (h.features[k] - means[j]) / stds[j]));
  const y = history.map(h => Math.log(h.finalMs));
  let ybar = 0; for (const v of y) ybar += v; ybar /= n;
  const XtX = Array.from({length: p}, () => new Array(p).fill(0));
  const Xty = new Array(p).fill(0);
  for (let i = 0; i < n; i++){
    const yc = y[i] - ybar;
    for (let a = 0; a < p; a++){
      Xty[a] += X[i][a] * yc;
      for (let b = 0; b < p; b++) XtX[a][b] += X[i][a] * X[i][b];
    }
  }
  for (let a = 0; a < p; a++) XtX[a][a] += lambda * n;
  const w = solveLinear(XtX, Xty);
  if (!w) return null;
  return (target) => {
    let pred = ybar;
    for (let j = 0; j < p; j++){
      const v = target.features[keys[j]];
      if (!Number.isFinite(v)) return null;
      pred += w[j] * (v - means[j]) / stds[j];
    }
    return Math.exp(pred);
  };
}
// Polynomial-expanded ridge: builds an extended feature set with squares and
// a handful of hand-picked interaction terms, then ridge over that. Captures
// non-linear relationships like time ~ logProd² that pure ridge misses.
function polyExpand(features, keys){
  const out = {};
  for (const k of keys){ out[k] = features[k]; out[k + '_sq'] = features[k] ** 2; }
  // hand-picked interactions: terms that "should" matter if combinatorial
  out['logProd_x_density']  = features.logProd * features.density;
  out['logProd_x_nOpen']    = features.logProd * features.nOpen;
  out['logProd_x_effBr']    = features.logProd * (features.effBranching || 0);
  out['logProdAdj_x_density'] = (features.logProdAdj || 0) * features.density;
  return out;
}
function prRidgePoly(history, keys, lambda){
  const expanded = history.map(h => ({ features: polyExpand(h.features, keys), finalMs: h.finalMs }));
  const expKeys = Object.keys(expanded[0].features);
  const inner = prRidge(expanded, expKeys, lambda);
  if (!inner) return null;
  return (target) => inner({ features: polyExpand(target.features, keys) });
}

// Kernel-weighted local ridge: in standardized feature space, weight each
// training point by exp(-d²/2h²), then fit ridge with those weights.
// Combines kernel's locality with ridge's ability to handle multivariate trends.
function prLocalRidge(history, keys, lambda, bandwidthMult){
  const n = history.length, p = keys.length;
  if (n < p + 2) return null;
  const means = keys.map(k => { let s = 0; for (const h of history) s += h.features[k]; return s / n; });
  const stds  = keys.map((k,j) => { let s = 0; for (const h of history) s += (h.features[k]-means[j])**2; return Math.sqrt(s/n) || 1; });
  const Xstd = history.map(h => keys.map((k,j) => (h.features[k]-means[j])/stds[j]));
  const y = history.map(h => Math.log(h.finalMs));
  // bandwidth = bandwidthMult × median pairwise distance, computed once
  const sample = Math.min(50, n);
  const dists = [];
  for (let i = 0; i < sample; i++) for (let j = i+1; j < sample; j++){
    let d2 = 0; for (let a = 0; a < p; a++){ const d = Xstd[i][a] - Xstd[j][a]; d2 += d*d; }
    dists.push(Math.sqrt(d2));
  }
  dists.sort((a,b) => a-b);
  const medianDist = dists[Math.floor(dists.length/2)] || 1;
  const bw = bandwidthMult * medianDist;
  return (target) => {
    const t = keys.map((k,j) => { const v = target.features[k]; return Number.isFinite(v) ? (v-means[j])/stds[j] : 0; });
    // weighted ridge: solve (X^T W X + λI) w = X^T W y
    const w = new Array(n);
    let wsum = 0;
    for (let i = 0; i < n; i++){
      let d2 = 0; for (let a = 0; a < p; a++){ const d = Xstd[i][a] - t[a]; d2 += d*d; }
      w[i] = Math.exp(-d2 / (2 * bw * bw));
      wsum += w[i];
    }
    if (wsum < 1e-12) return null;
    let ybar = 0; for (let i = 0; i < n; i++) ybar += w[i] * y[i]; ybar /= wsum;
    const XtX = Array.from({length: p}, () => new Array(p).fill(0));
    const Xty = new Array(p).fill(0);
    for (let i = 0; i < n; i++){
      const yc = y[i] - ybar;
      for (let a = 0; a < p; a++){
        Xty[a] += w[i] * Xstd[i][a] * yc;
        for (let b = 0; b < p; b++) XtX[a][b] += w[i] * Xstd[i][a] * Xstd[i][b];
      }
    }
    for (let a = 0; a < p; a++) XtX[a][a] += lambda * wsum;
    const wt = solveLinear(XtX, Xty);
    if (!wt) return null;
    let pred = ybar;
    for (let j = 0; j < p; j++) pred += wt[j] * t[j];
    return Math.exp(pred);
  };
}

// Stacking: build a single-feature kernel predictor for each feature, run each
// on each history point (LOO so we don't leak), then ridge over those predictions.
function prStacking(history, keys, lambda){
  const n = history.length;
  if (n < keys.length + 5) return null;
  // For each training point i, compute single-feature predictions using
  // train = history \ {i}. These are our "meta features".
  const metaFeatures = history.map(() => ({}));
  const metaY = history.map(h => Math.log(h.finalMs));
  const builtPreds = {};
  for (const k of keys){
    for (let i = 0; i < n; i++){
      const train = history.slice(0, i).concat(history.slice(i+1));
      const pred = prKernel1(train, k);
      const yhat = pred ? pred(history[i]) : null;
      metaFeatures[i]['k_' + k] = yhat && isFinite(yhat) && yhat > 0 ? Math.log(yhat) : metaY[i];
    }
    builtPreds['k_' + k] = prKernel1(history, k);
  }
  // ridge on meta features
  const metaKeys = Object.keys(metaFeatures[0]);
  const p = metaKeys.length;
  const means = metaKeys.map(k => { let s = 0; for (const f of metaFeatures) s += f[k]; return s/n; });
  const stds  = metaKeys.map((k,j) => { let s = 0; for (const f of metaFeatures) s += (f[k]-means[j])**2; return Math.sqrt(s/n) || 1; });
  const X = metaFeatures.map(f => metaKeys.map((k,j) => (f[k]-means[j])/stds[j]));
  let ybar = 0; for (const v of metaY) ybar += v; ybar /= n;
  const XtX = Array.from({length: p}, () => new Array(p).fill(0));
  const Xty = new Array(p).fill(0);
  for (let i = 0; i < n; i++){
    const yc = metaY[i] - ybar;
    for (let a = 0; a < p; a++){
      Xty[a] += X[i][a] * yc;
      for (let b = 0; b < p; b++) XtX[a][b] += X[i][a] * X[i][b];
    }
  }
  for (let a = 0; a < p; a++) XtX[a][a] += lambda * n;
  const w = solveLinear(XtX, Xty);
  if (!w) return null;
  return (target) => {
    const tm = {};
    for (const k of keys){
      const pp = builtPreds['k_' + k];
      const yhat = pp ? pp(target) : null;
      tm['k_' + k] = yhat && isFinite(yhat) && yhat > 0 ? Math.log(yhat) : ybar;
    }
    let pred = ybar;
    for (let j = 0; j < p; j++) pred += w[j] * (tm[metaKeys[j]] - means[j]) / stds[j];
    return Math.exp(pred);
  };
}

// Wrap a base predictor with bias correction: multiply its output by the
// historical median(truth/pred), computed in-sample. Cheap free win if the
// base predictor is biased.
function prBiasCorrect(buildBase, history){
  const base = buildBase(history);
  if (!base) return null;
  const ratios = [];
  for (let i = 0; i < history.length; i++){
    const train = history.slice(0, i).concat(history.slice(i+1));
    const inner = buildBase(train);
    if (!inner) continue;
    const yhat = inner(history[i]);
    if (!yhat || !isFinite(yhat) || yhat <= 0) continue;
    ratios.push(history[i].finalMs / yhat);
  }
  if (ratios.length === 0) return base;
  ratios.sort((a,b) => a-b);
  const correction = ratios[Math.floor(ratios.length/2)];
  return (target) => {
    const y = base(target);
    return (y && isFinite(y)) ? y * correction : null;
  };
}

function prKNN(history, keys, k){
  if (history.length < k) k = history.length;
  if (k < 1) return null;
  const means = keys.map(key => { let s = 0; for (const h of history) s += h.features[key]; return s/history.length; });
  const stds  = keys.map((key,j) => { let s = 0; for (const h of history) s += (h.features[key]-means[j])**2; return Math.sqrt(s/history.length) || 1; });
  const standardized = history.map(h => keys.map((key,j) => (h.features[key] - means[j])/stds[j]));
  return (target) => {
    const t = keys.map((key,j) => {
      const v = target.features[key]; return Number.isFinite(v) ? (v - means[j])/stds[j] : 0;
    });
    const dists = standardized.map((row, i) => {
      let d2 = 0; for (let j = 0; j < row.length; j++){ const d = row[j] - t[j]; d2 += d*d; }
      return { i, d: Math.sqrt(d2) };
    });
    dists.sort((a,b) => a.d - b.d);
    let s = 0;
    for (let i = 0; i < k; i++) s += Math.log(history[dists[i].i].finalMs);
    return Math.exp(s / k);
  };
}

// ----------------------------------------------------------------------------
// EVAL: leave-one-out CV across the history. Mean |log err| (lower is better),
// plus median ratio (pred/true), 90th-pct ratio, fraction within 2x.
// ----------------------------------------------------------------------------
function evalPrior(history, name, buildPredictor, evalMinMs){
  // LOO across the WHOLE history — predictors learn from every solve regardless
  // of length (more data = better fits). But the held-out test point only counts
  // toward the score if it's >= evalMinMs (we only care about predicting solves
  // long enough to matter).
  let sumAbsLog = 0, n = 0;
  const ratios = [];
  for (let i = 0; i < history.length; i++){
    if (history[i].finalMs < evalMinMs) continue;
    const train = history.slice(0, i).concat(history.slice(i+1));
    if (train.length < 3) continue;
    const pred = buildPredictor(train);
    if (!pred) continue;
    const yhat = pred(history[i]);
    if (!yhat || !isFinite(yhat) || yhat <= 0) continue;
    const y = history[i].finalMs;
    sumAbsLog += Math.abs(Math.log(yhat) - Math.log(y));
    ratios.push(yhat / y);
    n++;
  }
  if (!n) return null;
  ratios.sort((a,b) => a - b);
  const within2x = ratios.filter(r => r >= 0.5 && r <= 2).length / ratios.length;
  const within3x = ratios.filter(r => r >= 1/3 && r <= 3).length / ratios.length;
  const medRatio = ratios[Math.floor(ratios.length/2)];
  const p90Ratio = ratios[Math.floor(ratios.length*0.9)];
  const p10Ratio = ratios[Math.floor(ratios.length*0.1)];
  return {
    name, n,
    meanAbsLog: sumAbsLog / n,
    medRatio, p10Ratio, p90Ratio,
    within2x, within3x,
  };
}

// ----------------------------------------------------------------------------
// LIVE predictors — evaluated by replaying each board's trajectory.
// For each sample in the trajectory, we compute estimated REMAINING ms by
// each method, then compare against truth = finalMs - elapsed.
// Methods that need a "prior" use the LOO prediction from the best prior method.
// ----------------------------------------------------------------------------
function liveStratEtas(traj, finalMs, priorMs, opts){
  const out = [];
  const alphas = [0.05, 0.10, 0.20];
  const emaStates = alphas.map(() => ({ rate: null, lastE: null, lastP: null }));
  const calibFn = opts && opts.calibFn || (p => p);
  // Per-prior-length calibration: pick a warp keyed on priorMs
  const calibFnBucketed = opts && opts.calibFnBucketed
    ? opts.calibFnBucketed(priorMs) : calibFn;
  // σ²_live(p): variance of log-residual log(e/calib(p)) − log(finalMs), from training,
  // bucketed by avgP. Default: heuristic that's huge near p→0 and shrinks as p→1.
  const liveLogVar = opts && opts.liveLogVar || (p => {
    const pp = Math.max(0.001, Math.min(0.999, p));
    return 0.25 + 4 * (1 - pp) * (1 - pp) / pp;     // crude fallback
  });
  const liveLogVarMin = opts && opts.liveLogVarMin || liveLogVar;
  // Per-(method,phase) bias correction: subtract this from log(predTotal)
  // before exp, to remove systematic over/under-prediction. Default no-op.
  const biasFor = opts && opts.biasFor || ((name, phase, priorMs) => 0);
  const biasFor2d = opts && opts.biasFor2d || null;
  // σ²_prior: variance of bestPrior's log-residuals, from CV. Default modest.
  const priorLogVar = opts && opts.priorLogVar != null ? opts.priorLogVar : 0.06;
  // Track per-worker "has completed any branch" for gated min projection
  // (we infer this from pcts hitting > 0; nps would also work but is noisier).
  // probe-based predictors: at the first trajectory sample past T ms, lock in
  // an estimate of finalMs and let ETA count down from there. Probe times are
  // configurable; tests several so we can see the latency/accuracy tradeoff.
  const probeTimes = [200, 500, 1000, 2000];
  const probeTotals = {};       // probeTimes[i] -> locked-in total estimate (ms)
  for (const pt of traj){
    const e = pt.e;
    for (const T of probeTimes){
      if (probeTotals[T] == null && e >= T){
        const avgP = pt.pcts.reduce((a,b)=>a+b,0)/pt.pcts.length;
        const cp = Math.max(0.0001, Math.min(0.9999, calibFn(avgP)));
        probeTotals[T] = e / cp;
      }
    }
  }
  for (const pt of traj){
    const e = pt.e;
    const trueR = finalMs - e;
    if (trueR < 0) continue;
    const avgP = pt.pcts.reduce((a,b)=>a+b,0)/pt.pcts.length;
    const maxP = Math.max(...pt.pcts);
    const minP = Math.min(...pt.pcts);
    const sortP = pt.pcts.slice().sort((a,b)=>a-b);
    const medP  = sortP[Math.floor(sortP.length/2)];
    const npMin = pt.nps.length ? Math.min(...pt.nps) : 0;
    const npAvg = pt.nps.length ? pt.nps.reduce((a,b)=>a+b,0)/pt.nps.length : 0;
    function lin(p){ const sp = Math.max(0.0001, Math.min(0.9999, p)); return e * (1-sp)/sp; }
    // worker-agreement spread (0 = perfect agreement, 1 = max disagreement)
    let meanP = avgP, varP = 0;
    for (const p of pt.pcts) varP += (p - meanP) * (p - meanP);
    varP /= pt.pcts.length;
    const stdP = Math.sqrt(varP);
    const ests = {
      linAvg:    lin(avgP),
      linMax:    lin(maxP),     // fastest worker — collapses to 0 prematurely
      linMin:    lin(minP),     // slowest worker — bad early (minP=0), great late
      linMedian: lin(medP),
      linNodeMin: lin(npMin),
      linNodeAvg: lin(npAvg),
      linCalibAvg: lin(calibFn(avgP)),
      linCalibAvgBucket: lin(calibFnBucketed(avgP)),    // per-prior-length warp
      linCalibMed: lin(calibFn(medP)),                  // robust against stragglers/leaders
    };
    // Gated min: only project from slowest worker once it has non-zero progress
    // (else linMin produces "infinite remaining" early and ruins the average).
    const minHasProgress = minP > 0;
    if (minHasProgress){
      ests.gatedLinMin     = lin(minP);
      ests.gatedLinMinCal  = lin(calibFn(minP));
    }
    // EMA on rate (dp/dt of branch_pct_avg)
    for (let i = 0; i < alphas.length; i++){
      const st = emaStates[i];
      if (st.lastE != null){
        const dt = e - st.lastE, dp = avgP - st.lastP;
        if (dt > 0 && dp > 0){
          const inst = dp / dt;
          st.rate = st.rate == null ? inst : (alphas[i] * inst + (1 - alphas[i]) * st.rate);
        }
      }
      st.lastE = e; st.lastP = avgP;
      const safe = Math.max(0.0001, Math.min(0.9999, avgP));
      ests['ema_' + alphas[i]] = st.rate != null && st.rate > 0 ? (1 - safe) / st.rate : null;
    }
    // Prior-only (no live update): predicted - elapsed
    ests.priorOnly = priorMs != null ? Math.max(0, priorMs - e) : null;
    // Probe-based: lock estimate at T ms, then ETA = lockedTotal - elapsed
    for (const T of probeTimes){
      const tot = probeTotals[T];
      ests['probe@' + T] = tot != null && e >= T ? Math.max(0, tot - e) : null;
    }
    // Probe combined with prior: geomean of the two
    for (const T of probeTimes){
      const tot = probeTotals[T];
      if (tot != null && e >= T && priorMs != null){
        const combinedTotal = Math.sqrt(tot * priorMs);
        ests['probe@' + T + '×prior_geo'] = Math.max(0, combinedTotal - e);
      } else { ests['probe@' + T + '×prior_geo'] = null; }
    }
    // Prior×Live blends: w·prior_remaining + (1-w)·live_remaining
    if (priorMs != null){
      const priorR = Math.max(0, priorMs - e);
      const liveR = ests.linAvg;
      const liveRcal = ests.linCalibAvg;
      // weight decays as elapsed grows past prior's predicted halfway
      const tau = priorMs * 0.5;
      const w = Math.exp(-e / Math.max(1, tau));
      ests.blendPriorAvg_exp = w * priorR + (1 - w) * liveR;
      // simpler: hard switch at 25% of prior
      ests.blendPriorAvg_hard25 = e < priorMs * 0.25 ? priorR : liveR;
      // blend with linMax (slowest worker — best late-stage signal)
      const liveRmax = ests.linMax;
      ests.blendPriorMax_exp = w * priorR + (1 - w) * liveRmax;
      // "geomean" blend in log space
      const gw = Math.max(0.01, Math.min(0.99, w));
      ests.blendPriorAvg_geo = Math.exp(gw * Math.log(Math.max(1, priorR)) + (1-gw) * Math.log(Math.max(1, liveR)));
      // Calibrated-branch blends (likely the right answer — calib wins late,
      // prior wins early, so a smooth handoff captures both regimes)
      ests.blendPriorCalib_exp     = w * priorR + (1 - w) * liveRcal;
      ests.blendPriorCalib_hard25  = e < priorMs * 0.25 ? priorR : liveRcal;
      ests.blendPriorCalib_hard50  = e < priorMs * 0.50 ? priorR : liveRcal;
      ests.blendPriorCalib_geo     = Math.exp(gw * Math.log(Math.max(1, priorR)) + (1-gw) * Math.log(Math.max(1, liveRcal)));
      // Faster handoff: w decays at tau = priorMs/4 (so it's near zero by t=priorMs)
      const wFast = Math.exp(-e / Math.max(1, priorMs * 0.25));
      ests.blendPriorCalib_expFast = wFast * priorR + (1 - wFast) * liveRcal;

      // ---- new methods aimed at beating priorOnly across all phases ----

      // maxPriorMin: never claim less remaining than the slowest worker projects.
      // Inherits priorOnly's early-phase accuracy; tightens late-phase when min
      // worker is the true bottleneck. If min hasn't started, just use prior.
      if (minHasProgress){
        ests.maxPriorMin    = Math.max(priorR, ests.gatedLinMin);
        ests.maxPriorMinCal = Math.max(priorR, ests.gatedLinMinCal);
      } else {
        ests.maxPriorMin = priorR;
        ests.maxPriorMinCal = priorR;
      }

      // bayesLogBlend: precision-weighted log-space blend of prior total-time
      // and live total-time estimates. σ²_live(p, priorMs) is now 2D — varies
      // with both progress level and predicted solve length.
      {
        const calForBlend = calibFnBucketed;
        const cp = Math.max(0.001, Math.min(0.999, calForBlend(avgP)));
        const liveTot = e / cp;
        const lvar = Math.max(0.001, liveLogVar(avgP, priorMs));
        const pvar = Math.max(0.001, priorLogVar);
        const tauP = 1 / pvar, tauL = 1 / lvar;
        const muPost = (Math.log(priorMs) * tauP + Math.log(liveTot) * tauL) / (tauP + tauL);
        ests.bayesLogBlend = Math.max(0, Math.exp(muPost) - e);
      }

      // EXPERIMENT: bayesLogBlend_nocal — same Bayesian blend, but feed RAW
      // avgP to the live total estimate (no calibration warp). Tests whether
      // per-phase bias correction alone can substitute for calibration.
      {
        const pp = Math.max(0.001, Math.min(0.999, avgP));
        const liveTot = e / pp;
        const lvar = Math.max(0.001, liveLogVar(avgP, priorMs));   // reuse 2D variance
        const pvar = Math.max(0.001, priorLogVar);
        const tauP = 1 / pvar, tauL = 1 / lvar;
        const muPost = (Math.log(priorMs) * tauP + Math.log(liveTot) * tauL) / (tauP + tauL);
        ests.bayesLogBlend_nocal = Math.max(0, Math.exp(muPost) - e);
      }

      // bayesLogBlendMin: same as bayesLogBlend but evidence is from slowest worker
      // (more conservative — slow worker is the true bottleneck once it starts).
      if (minHasProgress){
        const calForBlend = calibFnBucketed;
        const cp = Math.max(0.001, Math.min(0.999, calForBlend(minP)));
        const liveTot = e / cp;
        const lvar = Math.max(0.001, liveLogVarMin(minP, priorMs));
        const pvar = Math.max(0.001, priorLogVar);
        const tauP = 1 / pvar, tauL = 1 / lvar;
        const muPost = (Math.log(priorMs) * tauP + Math.log(liveTot) * tauL) / (tauP + tauL);
        ests.bayesLogBlendMin = Math.max(0, Math.exp(muPost) - e);
      }

      // bayesDual: combine prior + avgP-evidence + minP-evidence in log-space
      // with three precisions. Each signal contributes inversely to its variance,
      // so weak signals (e.g. minP=0 case → drop) auto-discount. This should be
      // smoothly best across all phases since it uses every reliable channel.
      {
        const pvar = Math.max(0.001, priorLogVar);
        const cpA = Math.max(0.001, Math.min(0.999, calibFnBucketed(avgP)));
        const lvarA = Math.max(0.001, liveLogVar(avgP));
        let tauSum = 1 / pvar, muSum = Math.log(priorMs) / pvar;
        const liveTotA = e / cpA;
        tauSum += 1 / lvarA; muSum += Math.log(liveTotA) / lvarA;
        if (minHasProgress){
          const cpM = Math.max(0.001, Math.min(0.999, calibFnBucketed(minP)));
          const lvarM = Math.max(0.001, liveLogVarMin(minP, priorMs));
          const liveTotM = e / cpM;
          tauSum += 1 / lvarM; muSum += Math.log(liveTotM) / lvarM;
        }
        ests.bayesDual = Math.max(0, Math.exp(muSum / tauSum) - e);
      }

      // phaseHybrid: best-per-phase composite. bayesLogBlend until elapsed
      // reaches predicted 75%, then gatedLinMin (which dominated final phase).
      // Falls back to bayesLogBlend if gated isn't ready (minP=0).
      {
        const predPhase = e / Math.max(1, priorMs);
        if (predPhase < 0.75 || !minHasProgress){
          ests.phaseHybrid = ests.bayesLogBlend;
        } else {
          ests.phaseHybrid = ests.gatedLinMin;
        }
      }

      // agreementBlend: trust live in proportion to how much workers agree.
      // stdP near 0 = all workers same phase, avgP is informative. stdP large
      // means there's a straggler, so the avg lies and we should trust prior.
      {
        const agree = Math.exp(-stdP / 0.15);     // ~1 when stdP<<0.15, →0 when stdP>>0.15
        const cp = Math.max(0.001, Math.min(0.999, calibFnBucketed(avgP)));
        const liveTotA = e / cp;
        const liveRcalA = Math.max(0, liveTotA - e);
        ests.agreementBlend = agree * liveRcalA + (1 - agree) * priorR;
      }

      // priorScaledByElapsed: if elapsed has already exceeded prior, we know
      // prior was an underestimate. Scale up by the over-budget ratio.
      // Guarantees we never report 0 remaining while the solve is still running.
      if (e > priorMs){
        const overrun = e / priorMs;
        // assume remaining work is proportional to the overrun factor relative to
        // a reasonable upper bound; cap multiplier so we don't explode
        const scaled = Math.max(priorR, e * 0.5);   // at minimum, claim half-elapsed more
        ests.priorScaled = Math.max(0, scaled);
      } else {
        ests.priorScaled = priorR;
      }
    }
    // Build composite methods AFTER bc is available — `_bc2` variants use
    // bias-corrected components so they inherit the correction.
    // (Run twice: first build bc variants for everything, then composites.)
    // Bias-corrected (_bc) variant of EVERY method computed above. Pass 1
    // measures each method's mean log-residual per predicted-phase bucket;
    // pass 2 (biasFor != 0) subtracts it. No-op when bias < 1e-6.
    const phaseForBC = priorMs ? Math.min(0.999, e / Math.max(1, priorMs)) : 0;
    const baseKeys = Object.keys(ests);
    for (const target of baseKeys){
      const orig = ests[target];
      if (orig == null || !isFinite(orig)) continue;
      // 1D bias (predicted-phase only)
      const bias1d = biasFor(target, phaseForBC, priorMs);
      if (Math.abs(bias1d) >= 1e-6){
        const totOrig = orig + e;
        const totBC   = Math.exp(Math.log(Math.max(1, totOrig)) - bias1d);
        ests[target + '_bc'] = Math.max(0, totBC - e);
      }
      // 2D bias (predicted-phase × log(priorMs))
      if (biasFor2d){
        const bias2d = biasFor2d(target, phaseForBC, priorMs);
        if (Math.abs(bias2d) >= 1e-6){
          const totOrig = orig + e;
          const totBC   = Math.exp(Math.log(Math.max(1, totOrig)) - bias2d);
          ests[target + '_bc2d'] = Math.max(0, totBC - e);
        }
      }
    }

    // ---- composite methods built from BC-corrected components ----
    if (priorMs != null){
      const baseLogB = ests.bayesLogBlend_bc != null ? ests.bayesLogBlend_bc : ests.bayesLogBlend;
      const baseMin  = ests.gatedLinMin_bc   != null ? ests.gatedLinMin_bc   : ests.gatedLinMin;
      const predPhase = e / Math.max(1, priorMs);
      // Sharp handoff at predicted-75%
      if (baseMin != null && predPhase >= 0.75){
        ests.phaseHybrid_bc2 = baseMin;
      } else {
        ests.phaseHybrid_bc2 = baseLogB;
      }
      // Smooth sigmoid handoff centred on predPhase=0.75, width 0.1
      if (baseMin != null){
        const w = 1 / (1 + Math.exp(-(predPhase - 0.75) / 0.1));
        ests.phaseHybridSig_bc2 = (1 - w) * baseLogB + w * baseMin;
      }
      // Earlier handoff at 0.5
      if (baseMin != null && predPhase >= 0.5){
        ests.phaseHybrid50_bc2 = baseMin;
      } else {
        ests.phaseHybrid50_bc2 = baseLogB;
      }
      // Take the conservative max — never claim less than slowest worker
      if (baseMin != null){
        ests.maxBayesMin_bc2 = Math.max(baseLogB, baseMin);
      }
      // CHAMPION CANDIDATE: bayesLogBlend_bc (best overall) early/mid, switch
      // to uncorrected gatedLinMin late (where it dominates the worst-bucket).
      // Uses bc on the prior side because the prior has consistent log-bias;
      // skips bc on the live extrapolator because its bias is heterogeneous
      // and bc'ing it overcorrects in deployment.
      // Prefer 2D-corrected base if available
      const bayesBC = ests.bayesLogBlend_bc2d != null ? ests.bayesLogBlend_bc2d
                    : ests.bayesLogBlend_bc   != null ? ests.bayesLogBlend_bc
                    : ests.bayesLogBlend;
      const gated   = ests.gatedLinMin;     // raw, NOT _bc (bc hurts here)
      if (gated != null && predPhase >= 0.75){
        ests.champion = gated;
      } else {
        ests.champion = bayesBC;
      }
      // Same but smooth sigmoid handoff
      if (gated != null){
        const w = 1 / (1 + Math.exp(-(predPhase - 0.75) / 0.08));
        ests.championSmooth = (1 - w) * bayesBC + w * gated;
      } else {
        ests.championSmooth = bayesBC;
      }
      // Sweep of hard-switch thresholds to map the trade-off frontier
      for (const thr of [0.50, 0.60, 0.65, 0.70, 0.80, 0.85, 0.90]){
        const lbl = 'champion' + Math.round(thr*100);
        ests[lbl] = (gated != null && predPhase >= thr) ? gated : bayesBC;
      }
      // Sweep smooth-handoff centres at width 0.08
      for (const ctr of [0.60, 0.70, 0.80, 0.85]){
        const lbl = 'champSm' + Math.round(ctr*100);
        if (gated != null){
          const w = 1 / (1 + Math.exp(-(predPhase - ctr) / 0.08));
          ests[lbl] = (1 - w) * bayesBC + w * gated;
        } else {
          ests[lbl] = bayesBC;
        }
      }
    }
    out.push({ e, trueR, ests });
  }
  return out;
}

// build a calibration warp from training set: pool (avgP, true_p) pairs from
// every solve, sort, bin into quantiles, take mean of each bin.
// If `globalWarp` and `alpha>0` are passed, each bin's true_p is shrunk toward
// `globalWarp(bin_centroid_avgP)` with pseudocount `alpha` (Bayesian shrinkage).
function buildCalib(trainSet, alpha, globalWarp){
  const pairs = [];
  for (const r of trainSet){
    for (const pt of r.trajectory){
      const avgP = pt.pcts.reduce((a,b)=>a+b,0)/pt.pcts.length;
      const truep = pt.e / r.finalMs;
      if (truep >= 0 && truep <= 1) pairs.push([avgP, truep]);
    }
  }
  if (pairs.length < 50) return globalWarp || (p => p);
  pairs.sort((a,b) => a[0] - b[0]);
  const N = 20;
  const table = [{ p: 0, true_p: 0 }];
  const shrink = alpha > 0 && globalWarp;
  for (let i = 0; i < N; i++){
    const lo = Math.floor(i * pairs.length / N);
    const hi = Math.floor((i+1) * pairs.length / N);
    if (hi <= lo) continue;
    let sx = 0, sy = 0;
    for (let j = lo; j < hi; j++){ sx += pairs[j][0]; sy += pairs[j][1]; }
    const n = hi - lo;
    const pCent = sx/n;
    let tp = sy/n;
    if (shrink){
      tp = (n * tp + alpha * globalWarp(pCent)) / (n + alpha);
    }
    table.push({ p: pCent, true_p: tp });
  }
  table.push({ p: 1, true_p: 1 });
  // monotonize true_p
  for (let i = 1; i < table.length; i++) if (table[i].true_p < table[i-1].true_p) table[i].true_p = table[i-1].true_p;
  return (p) => {
    if (p <= 0) return 0; if (p >= 1) return 1;
    for (let i = 0; i < table.length - 1; i++){
      const a = table[i], b = table[i+1];
      if (p >= a.p && p <= b.p){
        const t = (p - a.p) / (b.p - a.p);
        return a.true_p + t * (b.true_p - a.true_p);
      }
    }
    return p;
  };
}

// Shared realistic time bins used by calibration, bias correction, and the
// 2D live-variance tables. Asymmetric/log-spaced to match how solve times
// actually distribute (most boards <30s, heavy tail to 11+ min).
const TIME_BINS = [
  { lo: 0,       hi: 1000,    name: '<1s' },
  { lo: 1000,    hi: 5000,    name: '1-5s' },
  { lo: 5000,    hi: 30000,   name: '5-30s' },
  { lo: 30000,   hi: 120000,  name: '30s-2m' },
  { lo: 120000,  hi: Infinity, name: '>2m' },
];
function timeBinOf(ms){
  if (ms == null || !isFinite(ms)) return -1;
  for (let k = 0; k < TIME_BINS.length; k++) if (ms >= TIME_BINS[k].lo && ms < TIME_BINS[k].hi) return k;
  return -1;
}

// Per-prior-length calibration warps. One 1-D warp per TIME_BINS bucket; at
// deployment we pick the warp by predicted finalMs (priorMs). Falls back to
// a pooled flat warp when a bin is too sparse to fit reliably.
function buildCalibBuckets(trainSet, alpha){
  const flat = buildCalib(trainSet);                       // pooled global (no shrinkage on itself)
  const perBin = TIME_BINS.map(bin => {
    const sub = trainSet.filter(r => r.finalMs >= bin.lo && r.finalMs < bin.hi);
    if (sub.length < 10) return flat;                       // truly empty bucket → just use global
    // With α>0, every per-length warp blends toward the pooled flat warp at each quantile centroid.
    return buildCalib(sub, alpha || 0, flat);
  });
  return (priorMs) => {
    const t = timeBinOf(priorMs);
    return t >= 0 ? perBin[t] : flat;
  };
}

// 2D live-residual variance: bin by (pct × log(finalMs)). Live signal noise
// at a given progress level varies with solve length — short solves have very
// few trajectory samples, large quantization error; long ones are smoother.
// Returns (p, predictedFinalMs) → variance. Falls back to 1D when sparse.
function buildLiveLogVar2d(trainSet, calibPickFn, useMin, alpha){
  const NBINS = 20;
  const TB = TIME_BINS.length;
  const sums   = Array.from({length: NBINS}, () => new Array(TB).fill(0));
  const sumsSq = Array.from({length: NBINS}, () => new Array(TB).fill(0));
  const counts = Array.from({length: NBINS}, () => new Array(TB).fill(0));
  // 1D fallback per pct-bin
  const sums1 = new Array(NBINS).fill(0), sq1 = new Array(NBINS).fill(0), cnt1 = new Array(NBINS).fill(0);
  for (const r of trainSet){
    const cal = calibPickFn ? calibPickFn(r.finalMs) : (p => p);
    const tBin = timeBinOf(r.finalMs);
    for (const pt of r.trajectory){
      if (!pt.pcts || pt.pcts.length === 0) continue;
      const p = useMin ? Math.min(...pt.pcts) : pt.pcts.reduce((a,b)=>a+b,0)/pt.pcts.length;
      if (pt.e <= 0 || p <= 0 || p >= 1) continue;
      const cp = Math.max(0.001, Math.min(0.999, cal(p)));
      const liveTot = pt.e / cp;
      if (!isFinite(liveTot) || liveTot <= 0) continue;
      const res = Math.log(liveTot) - Math.log(r.finalMs);
      const b = Math.min(NBINS-1, Math.max(0, Math.floor(p * NBINS)));
      sums1[b] += res; sq1[b] += res*res; cnt1[b]++;
      if (tBin >= 0){
        sums[b][tBin] += res; sumsSq[b][tBin] += res*res; counts[b][tBin]++;
      }
    }
  }
  // 1D fallback table
  const var1 = new Array(NBINS);
  for (let b = 0; b < NBINS; b++){
    if (cnt1[b] < 5){ var1[b] = null; continue; }
    const m = sums1[b]/cnt1[b];
    var1[b] = Math.max(0.001, sq1[b]/cnt1[b] - m*m);
  }
  for (let b = 1; b < NBINS; b++) if (var1[b] == null && var1[b-1] != null) var1[b] = var1[b-1];
  for (let b = NBINS-2; b >= 0; b--) if (var1[b] == null && var1[b+1] != null) var1[b] = var1[b+1];
  for (let b = 0; b < NBINS; b++) if (var1[b] == null) var1[b] = 1.0;
  // 2D table. With α>0, each cell's variance is shrunk toward its 1D fallback
  // by pseudocount α; α=0 reproduces the hard-cutoff behavior (cell mean above
  // threshold, 1D fallback below).
  const A = Math.max(0, alpha || 0);
  const vars2 = Array.from({length: NBINS}, () => new Array(TB).fill(null));
  for (let b = 0; b < NBINS; b++) for (let t = 0; t < TB; t++){
    const n = counts[b][t];
    if (A <= 0){
      if (n < 8){ vars2[b][t] = var1[b]; continue; }
      const m = sums[b][t] / n;
      vars2[b][t] = Math.max(0.001, sumsSq[b][t]/n - m*m);
    } else {
      // shrink toward 1D global variance for this avgP-bin
      const varCell = n > 0 ? Math.max(0, sumsSq[b][t]/n - (sums[b][t]/n)**2) : 0;
      vars2[b][t] = Math.max(0.001, (n * varCell + A * var1[b]) / (n + A));
    }
  }
  return (p, priorMs) => {
    const tBin = timeBinOf(priorMs);
    const x = Math.max(0, Math.min(0.999999, p)) * NBINS;
    const i = Math.min(NBINS-1, Math.floor(x));
    const j = Math.min(NBINS-1, i+1);
    const t = x - i;
    const a = tBin >= 0 ? vars2[i][tBin] : var1[i];
    const b = tBin >= 0 ? vars2[j][tBin] : var1[j];
    return a * (1-t) + b * t;
  };
}

// Same as buildLiveLogVar but evidence is min(pcts) — slowest worker —
// instead of avg. Used by bayesLogBlendMin / bayesDual.
function buildLiveLogVarMin(trainSet, calibPickFn){
  const NBINS = 20;
  const sums = new Array(NBINS).fill(0);
  const sumsSq = new Array(NBINS).fill(0);
  const counts = new Array(NBINS).fill(0);
  for (const r of trainSet){
    const cal = calibPickFn ? calibPickFn(r.finalMs) : (p => p);
    for (const pt of r.trajectory){
      if (!pt.pcts || pt.pcts.length === 0) continue;
      const minP = Math.min(...pt.pcts);
      if (pt.e <= 0 || minP <= 0 || minP >= 1) continue;
      const cp = Math.max(0.001, Math.min(0.999, cal(minP)));
      const liveTot = pt.e / cp;
      if (!isFinite(liveTot) || liveTot <= 0) continue;
      const res = Math.log(liveTot) - Math.log(r.finalMs);
      const b = Math.min(NBINS-1, Math.max(0, Math.floor(minP * NBINS)));
      sums[b] += res; sumsSq[b] += res*res; counts[b]++;
    }
  }
  const vars = new Array(NBINS);
  for (let b = 0; b < NBINS; b++){
    if (counts[b] < 5) { vars[b] = null; continue; }
    const m = sums[b] / counts[b];
    vars[b] = Math.max(0.001, sumsSq[b]/counts[b] - m*m);
  }
  for (let b = 1; b < NBINS; b++) if (vars[b] == null && vars[b-1] != null) vars[b] = vars[b-1];
  for (let b = NBINS-2; b >= 0; b--) if (vars[b] == null && vars[b+1] != null) vars[b] = vars[b+1];
  for (let b = 0; b < NBINS; b++) if (vars[b] == null) vars[b] = 1.0;
  return (p) => {
    const x = Math.max(0, Math.min(0.999999, p)) * NBINS;
    const i = Math.min(NBINS-1, Math.floor(x));
    const j = Math.min(NBINS-1, i+1);
    const t = x - i;
    return vars[i] * (1-t) + vars[j] * t;
  };
}

// Variance of log-residual log(e/calib(avgP)) − log(finalMs), bucketed by avgP.
// Used by bayesLogBlend to know how much to trust the live signal at each
// progress level. Bins of 0.05 width; linear interp between bin centers.
function buildLiveLogVar(trainSet, calibPickFn){
  const NBINS = 20;     // 0.05 bins on [0,1]
  const sums = new Array(NBINS).fill(0);
  const sumsSq = new Array(NBINS).fill(0);
  const counts = new Array(NBINS).fill(0);
  for (const r of trainSet){
    const cal = calibPickFn ? calibPickFn(r.finalMs) : (p => p);
    for (const pt of r.trajectory){
      const avgP = pt.pcts.reduce((a,b)=>a+b,0)/pt.pcts.length;
      if (pt.e <= 0 || avgP <= 0 || avgP >= 1) continue;
      const cp = Math.max(0.001, Math.min(0.999, cal(avgP)));
      const liveTot = pt.e / cp;
      if (!isFinite(liveTot) || liveTot <= 0) continue;
      const res = Math.log(liveTot) - Math.log(r.finalMs);
      const b = Math.min(NBINS-1, Math.max(0, Math.floor(avgP * NBINS)));
      sums[b] += res; sumsSq[b] += res*res; counts[b]++;
    }
  }
  const vars = new Array(NBINS);
  for (let b = 0; b < NBINS; b++){
    if (counts[b] < 5) { vars[b] = null; continue; }
    const m = sums[b] / counts[b];
    vars[b] = Math.max(0.001, sumsSq[b]/counts[b] - m*m);
  }
  // fill nulls by forward/backward propagation
  for (let b = 1; b < NBINS; b++) if (vars[b] == null && vars[b-1] != null) vars[b] = vars[b-1];
  for (let b = NBINS-2; b >= 0; b--) if (vars[b] == null && vars[b+1] != null) vars[b] = vars[b+1];
  for (let b = 0; b < NBINS; b++) if (vars[b] == null) vars[b] = 1.0;
  return (p) => {
    const x = Math.max(0, Math.min(0.999999, p)) * NBINS;
    const i = Math.min(NBINS-1, Math.floor(x));
    const j = Math.min(NBINS-1, i+1);
    const t = x - i;
    return vars[i] * (1-t) + vars[j] * t;
  };
}

// Compute LOO log-residual variance of a prior predictor across the history.
// Used as σ²_prior in bayesLogBlend.
function priorLogResidualVar(history, buildPredictor, evalMinMs){
  let n = 0, sum = 0, sumSq = 0;
  for (let i = 0; i < history.length; i++){
    if (history[i].finalMs < evalMinMs) continue;
    const train = history.slice(0, i).concat(history.slice(i+1));
    if (train.length < 3) continue;
    const pred = buildPredictor(train);
    if (!pred) continue;
    const yhat = pred(history[i]);
    if (!yhat || !isFinite(yhat) || yhat <= 0) continue;
    const r = Math.log(yhat) - Math.log(history[i].finalMs);
    sum += r; sumSq += r*r; n++;
  }
  if (n < 2) return 0.06;
  const m = sum / n;
  return Math.max(0.005, sumSq/n - m*m);
}

// ----------------------------------------------------------------------------
// Main run loop
// ----------------------------------------------------------------------------
function fmtSec(ms){
  if (ms < 1000) return ms.toFixed(0) + 'ms';
  if (ms < 60_000) return (ms/1000).toFixed(2) + 's';
  return Math.floor(ms/60000) + 'm' + ((ms%60000)/1000).toFixed(0) + 's';
}

async function runBench(){
  // ALWAYS append. Never clobber the user's accumulated data. To start fresh,
  // delete bench-results.jsonl manually before running.
  const existing = fs.existsSync(RESULTS_PATH)
    ? fs.readFileSync(RESULTS_PATH, 'utf8').split('\n').filter(s => s.trim()).length
    : 0;
  console.log(`Pool size: ${POOL_SIZE}; budget: ${BUDGET_MIN} min; output: ${path.basename(RESULTS_PATH)} (appending to ${existing} existing solves)`);
  const pool = new Pool(POOL_SIZE);
  const startedAt = performance.now();
  let n = 0;
  while (performance.now() - startedAt < BUDGET_MS){
    const board = genBoard();
    if (!board) continue;
    const t0 = performance.now();
    const slotStr0 = board.slots.map(s => `${s.count}×${s.shape}`).join('+');
    // heartbeat: every 30s while we're still inside this solve, dump a "still alive" line
    const heartbeat = setInterval(() => {
      const secs = ((performance.now() - t0) / 1000).toFixed(0);
      console.log(`  ... board #${n+1} running for ${secs}s  (${slotStr0}, logProd=${board.features.logProd.toFixed(2)})`);
    }, 30_000);
    let r;
    try { r = await pool.solve(board); }
    catch (e){ clearInterval(heartbeat); console.log('  solve failed:', e.message); continue; }
    clearInterval(heartbeat);
    const elapsedTotal = performance.now() - startedAt;
    const remaining = BUDGET_MS - elapsedTotal;
    n++;
    // we drop boards with zero solutions: still informative for time but
    // breaks log-time analysis if total=0 means a tiny constant time
    if (r.finalMs < 30){                // sub-30ms solves carry no trajectory; don't save
      console.log(`  (skip #${n} ${fmtSec(r.finalMs).padStart(6)}  ${slotStr0}  logProd=${board.features.logProd.toFixed(2)})`);
      continue;
    }
    const rec = {
      i: n,
      slots: board.slots,
      state: board.state,
      features: board.features,
      finalMs: r.finalMs,
      wallMs: r.wallMs,
      workerMs: r.workerMs,
      total: r.total,
      leaves: r.leaves,
      trajectory: r.trajectory,
    };
    fs.appendFileSync(RESULTS_PATH, JSON.stringify(rec) + '\n');
    console.log(`#${String(n).padStart(4)} ${fmtSec(r.finalMs).padStart(8)}  ` +
      `traj=${String(r.trajectory.length).padStart(3)}  ` +
      `logProd=${board.features.logProd.toFixed(2).padStart(6)}  ` +
      `${slotStr0}  open=${board.features.nOpen}  ` +
      `[budget ${fmtSec(remaining)} left]`);
  }
  pool.shutdown();
  console.log(`\nDone: ${n} boards in ${fmtSec(performance.now() - startedAt)}.\n`);
}

// ----------------------------------------------------------------------------
// Report generator (works from the JSONL file alone)
// ----------------------------------------------------------------------------
function loadResults(){
  const txt = fs.readFileSync(RESULTS_PATH, 'utf8');
  const out = [];
  for (const line of txt.split('\n')){
    if (!line.trim()) continue;
    try {
      const r = JSON.parse(line);
      // Recompute features at load time so adding features doesn't require re-running
      // the bench. The stored features become advisory.
      r.features = patchFeatures(featuresFor(r.state, r.slots));
      if (!r.features.infeasible) out.push(r);
    } catch (e) {}
  }
  return out;
}

let lastPriorMethods = null;
function reportPrior(history, evalMinMs, lines){
  const nEval = history.filter(h => h.finalMs >= evalMinMs).length;
  lines.push('');
  lines.push('============================================================');
  lines.push('PRIOR PREDICTION (predict finalMs from features before solve)');
  lines.push('============================================================');
  lines.push(`History: ${history.length} boards; scored on ${nEval} solves with finalMs >= ${evalMinMs}ms.`);
  lines.push('Lower meanAbsLog is better; |log err|=0.7 ≈ 2x off, 1.1 ≈ 3x off.');
  lines.push('within2x = fraction of preds inside [truth/2, truth*2].');
  lines.push('');
  const methods = [];
  methods.push(['mean (baseline)', train => prMean(train)]);
  for (const k of FEATURE_KEYS){
    methods.push([`kernel(${k})`,    train => prKernel1(train, k)]);
    methods.push([`loglinear(${k})`, train => prLogLinear1(train, k)]);
  }
  // ridge / kNN on various feature subsets
  const groups = {
    'core4':   ['logProd', 'logTopRange', 'nOpen', 'density'],
    'core8':   ['logProd', 'logSum', 'logTopRange', 'logMinPl', 'nOpen', 'density', 'topArea', 'nPrizes'],
    'core9adj': ['logProd', 'logProdAdj', 'logFactSum', 'effBranching', 'logTopRange', 'nOpen', 'density', 'topArea', 'nPrizes'],
    'all':     FEATURE_KEYS,
    'split':   ['logProdTop', 'logProdRest', 'logTopRange', 'nOpen', 'density', 'topArea'],
    'geom':    ['logProdAdj', 'logTopRange', 'nOpen', 'density', 'openComponents', 'logBboxArea', 'bboxFill', 'effBranching'],
  };
  for (const [gname, keys] of Object.entries(groups)){
    for (const lam of [0.001, 0.01, 0.1, 1.0]){
      methods.push([`ridge(${gname},λ=${lam})`, train => prRidge(train, keys, lam)]);
    }
    for (const k of [3, 5, 10]){
      methods.push([`knn(${gname},k=${k})`, train => prKNN(train, keys, k)]);
    }
  }
  // polynomial / local / stacking — only on the more compact key sets so we
  // don't blow out the feature dimension relative to history size
  for (const gname of ['core4', 'core8', 'core9adj']){
    const keys = groups[gname];
    for (const lam of [0.01, 0.1, 1.0]) methods.push([`ridgePoly(${gname},λ=${lam})`, train => prRidgePoly(train, keys, lam)]);
    for (const bw of [0.5, 1.0, 2.0]) methods.push([`localRidge(${gname},bw=${bw})`, train => prLocalRidge(train, keys, 0.1, bw)]);
  }
  for (const lam of [0.01, 0.1, 1.0]) methods.push([`stacking(all,λ=${lam})`, train => prStacking(train, FEATURE_KEYS, lam)]);
  // Bias-corrected variants of the strongest base methods
  methods.push(['biasCorr[ridge(all,λ=0.001)]', train => prBiasCorrect(t => prRidge(t, FEATURE_KEYS, 0.001), train)]);
  methods.push(['biasCorr[ridge(core9adj,λ=0.01)]', train => prBiasCorrect(t => prRidge(t, groups.core9adj, 0.01), train)]);
  methods.push(['biasCorr[ridgePoly(core9adj,λ=0.1)]', train => prBiasCorrect(t => prRidgePoly(t, groups.core9adj, 0.1), train)]);
  // Training-threshold variants: train ONLY on solves above a floor (avoids
  // letting sub-second noise dominate the fit). Applied uniformly to all
  // leading method families so the comparison is fair.
  const filtFamilies = [
    ['ridge(all,λ=0.001)',         FEATURE_KEYS, 0.001, train => prRidge(train, FEATURE_KEYS, 0.001)],
    ['ridge(core9adj,λ=0.01)',     groups.core9adj, 0.01, train => prRidge(train, groups.core9adj, 0.01)],
    ['ridge(core9adj,λ=0.001)',    groups.core9adj, 0.001, train => prRidge(train, groups.core9adj, 0.001)],
    ['ridge(geom,λ=0.001)',        groups.geom, 0.001, train => prRidge(train, groups.geom, 0.001)],
    ['ridgePoly(core9adj,λ=0.01)', groups.core9adj, 0.01, train => prRidgePoly(train, groups.core9adj, 0.01)],
    ['ridgePoly(core9adj,λ=0.1)',  groups.core9adj, 0.1, train => prRidgePoly(train, groups.core9adj, 0.1)],
    ['localRidge(core9adj,bw=1)',  groups.core9adj, null, train => prLocalRidge(train, groups.core9adj, 0.1, 1.0)],
  ];
  for (const minMs of [500, 2000, 5000]){
    const filter = train => train.filter(h => h.finalMs >= minMs);
    for (const [label, _keys, _lam, build] of filtFamilies){
      methods.push([`${label}|train≥${minMs}ms`, train => build(filter(train))]);
    }
    // Bias-corrected + filtered (the user's "uniformly off in one direction" check)
    methods.push([`biasCorr[ridge(all,λ=0.001)]|train≥${minMs}ms`,
      train => prBiasCorrect(t => prRidge(t, FEATURE_KEYS, 0.001), filter(train))]);
  }
  lastPriorMethods = methods;
  const results = [];
  for (const [name, build] of methods){
    const r = evalPrior(history, name, build, evalMinMs);
    if (r) results.push(r);
  }
  results.sort((a,b) => a.meanAbsLog - b.meanAbsLog);
  lines.push('rank  meanAbsLog  med×    [p10..p90]            within2x  within3x  method');
  lines.push('----  ----------  ------  -------------------   --------  --------  ------');
  results.forEach((r,i) => {
    const r10 = r.p10Ratio.toFixed(2), r90 = r.p90Ratio.toFixed(2);
    lines.push(
      String(i+1).padStart(4) + '  ' +
      r.meanAbsLog.toFixed(3).padStart(10) + '  ' +
      r.medRatio.toFixed(2).padStart(6) + '  ' +
      `[${r10.padStart(5)}..${r90.padStart(5)}]   ` +
      (r.within2x*100).toFixed(1).padStart(6) + '%  ' +
      (r.within3x*100).toFixed(1).padStart(6) + '%  ' +
      r.name
    );
  });
  return results;
}

function reportLive(history, bestPrior, evalMinMs, lines){
  const evalSet = history.filter(h => h.finalMs >= evalMinMs);
  lines.push('');
  lines.push('============================================================');
  lines.push('LIVE PREDICTION (estimate REMAINING ms during the solve)');
  lines.push('============================================================');
  lines.push(`Trained on ${history.length} boards; scored on ${evalSet.length} solves with finalMs >= ${evalMinMs}ms.`);
  lines.push(`"prior" methods are seeded by the best prior predictor: ${bestPrior ? bestPrior.name : 'none'}`);
  lines.push('Three metrics per (method × phase):');
  lines.push('  absLog = mean |log((predR + 1s)/(trueR + 1s))|  — scale-invariant; treats 30s on a 1m solve same as 5m on a 10m solve');
  lines.push('  absRel = mean |predR - trueR| / (trueR + finalMs*0.05)  — relative error vs truth with a floor');
  lines.push('  absSec = mean |predR - trueR| in seconds (raw)');
  lines.push('Methods are ranked by overall absLog (best balance across phases).');
  lines.push('');
  const buckets = [
    [0.00, 0.10, 'early(0-10%)'],
    [0.10, 0.25, 'ramp(10-25%)'],
    [0.25, 0.50, 'mid (25-50%)'],
    [0.50, 0.75, 'late(50-75%)'],
    [0.75, 1.00, 'final(75-100%)'],
  ];
  // method -> bucket -> { sumLog, sumRel, sumSec, n } — populated by runReplay below.
  // Per-prior-length calibration warps + live-residual variance tables.
  // Filter training trajectories to solves ≥ CALIB_MIN_MS — sub-second solves
  // have near-zero-length trajectories whose progress points are all (≈0, ≈1)
  // and pollute the warp / variance estimates. Same fairness as the prior side.
  const CALIB_MIN_MS = 2000;
  const calibTrain = history.filter(h => h.finalMs >= CALIB_MIN_MS);
  const calibFnPick = buildCalibBuckets(calibTrain, ALPHA_CALIB);
  const calibFn = buildCalib(calibTrain);
  const liveLogVar    = buildLiveLogVar2d(calibTrain, calibFnPick, false, ALPHA_VAR);
  const liveLogVarMin = buildLiveLogVar2d(calibTrain, calibFnPick, true,  ALPHA_VAR);
  const priorLogVar = bestPrior ? priorLogResidualVar(history, bestPrior.build, evalMinMs) : 0.06;
  lines.push(`σ²_prior = ${priorLogVar.toFixed(4)}  (log-residual variance of the prior model on LOO)`);
  lines.push(`Calibration/variance trained on ${calibTrain.length} solves with finalMs ≥ ${CALIB_MIN_MS}ms`);
  lines.push(`Bayesian shrinkage α: bias=${ALPHA_BIAS}, var=${ALPHA_VAR}, calib=${ALPHA_CALIB} (α=0 = legacy hard-cutoff fallbacks)`);
  lines.push('');

  // First pass: build per-method per-bucket BIAS tables (1D and 2D).
  // (TIME_BINS / timeBinOf are module-scoped — shared with calibration.)
  function runReplay(biasFor, biasFor2d){
    const stats0 = new Map();
    const biasStats = new Map();     // 1D: method -> phase-bucket -> {sum,n}
    const biasStats2d = new Map();   // 2D: method -> "phaseB:timeB" -> {sum,n}
    for (let i = 0; i < history.length; i++){
      const rec = history[i];
      if (rec.finalMs < evalMinMs) continue;
      const train = history.slice(0, i).concat(history.slice(i+1));
      let priorMs = null;
      if (bestPrior){
        const pred = bestPrior.build(train);
        if (pred) priorMs = pred(rec);
      }
      const replay = liveStratEtas(rec.trajectory, rec.finalMs, priorMs, {
        calibFn, calibFnBucketed: calibFnPick, liveLogVar, liveLogVarMin, priorLogVar,
        biasFor, biasFor2d,
      });
      for (const pt of replay){
        const truePhase = pt.e / rec.finalMs;
        let b = -1;
        for (let k = 0; k < buckets.length; k++) if (truePhase >= buckets[k][0] && truePhase < buckets[k][1]) { b = k; break; }
        if (b < 0) continue;
        const predPhase = priorMs ? Math.min(0.999, pt.e / Math.max(1, priorMs)) : 0;
        let pBin = -1;
        for (let k = 0; k < buckets.length; k++) if (predPhase >= buckets[k][0] && predPhase < buckets[k][1]) { pBin = k; break; }
        const tBin = timeBinOf(priorMs);
        for (const [name, est] of Object.entries(pt.ests)){
          if (est == null || !isFinite(est)) continue;
          add0(stats0, name, b, est, pt.trueR, rec.finalMs);
          if (pBin < 0) continue;
          const totPred = est + pt.e;
          const totTrue = rec.finalMs;
          const r = Math.log(Math.max(1, totPred)) - Math.log(Math.max(1, totTrue));
          // 1D
          let bs = biasStats.get(name);
          if (!bs){ bs = buckets.map(() => ({ sum: 0, n: 0 })); biasStats.set(name, bs); }
          bs[pBin].sum += r; bs[pBin].n++;
          // 2D
          if (tBin >= 0){
            let bs2 = biasStats2d.get(name);
            if (!bs2){ bs2 = {}; biasStats2d.set(name, bs2); }
            const key = pBin + ':' + tBin;
            const slot = bs2[key] || (bs2[key] = { sum: 0, n: 0 });
            slot.sum += r; slot.n++;
          }
        }
      }
    }
    return { stats: stats0, biasStats, biasStats2d };
  }
  function add0(stats0, name, b, predR, trueR, finalMs){
    let m = stats0.get(name);
    if (!m){ m = buckets.map(() => ({ sumLog: 0, sumRel: 0, sumSec: 0, n: 0 })); stats0.set(name, m); }
    const slot = m[b];
    const offset = 1000;
    slot.sumLog += Math.abs(Math.log((predR + offset) / (trueR + offset)));
    const denom = Math.max(trueR, finalMs * 0.05);
    slot.sumRel += Math.abs(predR - trueR) / Math.max(1, denom);
    slot.sumSec += Math.abs(predR - trueR);
    slot.n++;
  }

  // Pass 1: measure bias of every method per phase (and per phase × log-priorMs)
  const pass1 = runReplay(() => 0, null);
  const biasTable = new Map();
  for (const [name, arr] of pass1.biasStats){
    // Global = method's mean log-residual pooled across all phase buckets.
    let gSum = 0, gN = 0;
    for (const x of arr){ gSum += x.sum; gN += x.n; }
    const gMean = gN > 0 ? gSum / gN : 0;
    const A1 = Math.max(0, ALPHA_BIAS);
    biasTable.set(name, arr.map(x => {
      if (A1 <= 0) return x.n >= 5 ? x.sum / x.n : 0;
      const cell = x.n > 0 ? x.sum / x.n : 0;
      return (x.n * cell + A1 * gMean) / (x.n + A1);
    }));
  }
  // 2D bias table: per (method, phaseBin, timeBin). With α>0, each cell mean is
  // shrunk toward the 1D per-phase fallback by pseudocount α (Bayesian shrinkage).
  // α=0 reproduces the hard-cutoff behavior (cell mean above n≥10, 1D below).
  const biasTable2d = new Map();
  const A = Math.max(0, ALPHA_BIAS);
  for (const [name, obj] of pass1.biasStats2d){
    const oneD = biasTable.get(name) || [];
    const tab = {};
    for (let p = 0; p < buckets.length; p++) for (let t = 0; t < TIME_BINS.length; t++){
      const k = p + ':' + t;
      const slot = obj[k];
      const fallback = oneD[p] || 0;
      if (A <= 0){
        tab[k] = (slot && slot.n >= 10) ? slot.sum / slot.n : fallback;
      } else {
        const n = slot ? slot.n : 0;
        const cellMean = n > 0 ? slot.sum / n : 0;
        tab[k] = (n * cellMean + A * fallback) / (n + A);
      }
    }
    biasTable2d.set(name, tab);
  }
  // Pass 2: apply per-phase bias correction (1D and 2D)
  const biasFor1d = (name, predPhase) => {
    const arr = biasTable.get(name);
    if (!arr) return 0;
    let pBin = -1;
    for (let k = 0; k < buckets.length; k++) if (predPhase >= buckets[k][0] && predPhase < buckets[k][1]) { pBin = k; break; }
    return pBin >= 0 ? arr[pBin] : 0;
  };
  const biasFor2dFn = (name, predPhase, priorMs) => {
    const tab = biasTable2d.get(name);
    if (!tab) return 0;
    let pBin = -1;
    for (let k = 0; k < buckets.length; k++) if (predPhase >= buckets[k][0] && predPhase < buckets[k][1]) { pBin = k; break; }
    const tBin = timeBinOf(priorMs);
    if (pBin < 0 || tBin < 0) return biasFor1d(name, predPhase);
    return tab[pBin + ':' + tBin] || 0;
  };
  const stats = runReplay(biasFor1d, biasFor2dFn).stats;
  // `stats` was populated by runReplay above (pass 2 with bias correction).

  function overallOf(name, key){
    const m = stats.get(name);
    let s = 0, n = 0;
    for (const slot of m){ s += slot[key]; n += slot.n; }
    return n ? s / n : null;
  }
  function maxBucketOf(name, key){
    const m = stats.get(name);
    let mx = 0;
    for (const slot of m){ if (slot.n){ const v = slot[key]/slot.n; if (v > mx) mx = v; } }
    return mx;
  }

  const methodNames = Array.from(stats.keys()).filter(n => stats.get(n).some(s => s.n > 0));
  methodNames.sort((a,b) => (overallOf(a, 'sumLog') ?? Infinity) - (overallOf(b, 'sumLog') ?? Infinity));

  // ---- Table 1: absLog ----
  lines.push('--- absLog (lower is better; ~0.3 ≈ 35% off, 0.7 ≈ 2x off) — ranked by overall ---');
  const headLog = 'method'.padEnd(26) + ' overall  worst  ' + buckets.map(([,,n]) => n.padStart(12)).join(' ');
  lines.push(headLog);
  lines.push('-'.repeat(headLog.length));
  for (const name of methodNames){
    const m = stats.get(name);
    const ov = overallOf(name, 'sumLog');
    const wx = maxBucketOf(name, 'sumLog');
    const cells = m.map(x => x.n ? (x.sumLog / x.n).toFixed(3).padStart(12) : '         -- ');
    lines.push(name.padEnd(26) + ' ' + (ov==null?'  -- ':ov.toFixed(3).padStart(6))
      + '  ' + wx.toFixed(3).padStart(5) + '  ' + cells.join(' '));
  }
  // ---- Table 2: absRel ----
  lines.push('');
  lines.push('--- absRel (mean |Δ|/max(trueR, 5% finalMs); 0.5 ≈ off by half-the-truth) ---');
  const headRel = 'method'.padEnd(26) + ' overall  worst  ' + buckets.map(([,,n]) => n.padStart(12)).join(' ');
  lines.push(headRel);
  lines.push('-'.repeat(headRel.length));
  for (const name of methodNames){
    const m = stats.get(name);
    const ov = overallOf(name, 'sumRel');
    const wx = maxBucketOf(name, 'sumRel');
    const cells = m.map(x => x.n ? (x.sumRel / x.n).toFixed(3).padStart(12) : '         -- ');
    lines.push(name.padEnd(26) + ' ' + (ov==null?'  -- ':ov.toFixed(3).padStart(6))
      + '  ' + wx.toFixed(3).padStart(5) + '  ' + cells.join(' '));
  }
  // ---- Table 3: absSec (legacy / sanity) ----
  lines.push('');
  lines.push('--- absSec (raw seconds; dominated by long solves) ---');
  const headSec = 'method'.padEnd(26) + ' overall ' + buckets.map(([,,n]) => n.padStart(14)).join(' ');
  lines.push(headSec);
  lines.push('-'.repeat(headSec.length));
  for (const name of methodNames){
    const m = stats.get(name);
    const ov = overallOf(name, 'sumSec');
    const cells = m.map(x => x.n ? (x.sumSec / x.n / 1000).toFixed(2).padStart(12) + 's' : '         --  ');
    lines.push(name.padEnd(26) + ' ' + (ov==null?'  -- ':(ov/1000).toFixed(2).padStart(6) + 's')
      + ' ' + cells.join(' '));
  }
  // ---- Table 4: signed bias (mean log-residual of total-time pred, by predicted phase) ----
  // Positive = systematically OVERpredict total time (predR too large).
  // Negative = systematically UNDERpredict.  |x| > ~0.1 ≈ 10% off uniformly.
  // The *_bc methods already subtract these; this table tells you whether
  // adding a *_bc variant for a method would help.
  lines.push('');
  lines.push('--- bias: mean(log(totPred) − log(totTrue)), by PREDICTED phase (e/priorMs) ---');
  lines.push('--- Positive=overpredict total; |x|>0.05 worth correcting ---');
  const headBias = 'method'.padEnd(26) + '  ' + buckets.map(([,,n]) => n.padStart(12)).join(' ');
  lines.push(headBias);
  lines.push('-'.repeat(headBias.length));
  // sort methods by overall absLog (same order as table 1)
  for (const name of methodNames){
    const arr = pass1.biasStats.get(name);
    if (!arr) continue;
    const cells = arr.map(x => x.n >= 5 ? (x.sum/x.n).toFixed(3).padStart(12) : '         -- ');
    lines.push(name.padEnd(26) + '  ' + cells.join(' '));
  }
}

function generateReport(){
  const all = loadResults();
  // only use solves long enough that startup overhead doesn't dominate; tune by feel
  const history = all.filter(r => r.finalMs >= 50);
  console.log(`Loaded ${all.length} solves; ${history.length} usable (finalMs >= 50ms).`);
  const lines = [];
  lines.push(`Bench report — ${new Date().toISOString()}`);
  lines.push(`Pool size: ${POOL_SIZE}; total solves: ${all.length}; used: ${history.length}`);
  // distribution
  const sorted = history.map(h => h.finalMs).sort((a,b) => a-b);
  if (sorted.length){
    const pct = q => sorted[Math.floor(sorted.length * q)];
    lines.push(`finalMs:  min ${fmtSec(sorted[0])}  p25 ${fmtSec(pct(0.25))}  median ${fmtSec(pct(0.5))}  p75 ${fmtSec(pct(0.75))}  p95 ${fmtSec(pct(0.95))}  max ${fmtSec(sorted[sorted.length-1])}`);
  }
  if (history.length < 5){
    lines.push('Not enough data for analysis.');
    const txt = lines.join('\n');
    fs.writeFileSync(REPORT_PATH, txt);
    console.log(txt);
    return;
  }
  const priorResults = reportPrior(history, EVAL_MIN_MS, lines);
  // The reportPrior call exposes the methods table via `lastPriorMethods` so we
  // don't need to re-parse names from strings.
  let bestPrior = null;
  if (priorResults.length && lastPriorMethods){
    const top = priorResults[0];
    const entry = lastPriorMethods.find(([n]) => n === top.name);
    if (entry) bestPrior = { name: top.name, build: entry[1] };
  }
  reportLive(history, bestPrior, EVAL_MIN_MS, lines);
  const txt = lines.join('\n');
  fs.writeFileSync(REPORT_PATH, txt);
  console.log('\n' + txt);
  console.log(`\nWrote ${REPORT_PATH}`);
}

// ----------------------------------------------------------------------------
function shrinkResults(){
  // Rewrite bench-results.jsonl in place with rounded trajectories.
  // Backup yourself first — this clobbers. Rounds: e→int ms, pcts/nps→4 decimals.
  const round4 = x => (typeof x === 'number' && isFinite(x)) ? Math.round(x * 10000) / 10000 : x;
  const tmp = RESULTS_PATH + '.shrink.tmp';
  const txt = fs.readFileSync(RESULTS_PATH, 'utf8');
  const lines = txt.split('\n');
  let kept = 0, dropped = 0;
  const outLines = [];
  for (const line of lines){
    if (!line.trim()) continue;
    try {
      const r = JSON.parse(line);
      if (Array.isArray(r.trajectory)){
        r.trajectory = r.trajectory.map(pt => ({
          e: Math.round(pt.e),
          pcts: (pt.pcts || []).map(round4),
          nps:  (pt.nps  || []).map(round4),
        }));
      }
      if (typeof r.finalMs === 'number') r.finalMs = Math.round(r.finalMs * 100) / 100;
      if (typeof r.wallMs  === 'number') r.wallMs  = Math.round(r.wallMs  * 100) / 100;
      if (Array.isArray(r.workerMs)) r.workerMs = r.workerMs.map(v => Math.round(v * 100) / 100);
      outLines.push(JSON.stringify(r));
      kept++;
    } catch (e){ dropped++; }
  }
  fs.writeFileSync(tmp, outLines.join('\n') + '\n');
  const beforeSize = fs.statSync(RESULTS_PATH).size;
  const afterSize  = fs.statSync(tmp).size;
  fs.renameSync(tmp, RESULTS_PATH);
  console.log(`Shrink complete: ${kept} kept, ${dropped} dropped.`);
  console.log(`Size: ${(beforeSize/1024/1024).toFixed(2)} MB → ${(afterSize/1024/1024).toFixed(2)} MB (${(100*(1-afterSize/beforeSize)).toFixed(1)}% smaller)`);
}

// ----------------------------------------------------------------------------
// Bayesian-shrinkage benchmark. Scores ONLY champSm80 (the shipping recipe)
// across a configurable α grid on three diagnostics:
//   1. α sweep on the full 80% training pool (sanity: should not regress)
//   2. Cold-start learning curve: train on first k records, score eval set
//   3. Held-out-time-bin: drop one TIME_BIN entirely from training
// Eval set = random 20% of solves ≥3s (seeded).
// ----------------------------------------------------------------------------
function scoreChampSm80(trainSet, evalSet, opts){
  opts = opts || {};
  const aBias = opts.aBias  != null ? opts.aBias  : 0;
  const aVar  = opts.aVar   != null ? opts.aVar   : 0;
  const aCal  = opts.aCalib != null ? opts.aCalib : 0;

  // Prior model — `ridge(top10, λ=0.001)` on solves ≥2s (or anything we have,
  // when scarce). No `n < p+2` guard — ridge with λ>0 degrades gracefully to
  // `ybar` (geomean of training finalMs) as n→1.
  const ridgeTrain = (() => {
    const big = trainSet.filter(h => h.finalMs >= 2000);
    return big.length >= 3 ? big : trainSet;     // fall back to all when sparse
  })();
  if (ridgeTrain.length < 1) return null;
  const priorPred = prRidge(ridgeTrain, FEATURE_KEYS_LEAN, 0.001);
  if (!priorPred) return null;

  // Calibration + 2D live variance. We *want* shrinkage to kick in here at low
  // n — that's the entire point. So just feed whatever training we have.
  const calibTrain = trainSet.filter(h => h.finalMs >= 2000);
  const calibFnPick = buildCalibBuckets(calibTrain, aCal);
  const liveLogVar = buildLiveLogVar2d(calibTrain, calibFnPick, false, aVar);

  // In-sample priorLogVar (cheap; LOO would be more honest but cost-prohibitive in the inner loop).
  let pn = 0, ps = 0, ps2 = 0;
  for (const h of ridgeTrain){
    const y = priorPred(h);
    if (!y || !isFinite(y) || y <= 0) continue;
    const r = Math.log(y) - Math.log(h.finalMs);
    ps += r; ps2 += r*r; pn++;
  }
  const priorLogVar = pn > 1 ? Math.max(0.005, ps2/pn - (ps/pn)**2) : 0.06;

  // Pass 1: build 2D bias table for bayesLogBlend across trainSet.
  const buckets = [[0,0.10],[0.10,0.25],[0.25,0.50],[0.50,0.75],[0.75,1.00]];
  const TB = TIME_BINS.length;
  const sums2 = Array.from({length:5}, () => new Array(TB).fill(0));
  const cnts2 = Array.from({length:5}, () => new Array(TB).fill(0));
  const sums1 = new Array(5).fill(0), cnts1 = new Array(5).fill(0);
  function pBinOf(p){ for (let k=0;k<5;k++) if (p>=buckets[k][0] && p<buckets[k][1]) return k; return -1; }
  function bayesTotalPred(rec, pt, priorMs){
    if (pt.e <= 0) return null;
    const avgP = pt.pcts.reduce((a,b)=>a+b,0)/pt.pcts.length;
    if (avgP <= 0 || avgP >= 1) return null;
    const cp = Math.max(0.001, Math.min(0.999, calibFnPick(priorMs)(avgP)));
    const liveTot = pt.e / cp;
    if (!isFinite(liveTot) || liveTot <= 0) return null;
    const lvar = Math.max(0.001, liveLogVar(avgP, priorMs));
    const pvar = Math.max(0.001, priorLogVar);
    const tauP = 1/pvar, tauL = 1/lvar;
    return (Math.log(priorMs)*tauP + Math.log(liveTot)*tauL) / (tauP + tauL); // returns muPost (log)
  }
  for (const rec of trainSet){
    const priorMs = priorPred(rec);
    if (!priorMs || !isFinite(priorMs) || priorMs <= 0) continue;
    const tBin = timeBinOf(priorMs);
    for (const pt of rec.trajectory){
      const mu = bayesTotalPred(rec, pt, priorMs);
      if (mu == null) continue;
      const r = mu - Math.log(Math.max(1, rec.finalMs));
      const predPhase = Math.min(0.999, pt.e / priorMs);
      const pb = pBinOf(predPhase);
      if (pb < 0) continue;
      sums1[pb] += r; cnts1[pb]++;
      if (tBin >= 0){ sums2[pb][tBin] += r; cnts2[pb][tBin]++; }
    }
  }
  let gSum = 0, gN = 0;
  for (let p=0; p<5; p++){ gSum += sums1[p]; gN += cnts1[p]; }
  const gMean = gN > 0 ? gSum/gN : 0;
  const bias1 = new Array(5);
  for (let p=0; p<5; p++){
    const n = cnts1[p];
    const cell = n > 0 ? sums1[p]/n : 0;
    bias1[p] = aBias > 0 ? (n*cell + aBias*gMean)/(n+aBias) : (n >= 5 ? cell : 0);
  }
  const bias2d = Array.from({length:5}, () => new Array(TB).fill(0));
  for (let p=0; p<5; p++) for (let t=0; t<TB; t++){
    const n = cnts2[p][t];
    const cell = n > 0 ? sums2[p][t]/n : 0;
    bias2d[p][t] = aBias > 0 ? (n*cell + aBias*bias1[p])/(n+aBias) : (n >= 10 ? cell : bias1[p]);
  }

  // Pass 2: score champSm80 on evalSet.
  let sumLog = 0, ntot = 0;
  const bSum = new Array(5).fill(0), bN = new Array(5).fill(0);
  for (const rec of evalSet){
    if (rec.finalMs < 3000) continue;
    const priorMs = priorPred(rec);
    if (!priorMs || !isFinite(priorMs) || priorMs <= 0) continue;
    const tBin = timeBinOf(priorMs);
    for (const pt of rec.trajectory){
      const e = pt.e;
      const trueR = rec.finalMs - e;
      if (trueR < 0) continue;
      const truePhase = e / rec.finalMs;
      let tb = -1; for (let k=0;k<5;k++) if (truePhase >= buckets[k][0] && truePhase < buckets[k][1]){ tb = k; break; }
      if (tb < 0) continue;
      const mu = bayesTotalPred(rec, pt, priorMs);
      if (mu == null) continue;
      const predPhase = Math.min(0.999, e / priorMs);
      const pb = pBinOf(predPhase);
      const b = (pb >= 0 && tBin >= 0) ? bias2d[pb][tBin] : (pb >= 0 ? bias1[pb] : 0);
      const bayesBC = Math.max(0, Math.exp(mu - b) - e);
      let predR = bayesBC;
      const minP = Math.min(...pt.pcts);
      const ctr = opts.center != null ? opts.center : 0.80;
      const wid = opts.width  != null ? opts.width  : 0.08;
      if (opts.noHandoff !== true && minP > 0){
        const gated = e * (1 - minP) / minP;
        const w = 1 / (1 + Math.exp(-(predPhase - ctr) / wid));
        predR = (1 - w) * bayesBC + w * gated;
      }
      const offset = 1000;
      const al = Math.abs(Math.log((predR + offset) / (trueR + offset)));
      sumLog += al; ntot++;
      bSum[tb] += al; bN[tb]++;
    }
  }
  if (!ntot) return null;
  let worst = 0;
  for (let k=0; k<5; k++) if (bN[k]){ const v = bSum[k]/bN[k]; if (v > worst) worst = v; }
  return { overall: sumLog/ntot, worst, n: ntot };
}

function runShrinkageBenchmark(){
  const all = loadResults();
  const history = all.filter(r => r.finalMs >= 50);
  console.log(`Loaded ${history.length} solves (finalMs ≥ 50ms).`);

  // Seeded deterministic shuffle (LCG)
  let seed = 0x13572468;
  function rng(){ seed = (Math.imul(seed, 1664525) + 1013904223) >>> 0; return seed / 0x100000000; }
  const shuffled = history.slice();
  for (let i = shuffled.length - 1; i > 0; i--){
    const j = Math.floor(rng() * (i + 1));
    [shuffled[i], shuffled[j]] = [shuffled[j], shuffled[i]];
  }
  const evalCandidates = shuffled.filter(h => h.finalMs >= 3000);
  const nEval = Math.floor(evalCandidates.length * 0.2);
  const evalSet = evalCandidates.slice(0, nEval);
  const evalIds = new Set(evalSet.map(h => h.i));
  const trainPool = shuffled.filter(h => !evalIds.has(h.i));
  console.log(`Eval set: ${evalSet.length} solves (random 20% of finalMs ≥ 3s).`);
  console.log(`Train pool: ${trainPool.length} solves.`);

  const ALPHAS = [0, 1, 5, 10, 20, 50, 100, 1e6];
  const KS = [1, 3, 5, 10, 25, 50, 100, 200, trainPool.length];

  console.log('\n=== α SWEEP on full 80% training pool ===');
  console.log('  α        overall   worst');
  for (const a of ALPHAS){
    const r = scoreChampSm80(trainPool, evalSet, { aBias: a, aVar: a, aCalib: a });
    const aStr = (a >= 1e5 ? '∞' : String(a)).padStart(5);
    console.log('  ' + aStr + '   ' + (r ? r.overall.toFixed(4).padStart(7) + '   ' + r.worst.toFixed(4) : '   --'));
  }

  console.log('\n=== COLD-START learning curve (train on first k of trainPool) ===');
  console.log('  k        α=0  ovr/worst        α=20 ovr/worst       Δoverall(20−0)');
  for (const k of KS){
    const tr = trainPool.slice(0, k);
    const r0  = scoreChampSm80(tr, evalSet, { aBias:0,  aVar:0,  aCalib:0  });
    const r20 = scoreChampSm80(tr, evalSet, { aBias:20, aVar:20, aCalib:20 });
    const s0  = r0  ? r0.overall.toFixed(4) + ' / ' + r0.worst.toFixed(4) : '  -- ';
    const s20 = r20 ? r20.overall.toFixed(4) + ' / ' + r20.worst.toFixed(4) : '  -- ';
    const dlt = (r0 && r20) ? (r20.overall - r0.overall).toFixed(4) : '  -- ';
    console.log('  ' + String(k).padStart(5) + '    ' + s0.padEnd(20) + s20.padEnd(20) + dlt);
  }

  console.log('\n=== HELD-OUT TIME BIN (train on all but one bin; eval on that bin) ===');
  const heldAlphas = [0, 5, 10, 20, 50, 100];
  let hdr = '  bin           n_eval  ' + heldAlphas.map(a => ('α='+a).padStart(8)).join(' ');
  console.log(hdr + '   (overall absLog)');
  for (let t = 0; t < TIME_BINS.length; t++){
    const bin = TIME_BINS[t];
    const tr = history.filter(h => timeBinOf(h.finalMs) !== t);
    const ev = history.filter(h => timeBinOf(h.finalMs) === t && h.finalMs >= 3000);
    if (ev.length === 0){
      console.log('  ' + bin.name.padEnd(12) + '   ' + String(0).padStart(5) + '   (no eval samples)');
      continue;
    }
    const row = heldAlphas.map(a => {
      const r = scoreChampSm80(tr, ev, { aBias:a, aVar:a, aCalib:a });
      return r ? r.overall.toFixed(4).padStart(8) : '    --  ';
    });
    console.log('  ' + bin.name.padEnd(12) + '   ' + String(ev.length).padStart(5) + '   ' + row.join(' '));
  }
}

// ----------------------------------------------------------------------------
// Full sweep: alpha (shrinkage) x sigmoid (center, width) x subset size.
// Goal: confirm the recipe across data regimes typical users will have, not
// just the 419-solve bench pool. Reports best (center, width, alpha) per
// subset size so we can decide whether to ship n-adaptive sigmoid params.
// ----------------------------------------------------------------------------
function runFullSweep(){
  const all = loadResults();
  const history = all.filter(r => r.finalMs >= 50);
  console.log('Loaded ' + history.length + ' solves (finalMs >= 50ms).');

  // Same seeded split as runShrinkageBenchmark, for comparability.
  let seed = 0x13572468;
  function rng(){ seed = (Math.imul(seed, 1664525) + 1013904223) >>> 0; return seed / 0x100000000; }
  const shuffled = history.slice();
  for (let i = shuffled.length - 1; i > 0; i--){
    const j = Math.floor(rng() * (i + 1));
    [shuffled[i], shuffled[j]] = [shuffled[j], shuffled[i]];
  }
  const evalCandidates = shuffled.filter(h => h.finalMs >= 3000);
  const nEval = Math.floor(evalCandidates.length * 0.2);
  const evalSet = evalCandidates.slice(0, nEval);
  const evalIds = new Set(evalSet.map(h => h.i));
  const trainPool = shuffled.filter(h => !evalIds.has(h.i));
  console.log('Eval: ' + evalSet.length + '   Train pool: ' + trainPool.length);

  const ALPHAS  = [0, 5, 10, 20, 50, 100];
  const CENTERS = [0.70, 0.75, 0.80, 0.85, 0.90, 0.95];
  const WIDTHS  = [0.05, 0.08, 0.12, 0.20, 0.30];
  const KS      = [10, 25, 50, 100, 200, trainPool.length];

  function scoreAt(tr, alpha, opts){
    return scoreChampSm80(tr, evalSet,
      Object.assign({ aBias: alpha, aVar: alpha, aCalib: alpha }, opts || {}));
  }

  // PASS 1: full pool. Best sigmoid given alpha, best alpha given sigmoid.
  console.log('\n=== FULL POOL, sigmoid sweep at each alpha (overall absLog) ===');
  console.log('         ' + WIDTHS.map(w => ('w=' + w.toFixed(2)).padStart(8)).join('  ') + '    no-handoff');
  let bestFull = { score: Infinity, alpha: null, ctr: null, wid: null, noHand: false };
  for (const alpha of ALPHAS){
    console.log('--- alpha=' + alpha + ' ---');
    for (const ctr of CENTERS){
      const row = WIDTHS.map(wid => {
        const r = scoreAt(trainPool, alpha, { center: ctr, width: wid });
        if (r && r.overall < bestFull.score) bestFull = { score: r.overall, alpha, ctr, wid, noHand: false };
        return r ? r.overall.toFixed(4).padStart(8) : '    --  ';
      });
      const rNH = scoreAt(trainPool, alpha, { noHandoff: true });
      if (rNH && rNH.overall < bestFull.score) bestFull = { score: rNH.overall, alpha, ctr: null, wid: null, noHand: true };
      const nhStr = rNH ? rNH.overall.toFixed(4) : '  --  ';
      console.log('  c=' + ctr.toFixed(2) + '  ' + row.join('  ') + '    ' + nhStr);
    }
  }
  console.log('\nFULL POOL best: alpha=' + bestFull.alpha
    + (bestFull.noHand ? '  no-handoff' : '  center=' + bestFull.ctr + '  width=' + bestFull.wid)
    + '  -> absLog ' + bestFull.score.toFixed(4));

  // PASS 2: subset sweep. For each subset size k, sweep (alpha, ctr, wid) and
  // also score the FULL-POOL champ on this subset, so we can see how much
  // leaning the parameters per regime would actually buy.
  console.log('\n=== SUBSET SWEEP: best (alpha, center, width) per training size ===');
  console.log('  k       full-champ      best-tuned                                  no-handoff (best a)');
  const subsetRows = [];
  for (const k of KS){
    const tr = trainPool.slice(0, k);
    const rFull = bestFull.noHand
      ? scoreAt(tr, bestFull.alpha, { noHandoff: true })
      : scoreAt(tr, bestFull.alpha, { center: bestFull.ctr, width: bestFull.wid });
    let best = { score: Infinity, alpha: null, ctr: null, wid: null };
    for (const alpha of ALPHAS) for (const ctr of CENTERS) for (const wid of WIDTHS){
      const r = scoreAt(tr, alpha, { center: ctr, width: wid });
      if (r && r.overall < best.score) best = { score: r.overall, alpha, ctr, wid };
    }
    let bestNH = { score: Infinity, alpha: null };
    for (const alpha of ALPHAS){
      const r = scoreAt(tr, alpha, { noHandoff: true });
      if (r && r.overall < bestNH.score) bestNH = { score: r.overall, alpha };
    }
    const fullStr = rFull ? rFull.overall.toFixed(4) : '  --  ';
    const bestStr = best.score < Infinity
      ? best.score.toFixed(4) + '  (a=' + best.alpha + ', c=' + best.ctr.toFixed(2) + ', w=' + best.wid.toFixed(2) + ')'
      : '  --';
    const nhStr = bestNH.score < Infinity
      ? bestNH.score.toFixed(4) + '  (a=' + bestNH.alpha + ')'
      : '  --';
    console.log('  ' + String(k).padStart(5) + '   ' + fullStr.padEnd(14) + '  ' + bestStr.padEnd(40) + nhStr);
    subsetRows.push({ k, full: rFull && rFull.overall, best, bestNH });
  }

  // PASS 3: held-out time bin x (alpha, center, width).
  console.log('\n=== HELD-OUT TIME BIN: best (alpha, center, width) per bin ===');
  console.log('  bin           n_eval  full-champ   best-tuned                                  no-handoff (best a)');
  for (let t = 0; t < TIME_BINS.length; t++){
    const bin = TIME_BINS[t];
    const tr = history.filter(h => timeBinOf(h.finalMs) !== t);
    const ev = history.filter(h => timeBinOf(h.finalMs) === t && h.finalMs >= 3000);
    if (ev.length === 0){
      console.log('  ' + bin.name.padEnd(12) + '   ' + String(0).padStart(5) + '   (no eval samples)');
      continue;
    }
    function scoreOnEv(alpha, sigOpts){
      return scoreChampSm80(tr, ev,
        Object.assign({ aBias: alpha, aVar: alpha, aCalib: alpha }, sigOpts));
    }
    const rFull = bestFull.noHand
      ? scoreOnEv(bestFull.alpha, { noHandoff: true })
      : scoreOnEv(bestFull.alpha, { center: bestFull.ctr, width: bestFull.wid });
    let best = { score: Infinity, alpha: null, ctr: null, wid: null };
    for (const alpha of ALPHAS) for (const ctr of CENTERS) for (const wid of WIDTHS){
      const r = scoreOnEv(alpha, { center: ctr, width: wid });
      if (r && r.overall < best.score) best = { score: r.overall, alpha, ctr, wid };
    }
    let bestNH = { score: Infinity, alpha: null };
    for (const alpha of ALPHAS){
      const r = scoreOnEv(alpha, { noHandoff: true });
      if (r && r.overall < bestNH.score) bestNH = { score: r.overall, alpha };
    }
    const fullStr = rFull ? rFull.overall.toFixed(4) : '  --  ';
    const bestStr = best.score < Infinity
      ? best.score.toFixed(4) + '  (a=' + best.alpha + ', c=' + best.ctr.toFixed(2) + ', w=' + best.wid.toFixed(2) + ')'
      : '  --';
    const nhStr = bestNH.score < Infinity
      ? bestNH.score.toFixed(4) + '  (a=' + bestNH.alpha + ')'
      : '  --';
    console.log('  ' + bin.name.padEnd(12) + '   ' + String(ev.length).padStart(5) + '   ' + fullStr.padEnd(12) + ' ' + bestStr.padEnd(40) + nhStr);
  }
}

// ----------------------------------------------------------------------------
// Feature audit. For the leaderboard prior `ridge(all, λ=0.001) | train≥2000ms`:
//   1. Per-feature (mean, std) so we can spot zero-variance / useless features.
//   2. Standardized ridge coefficients (magnitude = importance under standardization).
//   3. Leave-one-feature-out ΔabsLog vs the full feature set.
// ----------------------------------------------------------------------------
function runFeatureAudit(){
  const all = loadResults();
  const history = all.filter(r => r.finalMs >= 50);
  const train = history.filter(h => h.finalMs >= 2000);
  console.log(`History: ${history.length}; ridge train (≥2s): ${train.length}.`);
  console.log('\n--- per-feature stats (mean, std, range) ---');
  for (const k of FEATURE_KEYS){
    let mn = Infinity, mx = -Infinity, s = 0, s2 = 0, n = 0;
    for (const h of train){
      const v = h.features[k];
      if (!Number.isFinite(v)) continue;
      if (v < mn) mn = v;
      if (v > mx) mx = v;
      s += v; s2 += v*v; n++;
    }
    if (n === 0){ console.log(k.padEnd(18) + ' (no finite values)'); continue; }
    const mean = s/n, vari = s2/n - mean*mean;
    const std = Math.sqrt(Math.max(0, vari));
    const flag = std < 1e-9 ? '  ZERO-VAR (useless)' : (mx === mn ? '  CONST' : '');
    console.log(k.padEnd(18) + ` mean=${mean.toFixed(3).padStart(8)}  std=${std.toFixed(3).padStart(7)}  range=[${mn.toFixed(2)}..${mx.toFixed(2)}]${flag}`);
  }

  console.log('\n--- standardized ridge coefficients (|coef| ≈ feature importance) ---');
  // re-derive what prRidge computes, then print weights.
  const keys = FEATURE_KEYS;
  const n = train.length, p = keys.length;
  const means = keys.map(k => { let s = 0; for (const h of train) s += h.features[k]; return s / n; });
  const stds  = keys.map((k,j) => { let s = 0; for (const h of train) s += (h.features[k]-means[j])**2; return Math.sqrt(s/n) || 1; });
  const X = train.map(h => keys.map((k,j) => (h.features[k]-means[j])/stds[j]));
  const y = train.map(h => Math.log(h.finalMs));
  let ybar = 0; for (const v of y) ybar += v; ybar /= n;
  const XtX = Array.from({length:p}, () => new Array(p).fill(0));
  const Xty = new Array(p).fill(0);
  for (let i = 0; i < n; i++){
    const yc = y[i] - ybar;
    for (let a = 0; a < p; a++){
      Xty[a] += X[i][a] * yc;
      for (let b = 0; b < p; b++) XtX[a][b] += X[i][a] * X[i][b];
    }
  }
  for (let a = 0; a < p; a++) XtX[a][a] += 0.001 * n;
  const w = solveLinear(XtX, Xty);
  if (!w){ console.log('(ridge solve failed)'); return; }
  const ranked = keys.map((k,i) => ({ k, w: w[i] }))
    .sort((a,b) => Math.abs(b.w) - Math.abs(a.w));
  for (const r of ranked){
    const bar = '*'.repeat(Math.min(40, Math.round(Math.abs(r.w) * 40 / Math.abs(ranked[0].w))));
    console.log(r.k.padEnd(18) + ' ' + r.w.toFixed(3).padStart(7) + '  ' + bar);
  }

  console.log('\n--- leave-one-feature-out: full-set vs dropped (LOO meanAbsLog, scored on ≥3s) ---');
  const evalMs = 3000;
  const full = evalPrior(history, 'full', tr => prRidge(tr.filter(h=>h.finalMs>=2000), keys, 0.001), evalMs);
  console.log(`full set (${keys.length} feats): meanAbsLog=${full.meanAbsLog.toFixed(4)} within2x=${(full.within2x*100).toFixed(1)}%`);
  console.log('\n  feature dropped     ΔabsLog (positive = hurt when removed = important)');
  const rows = [];
  for (const drop of keys){
    const sub = keys.filter(k => k !== drop);
    const r = evalPrior(history, 'drop_'+drop, tr => prRidge(tr.filter(h=>h.finalMs>=2000), sub, 0.001), evalMs);
    if (!r) continue;
    rows.push({ drop, d: r.meanAbsLog - full.meanAbsLog, abs: r.meanAbsLog });
  }
  rows.sort((a,b) => b.d - a.d);
  for (const r of rows){
    const sign = r.d >= 0 ? '+' : '';
    console.log('  ' + r.drop.padEnd(18) + sign + r.d.toFixed(4).padStart(8) + '   (drop→' + r.abs.toFixed(4) + ')');
  }

  console.log('\n--- candidate compact feature sets (LOO absLog) ---');
  const sets = {
    'lean3':  ['logChooseFree', 'logProd', 'density'],
    'lean5':  ['logChooseFree', 'logProd', 'logSum', 'topShapeCount', 'density'],
    'lean7':  ['logChooseFree', 'logProd', 'logSum', 'topShapeCount', 'density', 'effBranching', 'logFactSum'],
    'top10':  ranked.slice(0, 10).map(r => r.k),
    'top6':   ranked.slice(0, 6).map(r => r.k),
    'top4':   ranked.slice(0, 4).map(r => r.k),
    'just1':  ['logChooseFree'],
  };
  for (const [name, ks] of Object.entries(sets)){
    for (const lam of [0.001, 0.01, 0.1, 1.0]){
      const r = evalPrior(history, name, tr => prRidge(tr.filter(h=>h.finalMs>=2000), ks, lam), evalMs);
      if (!r) continue;
      console.log(('  ' + name + ' (n=' + ks.length + ', λ=' + lam + ')').padEnd(28) + '  meanAbsLog=' + r.meanAbsLog.toFixed(4) + '  within2x=' + (r.within2x*100).toFixed(1) + '%');
    }
  }
}

(async function main(){
  if (SHRINK_ONLY){
    shrinkResults();
    return;
  }
  if (FEATURE_AUDIT){
    runFeatureAudit();
    return;
  }
  if (SHRINK_BENCH){
    runShrinkageBenchmark();
    return;
  }
  if (FULL_SWEEP){
    runFullSweep();
    return;
  }
  if (REPORT_ONLY){
    generateReport();
    return;
  }
  await runBench();
  generateReport();
  process.exit(0);
})().catch(e => { console.error(e); process.exit(1); });
