// Benchmark for the solver engine embedded in index.html.
//
//   node bench.js [games] [--threads=N]
//
// 1. Correctness: DP arrangement counts vs a brute-force enumerator on random
//    small boards with misses, hits and placed prizes.
// 2. Speed: DP time on every empty-board shape triple with counts 0 to
//    MAX_COUNT. The games below also report DP time per move.
// 3. Play quality: paired games on uniformly sampled hidden boards, the
//    page's policy (optimal flip when the search finishes within SEARCH_MS,
//    otherwise highest chance) against pure highest-chance play.
//
// Board mixes for the games: "low" draws 1-2 of each of three random shapes;
// "high" draws 1 to MAX_COUNT of each, rejecting mixes over 36 tiles of prize
// area.
// Writes bench-report.txt next to this file.
'use strict';
const { Worker, isMainThread, parentPort, workerData } = require('worker_threads');
const { performance } = require('perf_hooks');
const fs = require('fs');
const path = require('path');

const html = fs.readFileSync(path.join(__dirname, 'index.html'), 'utf8');
const engineSrc = html.match(/<script id="engine" type="text\/plain">([\s\S]*?)<\/script>/)[1];
const SEARCH_MS = +html.match(/const SEARCH_MS = (\d+);/)[1];
const MAX_COUNT = +html.match(/const MAX_COUNT = (\d+);/)[1];
const engine = new Function(engineSrc + '\nreturn { W, H, NCELLS, orientations, buildPlacements, analyze, makeSearch };')();
const { W, H, NCELLS, orientations, buildPlacements, analyze, makeSearch } = engine;

const SHAPES = ['1x2', '1x3', '1x4', '2x2', '2x3', '2x4', '3x3'];
const AREA = s => s.split('x').reduce((a, b) => a * b, 1);

function mulberry32(a){
  return function(){
    a |= 0; a = a + 0x6D2B79F5 | 0;
    let t = Math.imul(a ^ a >>> 15, 1 | a);
    t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t;
    return ((t ^ t >>> 14) >>> 0) / 4294967296;
  };
}
function pickShapes(rnd){
  const s = SHAPES.slice();
  for (let i = s.length - 1; i > 0; i--){ const j = Math.floor(rnd() * (i + 1)); [s[i], s[j]] = [s[j], s[i]]; }
  return s.slice(0, 3);
}

// ---- brute-force reference: every set of non-overlapping placements ----
function bruteForce(state, shapes, counts){
  const pls = buildPlacements(shapes);
  const free = pls.map(pl => pl.cells.every(i => state[i] !== 1 && state[i] !== 3));
  const cell = new Float64Array(NCELLS);
  const occ = new Uint8Array(NCELLS);
  const left = counts.slice();
  const picked = [];
  let total = 0;
  function rec(k){
    if (left.every(c => c === 0)){
      for (let i = 0; i < NCELLS; i++) if (state[i] === 2 && !occ[i]) return;
      total++;
      for (const id of picked) for (const i of pls[id].cells) cell[i]++;
      return;
    }
    for (let id = k; id < pls.length; id++){
      const pl = pls[id];
      if (!free[id] || left[pl.slot] === 0 || pl.cells.some(i => occ[i])) continue;
      for (const i of pl.cells) occ[i] = 1;
      left[pl.slot]--; picked.push(id);
      rec(id + 1);
      for (const i of pl.cells) occ[i] = 0;
      left[pl.slot]++; picked.pop();
    }
  }
  rec(0);
  return { total, cell };
}

function correctness(trials){
  const rnd = mulberry32(1);
  let checked = 0, bad = 0;
  while (checked < trials){
    const shapes = pickShapes(rnd);
    const counts = shapes.map(() => Math.floor(rnd() * 3));
    const state = new Int8Array(NCELLS);
    const marks = Math.floor(rnd() * 30);
    for (let k = 0; k < marks; k++){
      const x = rnd();
      state[Math.floor(rnd() * NCELLS)] = x < 0.75 ? 1 : x < 0.9 ? 2 : 3;
    }
    const pieces = counts.reduce((a, b) => a + b, 0);
    if (pieces > 4 || (pieces === 4 && marks < 15)) continue;
    const ref = bruteForce(state, shapes, counts);
    const got = analyze(buildPlacements(shapes), state, counts);
    const close = (a, b) => Math.abs(a - b) <= 1e-9 * Math.max(1, Math.abs(b));
    let ok = close(got.total, ref.total);
    for (let i = 0; i < NCELLS; i++) if (!close(got.cell[i], ref.cell[i])) ok = false;
    if (!ok) bad++;
    checked++;
  }
  return { checked, bad };
}

function quantiles(xs){
  const s = xs.slice().sort((a, b) => a - b);
  const q = p => s[Math.min(s.length - 1, Math.floor(p * s.length))];
  return { n: s.length, p50: q(0.5), p95: q(0.95), p99: q(0.99), max: s[s.length - 1] };
}

function speedEmptyBoards(){
  const times = [];
  let worst = null;
  const empty = new Int8Array(NCELLS);
  for (let a = 0; a < 7; a++) for (let b = a + 1; b < 7; b++) for (let c = b + 1; c < 7; c++){
    const shapes = [SHAPES[a], SHAPES[b], SHAPES[c]];
    const pls = buildPlacements(shapes);
    for (let x = 0; x <= MAX_COUNT; x++) for (let y = 0; y <= MAX_COUNT; y++) for (let z = 0; z <= MAX_COUNT; z++){
      if (x + y + z === 0) continue;
      const t0 = performance.now();
      const r = analyze(pls, empty, [x, y, z]);
      const ms = performance.now() - t0;
      if (!r.total) continue;
      times.push(ms);
      if (!worst || ms > worst.ms) worst = { ms, shapes, counts: [x, y, z], total: r.total };
    }
  }
  return { q: quantiles(times), worst };
}

// ---- games ----
function randomGame(rnd, mix){
  for (;;){
    const shapes = pickShapes(rnd);
    const counts = shapes.map(() => mix === 'low' ? 1 + Math.floor(rnd() * 2) : 1 + Math.floor(rnd() * MAX_COUNT));
    if (mix === 'high' && shapes.reduce((s, sh, i) => s + AREA(sh) * counts[i], 0) > 36) continue;
    const pls = buildPlacements(shapes);
    const hidden = sampleArrangement(pls, new Int8Array(NCELLS), counts, rnd);
    if (hidden) return { shapes, counts, pls, hidden };
  }
}

// Uniform arrangement, one tile at a time: take an open tile some
// arrangement covers, then mark it empty or pick the placement covering it,
// each with probability proportional to its share of the arrangements.
function sampleArrangement(pls, state0, counts0, rnd){
  const state = Int8Array.from(state0), counts = counts0.slice(), out = [];
  while (counts.some(c => c > 0)){
    const a = analyze(pls, state, counts);
    if (!a.total) return null;
    let t = 0;
    while (state[t] !== 0 || a.cell[t] === 0) t++;
    let x = rnd() * a.total, chosen = -1;
    for (let id = 0; id < pls.length && chosen < 0; id++){
      if (!a.pc[id] || !pls[id].cells.includes(t)) continue;
      x -= a.pc[id];
      if (x < 0) chosen = id;
    }
    if (chosen < 0){ state[t] = 1; continue; }
    for (const i of pls[chosen].cells) state[i] = 3;
    counts[pls[chosen].slot]--;
    out.push(chosen);
  }
  return out;
}

function highestChance(state, a, rnd){
  let best = -1, bv = -1, ties = 0;
  for (let i = 0; i < NCELLS; i++){
    if (state[i] !== 0) continue;
    const v = a.cell[i];
    if (v > bv * (1 + 1e-12)){ bv = v; best = i; ties = 1; }
    else if (v >= bv * (1 - 1e-12) && rnd() * ++ties < 1) best = i;
  }
  return best;
}

function playGame(seed, mix, policy){
  const rnd = mulberry32(seed);
  const g = randomGame(rnd, mix);
  const state = new Int8Array(NCELLS), counts = g.counts.slice();
  const owner = new Int32Array(NCELLS).fill(-1);
  for (const id of g.hidden) for (const i of g.pls[id].cells) owner[i] = id;
  const search = policy === 'page' ? makeSearch(g.pls) : null;
  const tieRnd = mulberry32(seed ^ 0x5bd1e995);
  let misses = 0, moves = 0, exactMoves = 0, probMs = 0, searchMs = 0;
  while (counts.some(c => c > 0)){
    let t0 = performance.now();
    const a = analyze(g.pls, state, counts);
    probMs += performance.now() - t0;
    let t = -1;
    if (search){
      t0 = performance.now();
      const r = search.solve(state, counts, SEARCH_MS);
      searchMs += performance.now() - t0;
      if (r){ t = r.bestTiles[Math.floor(tieRnd() * r.bestTiles.length)]; exactMoves++; }
    }
    if (t < 0) t = highestChance(state, a, tieRnd);
    moves++;
    if (owner[t] >= 0){
      const pl = g.pls[owner[t]];
      for (const i of pl.cells) state[i] = 3;
      counts[pl.slot]--;
    } else { state[t] = 1; misses++; }
  }
  return { misses, moves, exactMoves, probMs, searchMs };
}

if (!isMainThread){
  parentPort.on('message', job => parentPort.postMessage({ job, r: playGame(job.seed, job.mix, job.policy) }));
  return;
}

function runGames(jobs, threads){
  return new Promise(resolve => {
    const results = [];
    let next = 0;
    const workers = [];
    for (let k = 0; k < Math.min(threads, jobs.length); k++){
      const w = new Worker(__filename);
      workers.push(w);
      const feed = () => { if (next < jobs.length) w.postMessage(jobs[next++]); };
      w.on('message', m => {
        results.push(m);
        if (results.length % 100 === 0) process.stdout.write(`  ${results.length}/${jobs.length} games\n`);
        if (results.length === jobs.length){ workers.forEach(x => x.terminate()); resolve(results); }
        else feed();
      });
      feed();
    }
  });
}

function localStamp(){
  const d = new Date(), p = n => String(n).padStart(2, '0');
  return `${d.getFullYear()}-${p(d.getMonth() + 1)}-${p(d.getDate())} ${p(d.getHours())}:${p(d.getMinutes())}`;
}

(async () => {
  const games = +(process.argv.find(a => /^\d+$/.test(a)) || 200);
  const threadArg = process.argv.find(a => a.startsWith('--threads='));
  const threads = threadArg ? +threadArg.slice(10) : 16;
  const lines = [];
  const log = s => { console.log(s); lines.push(s); };
  const fmt = (x, d = 2) => x.toFixed(d);
  log(`Treasure Hunt Solver bench · ${localStamp()} · ` +
      `${games} paired games · ${threads} threads · SEARCH_MS=${SEARCH_MS}`);

  log('\n== correctness: DP vs brute force ==');
  const c = correctness(400);
  log(`${c.checked} random boards, ${c.bad} mismatches`);

  log(`\n== DP speed, empty boards (every shape triple, counts 0..${MAX_COUNT}, feasible only) ==`);
  const sp = speedEmptyBoards();
  log(`${sp.q.n} boards · p50 ${fmt(sp.q.p50)} ms · p95 ${fmt(sp.q.p95)} ms · p99 ${fmt(sp.q.p99)} ms · max ${fmt(sp.q.max)} ms`);
  log(`slowest: ${sp.worst.shapes.join(',')} x ${sp.worst.counts.join(',')} · ${sp.worst.total.toLocaleString('en-US')} arrangements`);

  log('\n== play quality: page policy vs highest chance, paired hidden boards ==');
  const jobs = [];
  for (let i = 0; i < games; i++){
    const mix = i % 2 ? 'low' : 'high';
    for (const policy of ['greedy', 'page']) jobs.push({ seed: 100000 + i, mix, policy });
  }
  const t0 = performance.now();
  const res = await runGames(jobs, threads);
  const by = {};
  for (const { job, r } of res) (by[job.seed] = by[job.seed] || { mix: job.mix })[job.policy] = r;
  for (const mix of ['low', 'high', 'all']){
    const rows = Object.values(by).filter(x => mix === 'all' || x.mix === mix);
    const mean = xs => xs.reduce((s, x) => s + x, 0) / xs.length;
    const g = rows.map(x => x.greedy.misses), p = rows.map(x => x.page.misses);
    const d = rows.map((x, i) => p[i] - g[i]);
    const md = mean(d), se = Math.sqrt(mean(d.map(v => (v - md) ** 2)) / d.length);
    const moves = rows.reduce((s, x) => s + x.page.moves, 0);
    const exact = rows.reduce((s, x) => s + x.page.exactMoves, 0);
    const probMs = rows.reduce((s, x) => s + x.greedy.probMs, 0) / rows.reduce((s, x) => s + x.greedy.moves, 0);
    const searchMs = rows.reduce((s, x) => s + x.page.searchMs, 0) / moves;
    log(`${mix.padEnd(4)} ${String(rows.length).padStart(4)} games · misses: highest chance ${fmt(mean(g), 3)}, page ${fmt(mean(p), 3)}, ` +
        `diff ${md >= 0 ? '+' : ''}${fmt(md, 3)} ± ${fmt(se, 3)} · moves solved exactly ${fmt(100 * exact / moves, 1)}% · ` +
        `DP ${fmt(probMs)} ms/move · search ${fmt(searchMs, 0)} ms/move`);
  }
  log(`games wall time ${fmt((performance.now() - t0) / 1000, 0)} s`);
  fs.writeFileSync(path.join(__dirname, 'bench-report.txt'), lines.join('\n') + '\n');
})();
