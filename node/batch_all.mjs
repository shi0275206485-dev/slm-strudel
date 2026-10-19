// ===========================================================================
//  batch_all.mjs — recursive failure analysis across ALL shots / models / seeds
// ---------------------------------------------------------------------------
//  Recursively scans a root folder (e.g. ./outputs) for
//  baseline_<model>_<k>shot_seed<n>.jsonl files, analyses each, then aggregates
//  the 5 seeds per (model, shot) into a mean.
//
//  USAGE:   node batch_all.mjs <root-folder>      (defaults to ".")
//  Example: node batch_all.mjs ./outputs
//
//  OUTPUTS (written into <root-folder>):
//    - funnel_by_model_shot.csv  : mean over seeds, one row per (model, shot)
//    - funnel_raw_per_seed.csv    : every file, before aggregation
//  and prints a per-shot comparison table to the console.
//
//  Metrics per file:2
//    name%   = function name appears (your hit metric)        [exact]
//    call%   = proper call F(...) / .F(...)                    [exact]
//    syntax% = syntactically valid                             [APPROX unless real transpiler]
//    hit0%   = name never appears; inv%/val% = invalid / valid-but-missed (all % of N)
//
//  deps: npm install acorn acorn-walk @strudel/mini
// ===========================================================================
import fs from 'fs';
import path from 'path';
import { checkSyntax, stripFences } from './validate.mjs';
import * as acorn from 'acorn';
import { simple as walk } from 'acorn-walk';

const root = process.argv[2] || '.';
const esc = s => s.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');

const nameAppears = (code, F) => (F === '$:' || F === '$') ? /\$:/.test(code)
  : new RegExp('(^|[^A-Za-z0-9_])' + esc(F) + '([^A-Za-z0-9_]|$)').test(code);
const properCall = (code, F, validJS) => {
  if (F === '$:' || F === '$') return /\$:/.test(code);
  if (validJS) { try {
    const ast = acorn.parse(code, { ecmaVersion: 'latest', allowReturnOutsideFunction: true });
    let f = false;
    walk(ast, { CallExpression(n) { const c = n.callee;
      if (c.type === 'Identifier' && c.name === F) f = true;
      if (c.type === 'MemberExpression' && c.property && c.property.name === F) f = true; } });
    if (f) return true;
  } catch {} }
  return new RegExp('(^|[^A-Za-z0-9_.])' + esc(F) + '\\s*\\(').test(code)
      || new RegExp('\\.' + esc(F) + '\\s*\\(').test(code);
};

// recursive walk for *.jsonl
const BASELINE_RE = /^(?:baseline|ablation)_.+?_\d+shot_seed\d+\.jsonl$/i;
function findJsonl(d, acc = []) {
  for (const e of fs.readdirSync(d, { withFileTypes: true })) {
    const fp = path.join(d, e.name);
    if (e.isDirectory()) findJsonl(fp, acc);
    else if (BASELINE_RE.test(e.name)) acc.push(fp);   // baseline_ or ablation_ *_<k>shot_seed<n>.jsonl
  }
  return acc;
}

// parse baseline_<model>_<k>shot_seed<n>.jsonl  (also ablation_)
function parseName(fn) {
  const base = path.basename(fn);
  const m = base.match(/^(?:baseline|ablation)_(.+?)_(\d+)shot_seed(\d+)\.jsonl$/i);
  if (m) return { model: m[1], shot: parseInt(m[2], 10), seed: parseInt(m[3], 10) };
  // fallback: shot from anything like _(\d+)shot
  const ms = base.match(/_(\d+)shot/i);
  return { model: base.replace(/\.jsonl$/, ''), shot: ms ? parseInt(ms[1], 10) : -1, seed: -1 };
}

function analyzeFile(fp) {
  const lines = fs.readFileSync(fp, 'utf8').split('\n').filter(x => x.trim());
  let nName = 0, nCall = 0, nSyn = 0, nMiss = 0, inv = 0, val = 0, skipped = 0; const ex = [];
  for (const line of lines) {
    let o; try { o = JSON.parse(line); } catch { skipped++; continue; }
    // tolerate records that are not in {messages:[...]} form
    if (!o || !Array.isArray(o.messages)) { skipped++; continue; }
    const m = Object.fromEntries(o.messages.map(x => [x.role, x.content]));
    // F source: prefer the explicit ref_function field stored by the ablation
    // generator (its prompt no longer contains the line); else parse it from the
    // user turn (baseline files still carry it). Lets ONE scorer handle A and B.
    let F = (o.ref_function != null) ? String(o.ref_function).trim() : null;
    if (!F) {
      const refM = (m.user || '').match(/Reference function:\s*(.+)/);
      if (!refM) { skipped++; continue; }
      F = refM[1].split('\\n')[0].trim();
    }
    const code = stripFences(m.assistant || '');  // align with pipeline: strip ``` fences first
    const syn = checkSyntax(code);
    const nm = nameAppears(code, F), cl = properCall(code, F, syn.valid);
    if (syn.valid) nSyn++; if (nm) nName++; if (cl) nCall++;
    if (!nm) { nMiss++; if (syn.valid) val++; else { inv++;
      if (ex.length < 3) ex.push(`[${F}] ${code.replace(/\n/g,' ').slice(0,55)}`); } }
  }
  const N = lines.length || 1;
  if (skipped) console.error(`  [warn] ${path.basename(fp)}: skipped ${skipped}/${lines.length} non-standard lines`);
  return { N, name: 100*nName/N, call: 100*nCall/N, syntax: 100*nSyn/N,
           hit0: 100*nMiss/N, inv: 100*inv/N, val: 100*val/N, examples: ex };
}

const files = findJsonl(root).sort();
if (!files.length) { console.error('No baseline_/ablation_*_<k>shot_seed<n>.jsonl found under', root); process.exit(1); }

// analyse every file
const raw = files.map(fp => { const meta = parseName(fp); return { fp, ...meta, ...analyzeFile(fp) }; });

// aggregate seeds -> (model, shot)
const agg = {};
for (const r of raw) {
  const key = r.model + '||' + r.shot;
  (agg[key] = agg[key] || []).push(r);
}
const mean = (a, k) => a.reduce((s, x) => s + x[k], 0) / a.length;
const rows = Object.entries(agg).map(([key, arr]) => {
  const [model, shot] = key.split('||');
  return { model, shot: parseInt(shot, 10), seeds: arr.length,
    name: mean(arr,'name'), call: mean(arr,'call'), syntax: mean(arr,'syntax'),
    hit0: mean(arr,'hit0'), inv: mean(arr,'inv'), val: mean(arr,'val'),
    examples: arr[0].examples };
}).sort((a,b)=> a.model.localeCompare(b.model) || a.shot-b.shot);

// console: one table per shot
const shots = [...new Set(rows.map(r=>r.shot))].sort((a,b)=>a-b);
const pad=(s,w)=>String(s).padEnd(w);
const f1=x=>x.toFixed(1);
for (const k of shots) {
  console.log(`\n===== ${k}-shot  (mean over seeds) =====`);
  console.log(pad('model',26),pad('seeds',6),pad('name%',7),pad('call%',7),pad('syntax%',8),pad('Hit0%',6),pad('inv%',5),pad('val%',5));
  console.log('-'.repeat(72));
  for (const r of rows.filter(r=>r.shot===k))
    console.log(pad(r.model,26),pad(r.seeds,6),pad(f1(r.name),7),pad(f1(r.call),7),pad(f1(r.syntax),8),pad(f1(r.hit0),6),pad(f1(r.inv),5),pad(f1(r.val),5));
}
console.log('\nname%/call% exact; syntax% APPROX unless real transpiler swapped into validate.mjs.');

// CSVs
const aggCsv = ['model,shot,seeds,name_pct,call_pct,syntax_pct_approx,hit0_pct,hit0_invalid_pct,hit0_valid_pct']
  .concat(rows.map(r=>[r.model,r.shot,r.seeds,f1(r.name),f1(r.call),f1(r.syntax),f1(r.hit0),f1(r.inv),f1(r.val)].join(','))).join('\n');
fs.writeFileSync(path.join(root,'funnel_by_model_shot.csv'), aggCsv);
const rawCsv = ['model,shot,seed,N,name_pct,call_pct,syntax_pct_approx,hit0_pct,hit0_invalid_pct,hit0_valid_pct']
  .concat(raw.map(r=>[r.model,r.shot,r.seed,r.N,f1(r.name),f1(r.call),f1(r.syntax),f1(r.hit0),f1(r.inv),f1(r.val)].join(','))).join('\n');
fs.writeFileSync(path.join(root,'funnel_raw_per_seed.csv'), rawCsv);
console.log('\nWrote funnel_by_model_shot.csv and funnel_raw_per_seed.csv into', root);

// zero-shot invalid examples (the could-not evidence)
const zero = rows.filter(r=>r.shot===0 && r.examples.length);
if (zero.length) { console.log('\n===== 0-shot invalid-code examples ====='); 
  for (const r of zero){ console.log(`\n# ${r.model}`); r.examples.forEach(e=>console.log('   '+e)); } }
