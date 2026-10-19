// ===========================================================================
//  score_ablation.mjs
//  ---------------------------------------------------------------------------
//  Reference-Function ablation, SCORING stage.
//
//  Scores the SAME funnel as the baseline failure-analysis (name% > call% >
//  syntax%) but reads the ground-truth function from the "ref_function"
//  metadata field instead of parsing it out of the prompt (Group B prompts no
//  longer contain it). This makes Group A and Group B directly comparable: any
//  drop in name%/call% from A to B is the instruction-following component.
//
//  USAGE:
//    node score_ablation.mjs <generations_glob_or_files...>
//    node score_ablation.mjs ablation_*_*shot_seed*.jsonl
//
//  EXPECTED INPUT (one object per line), as produced by your Group-B run:
//    {"messages":[system, user, {"role":"assistant","content":"<generated>"}],
//     "ref_function":"<F>", ...}
//  If "ref_function" is absent, the line is skipped (with a tally).
//
//  FILE NAMING (for aggregation): ablation_<model>_<K>shot_seed<n>.jsonl
//  (same convention as baseline_<model>_<K>shot_seed<n>.jsonl)
//
//  OUTPUT: a per-(model x shot) table averaged over seeds, plus a JSON dump
//  to ablation_scores.json for plotting / LaTeX table generation.
//
//  Syntax validity here uses the SAME validate.mjs you used for the baseline
//  funnel, so the two pipelines agree. If validate.mjs is absent, syntax% is
//  reported as null (name%/call% are transpiler-independent and still exact).
//
//  deps:  npm install acorn acorn-walk      (validate.mjs may add @strudel/*)
// ===========================================================================
import fs from 'fs';
import path from 'path';
import * as acorn from 'acorn';
import { simple as walk } from 'acorn-walk';

// ---- optional syntax checker (same one as the baseline) -------------------
let checkSyntax = null;
try {
  ({ checkSyntax } = await import('./validate.mjs'));
} catch {
  console.error('note: validate.mjs not found — syntax% will be null (name/call still exact)');
}

// ---- markdown-fence stripping (identical criteria to the baseline) --------
function stripFences(s) {
  if (!s) return '';
  // remove ```lang ... ``` fences, keep inner code
  const fenced = s.match(/```[a-zA-Z]*\n([\s\S]*?)```/);
  return (fenced ? fenced[1] : s).trim();
}

const esc = s => s.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');

// name appears as a token? ($: tag handled specially)
function nameAppears(code, F) {
  if (F === '$:' || F === '$') return /\$:/.test(code);
  return new RegExp('(^|[^A-Za-z0-9_])' + esc(F) + '([^A-Za-z0-9_]|$)').test(code);
}

// is F actually CALLED — F(...) or .F(...) ? ($: matched textually)
function isCalled(code, F) {
  if (F === '$:' || F === '$') return /\$:/.test(code);
  let called = false;
  let ast;
  try {
    ast = acorn.parse(code, { ecmaVersion: 'latest', allowReturnOutsideFunction: true });
  } catch {
    // fall back to a textual call check if it won't parse
    return new RegExp('(^|[^A-Za-z0-9_.])' + esc(F) + '\\s*\\(').test(code)
        || new RegExp('\\.\\s*' + esc(F) + '\\s*\\(').test(code);
  }
  walk(ast, {
    CallExpression(node) {
      const c = node.callee;
      if (c.type === 'Identifier' && c.name === F) called = true;
      if (c.type === 'MemberExpression' && c.property &&
          c.property.type === 'Identifier' && c.property.name === F) called = true;
    },
  });
  return called;
}

// ---- per-file scoring ------------------------------------------------------
function scoreFile(file) {
  const lines = fs.readFileSync(file, 'utf8').split('\n').filter(x => x.trim());
  let n = 0, name = 0, call = 0, syntax = 0, noRef = 0;
  for (const line of lines) {
    let obj;
    try { obj = JSON.parse(line); } catch { continue; }
    const F = obj.ref_function;
    if (!F) { noRef++; continue; }
    const asst = (obj.messages || []).find(m => m.role === 'assistant');
    const code = stripFences(asst ? asst.content : '');
    n++;
    const hasName = nameAppears(code, F);
    if (hasName) name++;
    if (isCalled(code, F)) call++;
    if (checkSyntax) { try { if (checkSyntax(code)) syntax++; } catch {} }
  }
  return { n, name, call, syntax, noRef };
}

// ---- filename parsing: ablation_<model>_<K>shot_seed<n>.jsonl --------------
function parseName(file) {
  const b = path.basename(file);
  const m = b.match(/^(?:ablation|baseline)_(.+)_(\d+)shot_seed(\d+)\.jsonl$/);
  if (!m) return null;
  return { model: m[1], shot: parseInt(m[2], 10), seed: parseInt(m[3], 10) };
}

// ---- main: aggregate by (model x shot) over seeds -------------------------
const files = process.argv.slice(2);
if (!files.length) { console.error('usage: node score_ablation.mjs <files...>'); process.exit(1); }

const groups = new Map(); // key: model|shot -> array of per-seed rate objects
let totalNoRef = 0;

for (const f of files) {
  const meta = parseName(f);
  if (!meta) { console.error(`skip (name not parseable): ${f}`); continue; }
  const r = scoreFile(f);
  totalNoRef += r.noRef;
  if (r.n === 0) { console.error(`skip (no scorable rows): ${f}`); continue; }
  const rates = {
    name:   100 * r.name   / r.n,
    call:   100 * r.call   / r.n,
    syntax: checkSyntax ? 100 * r.syntax / r.n : null,
    n: r.n, seed: meta.seed,
  };
  const key = `${meta.model}|${meta.shot}`;
  if (!groups.has(key)) groups.set(key, []);
  groups.get(key).push(rates);
}

const mean = a => a.reduce((s, x) => s + x, 0) / a.length;

const rows = [];
for (const [key, seeds] of [...groups.entries()].sort()) {
  const [model, shot] = key.split('|');
  rows.push({
    model, shot: parseInt(shot, 10), seeds: seeds.length,
    name:   +mean(seeds.map(s => s.name)).toFixed(1),
    call:   +mean(seeds.map(s => s.call)).toFixed(1),
    syntax: checkSyntax ? +mean(seeds.map(s => s.syntax)).toFixed(1) : null,
  });
}
rows.sort((a, b) => a.model.localeCompare(b.model) || a.shot - b.shot);

// pretty print
const pad = (s, w) => String(s).padEnd(w);
const padN = (s, w) => String(s).padStart(w);
console.log(pad('model', 30) + padN('K', 3) + padN('seeds', 7) +
            padN('name%', 8) + padN('call%', 8) + padN('syntax%', 9));
console.log('-'.repeat(65));
for (const r of rows) {
  console.log(pad(r.model, 30) + padN(r.shot, 3) + padN(r.seeds, 7) +
              padN(r.name.toFixed(1), 8) + padN(r.call.toFixed(1), 8) +
              padN(r.syntax === null ? '—' : r.syntax.toFixed(1), 9));
}
if (totalNoRef) console.error(`\nWARNING: ${totalNoRef} rows had no ref_function and were skipped`);

fs.writeFileSync('ablation_scores.json', JSON.stringify(rows, null, 2));
console.error('\nwrote ablation_scores.json');
