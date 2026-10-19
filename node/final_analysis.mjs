// ===========================================================================
//  final_analysis.mjs  —  zero-shot failure analysis for Strudel generations
// ---------------------------------------------------------------------------
//  Answers: when the model is given a reference function F in the prompt,
//  what happens? Produces a "funnel" and a two-class breakdown of misses.
//
//  USAGE:
//     node final_analysis.mjs <generations.jsonl>
//     node final_analysis.mjs baseline_gemma-2-2b-it_0shot_seed2.jsonl
//
//  INPUT FORMAT (one JSON object per line):
//     {"messages":[{"role":"system",...},
//                  {"role":"user","content":"... Reference function: <F> ..."},
//                  {"role":"assistant","content":"<generated code>"}]}
//
//  OUTPUT:
//   1. FUNNEL — how far the generations get:
//        name appears (your hit metric)  >  proper call F(...)  >  syntactically valid
//      (execution success: take from your own pipeline; not computed here)
//   2. Hit=0 cases (function name never appears) split into:
//        - invalid code      (fails syntax check)
//        - valid but missed  (parses, but F absent)
//      with concrete examples of each.
//
//  NOTE: "syntactically valid" uses validate.mjs, which APPROXIMATES the real
//  @strudel/transpiler (see validate.mjs header). For exact validity counts,
//  swap in your real transpiler there. The "name appears" (your hit metric)
//  and "proper call" figures do NOT depend on the transpiler and are exact.
//
//  deps:  npm install acorn acorn-walk @strudel/mini
// ===========================================================================
import fs from 'fs';
import * as acorn from 'acorn';
import { simple as walk } from 'acorn-walk';
import { transpiler } from '@strudel/transpiler';

export function checkSyntax(code) {
  try { transpiler(code, {}); return { valid: true }; }
  catch (e) { return { valid: false, error: String(e.message).slice(0,80) }; }
}
const file = process.argv[2];
if (!file) { console.error('usage: node final_analysis.mjs <file.jsonl>'); process.exit(1); }
const lines = fs.readFileSync(file, 'utf8').split('\n').filter(x => x.trim());
const esc = s => s.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');

// Your hit metric: does the function NAME appear anywhere as a token?
const nameAppears = (code, F) => (F === '$:' || F === '$') ? /\$:/.test(code)
  : new RegExp('(^|[^A-Za-z0-9_])' + esc(F) + '([^A-Za-z0-9_]|$)').test(code);

// Stricter: is F actually CALLED — F(...) or .F(...) ?
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

let nName = 0, nCall = 0, nSyn = 0;
const missInvalid = [], missValid = [];
for (const line of lines) {
  const o = JSON.parse(line);
  const m = Object.fromEntries(o.messages.map(x => [x.role, x.content]));
  const F = (m.user.match(/Reference function:\s*(.+)/) || [])[1].split('\\n')[0].trim();
  const code = m.assistant || '';
  const syn = checkSyntax(code);
  const nm = nameAppears(code, F), cl = properCall(code, F, syn.valid);
  if (syn.valid) nSyn++; if (nm) nName++; if (cl) nCall++;
  if (!nm) (syn.valid ? missValid : missInvalid).push({ F, code: code.replace(/\n/g, ' ') });
}
const N = lines.length, pc = n => (100 * n / N).toFixed(1) + '%';
console.log(`FILE: ${file}  (N=${N})`);
console.log('\n========== FUNNEL ==========');
console.log(`name appears (your hit metric) : ${nName}/${N}  (${pc(nName)})`);
console.log(`proper call F(...) / .F(...)   : ${nCall}/${N}  (${pc(nCall)})`);
console.log(`syntactically valid (approx)   : ${nSyn}/${N}  (${pc(nSyn)})   <- swap in real transpiler for exact`);
console.log(`execution success              : (take from your pipeline)`);
console.log('\n========== Hit=0 (function name never appears) ==========');
const tot = missInvalid.length + missValid.length;
console.log(`total Hit=0       : ${tot}/${N}  (${pc(tot)})`);
console.log(`  invalid code    : ${missInvalid.length}`);
console.log(`  valid but missed: ${missValid.length}`);
console.log('\n--- invalid-code examples ---');
missInvalid.slice(0, 8).forEach(x => console.log(`  [ref=${x.F}] ${JSON.stringify(x.code.slice(0, 90))}`));
console.log('\n--- valid-but-missed examples ---');
missValid.slice(0, 8).forEach(x => console.log(`  [ref=${x.F}] ${JSON.stringify(x.code.slice(0, 90))}`));
