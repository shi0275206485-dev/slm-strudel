// ===========================================================================
//  validate.mjs  —  Strudel syntax checker (REAL @strudel/transpiler)
// ---------------------------------------------------------------------------
//  Mirrors the original pipeline (n-shot-avg.mjs):
//    1. strip markdown code fences from the generated code
//    2. transpiler(code) — if it does not throw, the code is syntactically valid
//
//  This matches the pipeline's "pass rate" definition exactly, including the
//  markdown-fence stripping, so syntax numbers line up with your baseline table.
//
//  deps:  npm install @strudel/transpiler
// ===========================================================================
import { transpiler } from '@strudel/transpiler';

// Same fence-stripping as the original pipeline (n-shot-avg.mjs).
export function stripFences(code) {
  return code.replace(/```javascript|```js|```strudel|```/g, '').trim();
}

// Syntax check on ALREADY-STRIPPED code (callers strip first; see analysis scripts).
export function checkSyntax(code) {
  try { transpiler(code); return { valid: true }; }
  catch (e) { return { valid: false, error: String(e.message).slice(0, 80) }; }
}
