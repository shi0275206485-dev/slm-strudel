// epoch_hap_check.mjs
//
// CLI wrapper around the bundled hap_query_bundle.mjs (built from
// hap_query_entry.mjs -- see that file's docstring for the required build
// step). Reads a candidate completions JSONL file, queries each snippet's
// Haps over N_CYCLES cycles, and writes a per-entry JSONL manifest.
//
// Usage:
//   node epoch_hap_check.mjs <input.jsonl> <output_manifest.jsonl> [numCycles]
//
// SAFETY NOTES:
// - evaluate() runs generated code via Function(), same as the live Strudel
//   editor. The same unsafe-code pre-screen used throughout this project
//   (t9.py, epoch_syntax_check.mjs's sibling scripts) is applied here too,
//   to catch known non-terminating-loop patterns before evaluation -- but
//   unlike the Playwright-based checks, there is no per-item process
//   isolation here: a hang NOT caught by the pre-screen would stall this
//   whole batch, not just one entry. This is a known residual risk,
//   consistent with what the pre-screen already accepts elsewhere.
// - Results are written to disk INCREMENTALLY, one line per entry, instead
//   of being accumulated in memory and written once at the end. This was
//   changed after a real failure: a single entry produced a BigInt value
//   (root cause not fully isolated, but getHaps() now sanitizes it at the
//   source) that crashed JSON.stringify on the WHOLE batch array, silently
//   discarding all 150 entries' results including ones already computed.
//   Writing incrementally means a single bad entry can now only ever cost
//   that one line, never the rest of the batch.

import fs from 'fs';

const N_CYCLES_DEFAULT = 4;

// Simple regex-matchable non-terminating patterns.
const _UNSAFE_PATTERNS = [
  [/\bwhile\s*\(\s*true\s*\)/, 'while(true)'],
  [/\bwhile\s*\(\s*1\s*\)/, 'while(1)'],
  [/\bfor\s*\(\s*;\s*;\s*\)/, 'for(;;)'],
  [/\bfor\s*\(\s*;\s*true\s*;\s*\)/, 'for(;true;)'],
  [/\bdo\s*\{[\s\S]*?\}\s*while\s*\(\s*(?:true|1)\s*\)/, 'do...while(true)'],
];

// Event-explosion guard. These are syntactically valid, non-looping calls,
// but they instruct Strudel to pack an astronomically large number of
// events into the query window, which hangs Hap extraction / audio
// rendering just as effectively as an infinite loop. Two distinct cases:
//   (a) a SINGLE huge multiplier -- e.g. `hh*4096`, `.fast(2048)`, `.ply(500)`;
//   (b) MANY chained density calls whose PRODUCT explodes even though each
//       one alone is harmless -- e.g. `.fast(2).fast(4)...fast(2048)`
//       (the real case that hung Gemma3-1B's few-shot output: product = 2^66).
// `slow`/`div` shrink density rather than inflate event count, so they are
// deliberately NOT included here.
const SINGLE_FACTOR_LIMIT = 256;     // any one factor above this is rejected outright
const CUMULATIVE_FACTOR_LIMIT = 512; // product of chained fast/ply/mini-`*` factors

function screenUnsafeCode(code) {
  for (const [pat, label] of _UNSAFE_PATTERNS) {
    if (pat.test(code)) return label;
  }

  // (a) any single oversized density factor: fast(N) / ply(N) / *N (mini-notation)
  for (const m of code.matchAll(/(?:\bfast|\bply)\s*\(\s*(\d+(?:\.\d+)?)\s*\)/g)) {
    if (parseFloat(m[1]) > SINGLE_FACTOR_LIMIT) return `oversized density factor: ${m[0]}`;
  }
  for (const m of code.matchAll(/\*\s*(\d+)/g)) {
    if (parseInt(m[1], 10) > SINGLE_FACTOR_LIMIT) return `oversized repetition factor: *${m[1]}`;
  }

  // (b) cumulative product of ALL chained density-increasing factors
  let product = 1;
  for (const m of code.matchAll(/(?:\bfast|\bply)\s*\(\s*(\d+(?:\.\d+)?)\s*\)/g)) {
    product *= parseFloat(m[1]);
    if (product > CUMULATIVE_FACTOR_LIMIT) return `cumulative density explosion (product>${CUMULATIVE_FACTOR_LIMIT})`;
  }

  return null;
}

function jsonSafeReplacer(key, value) {
  // Second layer of defense (getHaps() already sanitizes BigInt at the
  // source) -- if anything unserializable slips through anywhere, convert
  // rather than let it throw.
  return typeof value === 'bigint' ? Number(value) : value;
}

function stringifySafe(obj) {
  try {
    return JSON.stringify(obj, jsonSafeReplacer);
  } catch (e) {
    // Even the fallback failed to serialize -- write a minimal, guaranteed-
    // safe record instead of losing this line's slot entirely.
    return JSON.stringify({ entry_index: obj.entry_index, status: 'SERIALIZE_ERROR', message: String(e) });
  }
}

async function main() {
  const inputFile = process.argv[2];
  const outputManifest = process.argv[3];
  const numCycles = process.argv[4] ? parseInt(process.argv[4], 10) : N_CYCLES_DEFAULT;

  if (!inputFile || !outputManifest) {
    console.error('Usage: node epoch_hap_check.mjs <input.jsonl> <output_manifest.jsonl> [numCycles]');
    process.exit(1);
  }

  await import('./hap_query_bundle.mjs');

  const lines = fs.readFileSync(inputFile, 'utf-8').split('\n').filter((l) => l.trim() !== '');
  const outFd = fs.openSync(outputManifest, 'w');
  let successCount = 0;

  for (let i = 0; i < lines.length; i++) {
    let result;
    let data;
    try {
      data = JSON.parse(lines[i]);
    } catch (e) {
      result = { entry_index: i, status: 'JSON_PARSE_ERROR', message: String(e), haps: [] };
      fs.writeSync(outFd, stringifySafe(result) + '\n');
      continue;
    }

    const code = data.messages[2].content
      .replace(/```javascript|```js|```strudel|```/g, '')
      .trim();

    const unsafe = screenUnsafeCode(code);
    if (unsafe) {
      result = { entry_index: i, status: 'UNSAFE_CODE', message: `non-terminating loop: ${unsafe}`, haps: [] };
      fs.writeSync(outFd, stringifySafe(result) + '\n');
      continue;
    }

    try {
      const haps = await globalThis.__getHaps(code, numCycles);
      result = { entry_index: i, status: 'SUCCESS', haps };
      successCount++;
    } catch (e) {
      result = { entry_index: i, status: 'HAP_QUERY_ERROR', message: String(e && e.message ? e.message : e), haps: [] };
    }
    fs.writeSync(outFd, stringifySafe(result) + '\n');
  }

  fs.closeSync(outFd);

  console.log(`HAP_QUERY_DONE=${lines.length}`);
  console.log(`HAP_QUERY_SUCCESS=${successCount}`);
}

main();
