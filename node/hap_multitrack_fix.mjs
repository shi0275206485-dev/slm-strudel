/**
 * hap_multitrack_fix.mjs
 *
 * Purpose
 * -------
 * Strudel's transpiler rewrites each `$:` line into a `.p(name)` call. If `.p`
 * is stubbed as an identity function, every `$:` statement still evaluates, but
 * only the LAST one survives as the expression value -- so `queryArc` sees a
 * single track and the rest of the piece is invisible to the metric.
 *
 * Evidence from the reference set: entry 17 has two tracks, a clap pattern and
 * a triangle melody, and returns 20 haps of which every single one is triangle.
 * Entry 10 has four tracks and returns four haps, all clap. 21 of the 23
 * multi-track reference items show only one sound source in their hap list.
 *
 * The fix is to make `.p` register the pattern instead of discarding it, then
 * stack everything that registered.
 *
 * Usage
 * -----
 *   node hap_multitrack_fix.mjs --verify
 *       Runs three built-in cases and prints old-behaviour vs new-behaviour
 *       hap counts side by side. Expect entry 17: 20 -> 28.
 *
 *   node hap_multitrack_fix.mjs --input <file.jsonl> --output <manifest.jsonl>
 *       Extracts haps for every line of a ChatML jsonl, writing a manifest in
 *       the same shape the existing pipeline already consumes:
 *         {entry_index, status, haps, message?}
 *
 * Port the `evaluateWithTracks` function into hap_query_bundle.mjs; nothing
 * else in the pipeline needs to change.
 */

import fs from 'fs';
import * as core from '@strudel/core';
import { evaluate as strudelEvaluate } from '@strudel/transpiler';

// ---------------------------------------------------------------------------
// Safety screen: cascading fast()/ply() factors explode the event count and
// hang the query. Same limits the rest of the pipeline uses.
// ---------------------------------------------------------------------------
const SINGLE_FACTOR_LIMIT = 256;
const CUMULATIVE_FACTOR_LIMIT = 512;

function isUnsafe(code) {
  if (/while\s*\(\s*true\s*\)|for\s*\(\s*;\s*;/.test(code)) return 'infinite loop';
  let cumulative = 1;
  const re = /\.(fast|ply)\s*\(\s*([0-9.]+)\s*\)/g;
  let m;
  while ((m = re.exec(code)) !== null) {
    const f = parseFloat(m[2]);
    if (!Number.isFinite(f) || f <= 0) continue;
    if (f > SINGLE_FACTOR_LIMIT) return `factor ${f} exceeds single limit`;
    cumulative *= f;
    if (cumulative > CUMULATIVE_FACTOR_LIMIT) return `cumulative factor ${cumulative} exceeds limit`;
  }
  return null;
}

// ---------------------------------------------------------------------------
// The fix.
// ---------------------------------------------------------------------------
/**
 * Evaluate `code` and return the pattern representing ALL of its tracks.
 *
 * `collectTracks = false` reproduces the old identity-stub behaviour, so the
 * two can be compared directly.
 */
async function evaluateWithTracks(code, { collectTracks = true } = {}) {
  const tracks = [];

  const Pattern = core.Pattern;
  const originalP = Pattern.prototype.p;

  // Tempo setters are no-ops: queryArc works in cycle-space regardless.
  const originalSetcps = globalThis.setcps;
  const originalSetcpm = globalThis.setcpm;
  const originalSetbpm = globalThis.setbpm;
  globalThis.setcps = () => {};
  globalThis.setcpm = () => {};
  globalThis.setbpm = () => {};

  Pattern.prototype.p = function patched() {
    if (collectTracks) tracks.push(this);
    return this;
  };

  try {
    const result = await strudelEvaluate(code);
    let pattern = result && result.pattern ? result.pattern : result;

    if (collectTracks && tracks.length > 1) {
      // More than one `$:` line registered -- combine them.
      pattern = core.stack(...tracks);
    } else if (collectTracks && tracks.length === 1) {
      pattern = tracks[0];
    }
    return pattern;
  } finally {
    Pattern.prototype.p = originalP;
    globalThis.setcps = originalSetcps;
    globalThis.setcpm = originalSetcpm;
    globalThis.setbpm = originalSetbpm;
  }
}

/** Strip BigInts and other non-serialisable values out of a hap value. */
function sanitise(value) {
  if (value === null || value === undefined) return value;
  if (typeof value === 'bigint') return Number(value);
  if (Array.isArray(value)) return value.map(sanitise);
  if (typeof value === 'object') {
    const out = {};
    for (const [k, v] of Object.entries(value)) {
      const s = sanitise(v);
      if (typeof s !== 'function' && s !== undefined) out[k] = s;
    }
    return out;
  }
  return value;
}

async function extractHaps(code, opts = {}) {
  const unsafe = isUnsafe(code);
  if (unsafe) return { status: 'UNSAFE_CODE', message: unsafe, haps: [] };

  try {
    const pattern = await evaluateWithTracks(code, opts);
    if (!pattern || typeof pattern.queryArc !== 'function') {
      return { status: 'HAP_QUERY_ERROR', message: 'evaluate did not yield a Pattern', haps: [] };
    }
    const raw = pattern.queryArc(0, 4);
    const haps = raw.map((h) => ({
      begin: Number(h.whole ? h.whole.begin : h.part.begin),
      end: Number(h.whole ? h.whole.end : h.part.end),
      value: sanitise(h.value),
    }));
    return { status: 'SUCCESS', haps };
  } catch (err) {
    return { status: 'HAP_QUERY_ERROR', message: String(err && err.message ? err.message : err), haps: [] };
  }
}

// ---------------------------------------------------------------------------
// Verification mode
// ---------------------------------------------------------------------------
const CASES = [
  {
    label: 'entry 17  (2 tracks: clap + triangle melody)',
    code: '$: s(`~ cp ~ cp`)\n$: n(`0 2 4 5 7`).scale(`D:dorian`).s(`triangle`)',
    expectOld: 20,
    expectNew: 28,
  },
  {
    label: 'entry 10  (4 tracks)',
    code:
      '$: chord(`<GM7 D Em7 CM7>`).voicing().s(`tri`).slow(8).room(0.99).gain(0.1)\n' +
      '$: n(`12 11 9 7 5 7 9 11`).scale(`G:major`).s(`tri`).slow(4).delay(0.35).gain(0.08)\n' +
      '$: n(`0`).scale(`G:major`).s(`sine`).slow(8).lpf(76).gain(0.3)\n' +
      '$: s(`~ ~ ~ cp`).gain(0.01)',
    expectOld: 4,
    expectNew: null,
  },
  {
    label: 'single track control  (should be unaffected)',
    code: 's(`bd hh sd hh`)',
    expectOld: 16,
    expectNew: 16,
  },
];

function soundsOf(haps) {
  const set = new Set();
  for (const h of haps) {
    const v = h.value || {};
    set.add(v.s || v.sound || v.note || '?');
  }
  return [...set].sort();
}

async function verify() {
  console.log('\nComparing identity-stub behaviour against stacked-track behaviour.');
  console.log('='.repeat(78));
  let allGood = true;

  for (const c of CASES) {
    const oldRes = await extractHaps(c.code, { collectTracks: false });
    const newRes = await extractHaps(c.code, { collectTracks: true });
    const oldN = oldRes.haps.length;
    const newN = newRes.haps.length;

    console.log(`\n${c.label}`);
    console.log(`  before : ${String(oldN).padStart(3)} haps   sounds: ${soundsOf(oldRes.haps).join(', ') || '-'}`
      + (oldRes.status !== 'SUCCESS' ? `   [${oldRes.status}: ${oldRes.message}]` : ''));
    console.log(`  after  : ${String(newN).padStart(3)} haps   sounds: ${soundsOf(newRes.haps).join(', ') || '-'}`
      + (newRes.status !== 'SUCCESS' ? `   [${newRes.status}: ${newRes.message}]` : ''));

    if (c.expectOld !== null && oldN !== c.expectOld) {
      console.log(`  NOTE: expected ${c.expectOld} haps before the fix, got ${oldN}.`);
    }
    if (c.expectNew !== null && newN !== c.expectNew) {
      console.log(`  WARNING: expected ${c.expectNew} haps after the fix, got ${newN}.`);
      allGood = false;
    }
    if (newN > oldN) console.log('  -> tracks recovered.');
    if (newN === oldN && c.expectNew !== c.expectOld) {
      console.log('  -> no change; the stub may be installed somewhere else in your bundle.');
      allGood = false;
    }
  }

  console.log('\n' + '='.repeat(78));
  console.log(allGood
    ? 'Looks right. Port evaluateWithTracks into hap_query_bundle.mjs and re-run.'
    : 'Something did not behave as predicted -- check how .p is stubbed in your bundle before porting.');
  console.log();
}

// ---------------------------------------------------------------------------
// Batch mode
// ---------------------------------------------------------------------------
async function runBatch(inputPath, outputPath) {
  const lines = fs.readFileSync(inputPath, 'utf-8').split('\n').filter((l) => l.trim());
  const out = fs.createWriteStream(outputPath, { encoding: 'utf-8' });
  let ok = 0, multi = 0;

  for (let i = 0; i < lines.length; i += 1) {
    let code = '';
    try {
      const row = JSON.parse(lines[i]);
      code = row.messages[row.messages.length - 1].content;
    } catch (err) {
      out.write(JSON.stringify({ entry_index: i, status: 'JSON_PARSE_ERROR', message: String(err), haps: [] }) + '\n');
      continue;
    }
    code = code.replace(/```(javascript|js|strudel)?/g, '').trim();

    const res = await extractHaps(code);
    if (res.status === 'SUCCESS') ok += 1;
    if ((code.match(/\$:/g) || []).length > 1) multi += 1;

    // written incrementally so one bad entry cannot cost the whole batch
    out.write(JSON.stringify({ entry_index: i, ...res }) + '\n');
    if ((i + 1) % 25 === 0) process.stdout.write(`  ${i + 1}/${lines.length}\r`);
  }

  out.end();
  console.log(`\n${ok}/${lines.length} extracted, ${multi} of them multi-track. Written to ${outputPath}`);
}

// ---------------------------------------------------------------------------
const args = process.argv.slice(2);
if (args.includes('--verify')) {
  await verify();
} else {
  const iIdx = args.indexOf('--input');
  const oIdx = args.indexOf('--output');
  if (iIdx === -1 || oIdx === -1) {
    console.log('Usage:\n  node hap_multitrack_fix.mjs --verify\n  node hap_multitrack_fix.mjs --input <in.jsonl> --output <manifest.jsonl>');
    process.exit(1);
  }
  await runBatch(args[iIdx + 1], args[oIdx + 1]);
}
