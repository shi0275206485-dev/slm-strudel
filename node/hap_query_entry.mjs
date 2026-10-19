/*
hap_query_entry.mjs

Entry point for querying a Strudel pattern's Haps directly (no audio
rendering, no browser needed) -- evaluates code via @strudel/core's
evaluate(), then calls pattern.queryArc(0, numCycles) to get the raw event
list.

WHY THIS NEEDS TO BE BUNDLED (not run directly with plain `node`):
@strudel/core depends on @kabelsalat/web, which is packaged using the
non-standard "module" field (a bundler convention, not part of Node's own
resolution algorithm) instead of a proper "exports" map. Plain `node
your_script.mjs` will hit `SyntaxError: ... does not provide an export
named 'SalatRepl'` because Node falls back to the wrong build artifact.
esbuild's bundler resolves this correctly.

BUILD (run once, from the same Node project directory that already has
@strudel/core, @strudel/mini, @strudel/transpiler, @strudel/tonal
installed -- e.g. wherever epoch_syntax_check.mjs lives):

    npm install --save-dev esbuild
    npm install @strudel/tonal          (needed for chord()/voicing()/scale() --
                                          NOT bundled with @strudel/core itself)
    npx esbuild hap_query_entry.mjs --bundle --format=esm --platform=browser --outfile=hap_query_bundle.mjs

Then epoch_hap_check.mjs (the CLI wrapper) imports the resulting
hap_query_bundle.mjs, not this file directly.

CHANGE LOG (if you have an older bundle, rebuild -- these two fixes were
found necessary when validating chord()/voicing() rows and multi-track
($:) code):
  - Added @strudel/tonal to the evalScope. Without it, chord(), voicing(),
    and scale() all throw "is not a function" -- these are NOT part of
    @strudel/core despite being extremely common in this dataset.
  - Added Pattern.prototype.p as a no-op. The transpiler auto-appends
    .p('$') to every $:-prefixed line (multi-track syntax) to register it
    with the REPL's playback scheduler; without a stub for it, any
    multi-track code throws "p is not a function".
*/

import { evaluate, evalScope, Pattern } from '@strudel/core';
import { transpiler } from '@strudel/transpiler';
import * as core from '@strudel/core';
import * as mini from '@strudel/mini';
import * as tonal from '@strudel/tonal';

// .p(name) registers/names a pattern for the REPL's multi-track ($:) playback
// scheduling -- pure bookkeeping, doesn't affect the pattern's own Haps, so
// it's a safe no-op passthrough for querying purposes.
Pattern.prototype.p = function (name) { return this; };

// setcps/setcpm/setbpm are REPL-scheduler side-effects (they belong to a
// live playback scheduler, see @strudel/core/repl.mjs), not pure pattern
// functions. They're irrelevant for Hap queries anyway: queryArc(0, N)
// operates in cycle-space regardless of tempo, so we just stub them out
// as no-ops rather than trying to honor them -- this also means, unlike
// the audio pipeline, NO cps detection is needed at all for this metric.
const replStubs = {
  setcps: () => {},
  setcpm: () => {},
  setbpm: () => {},
  hush: () => {},
  all: (x) => x,
};

let scopeReady = false;
async function ensureScope() {
  if (!scopeReady) {
    await evalScope(core, mini, tonal, replStubs);
    scopeReady = true;
  }
}

function sanitizeForJson(value) {
  // Some internal Strudel numeric computation can apparently yield BigInt
  // in certain edge cases (root cause not fully isolated), which
  // JSON.stringify cannot serialize at all -- convert defensively so a
  // single odd value doesn't poison the whole batch.
  if (typeof value === 'bigint') return Number(value);
  if (Array.isArray(value)) return value.map(sanitizeForJson);
  if (value && typeof value === 'object') {
    const out = {};
    for (const k of Object.keys(value)) out[k] = sanitizeForJson(value[k]);
    return out;
  }
  return value;
}

export async function getHaps(code, numCycles = 4) {
  await ensureScope();
  const { pattern } = await evaluate(code, transpiler);
  const haps = pattern.queryArc(0, numCycles);
  return haps.map((h) => sanitizeForJson({
    begin: h.whole ? h.whole.begin.valueOf() : null,
    end: h.whole ? h.whole.end.valueOf() : null,
    value: h.value,
  }));
}

globalThis.__getHaps = getHaps;
