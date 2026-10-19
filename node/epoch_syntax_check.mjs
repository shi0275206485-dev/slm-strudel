// epoch_syntax_check.mjs
//
// Adapted from n-shot-avg.mjs: same transpiler(code) try/catch check, but
// takes a single input file + writes a per-item manifest, and prints a
// SYNTAX_PASS_RATE=<float> line the calling Python process can grab from
// stdout. Meant to be invoked once per epoch via subprocess from the
// training callback, not run standalone for batch reporting like the
// original script.
//
// Usage:
//   node epoch_syntax_check.mjs <input.jsonl> <output_manifest.jsonl>

import { transpiler } from '@strudel/transpiler';
import fs from 'fs';

const inputFile = process.argv[2];
const outputManifest = process.argv[3];

if (!inputFile || !outputManifest) {
    console.error('Usage: node epoch_syntax_check.mjs <input.jsonl> <output_manifest.jsonl>');
    process.exit(1);
}

const lines = fs.readFileSync(inputFile, 'utf-8')
    .split('\n').filter(l => l.trim() !== '');

let valid = 0;
const results = [];

for (let i = 0; i < lines.length; i++) {
    let data;
    try {
        data = JSON.parse(lines[i]);
    } catch (e) {
        results.push({ entry_index: i, status: 'JSON_PARSE_ERROR', message: String(e) });
        continue;
    }
    const code = data.messages[2].content
        .replace(/```javascript|```js|```strudel|```/g, '')
        .trim();
    try {
        transpiler(code);
        valid++;
        results.push({ entry_index: i, status: 'SYNTAX_OK' });
    } catch (e) {
        results.push({ entry_index: i, status: 'SYNTAX_ERROR', message: String(e && e.message ? e.message : e) });
    }
}

fs.writeFileSync(outputManifest, results.map(r => JSON.stringify(r)).join('\n') + '\n', 'utf-8');

const rate = lines.length > 0 ? valid / lines.length : 0;
console.log(`SYNTAX_PASS_RATE=${rate}`);
console.log(`SYNTAX_VALID=${valid}`);
console.log(`SYNTAX_TOTAL=${lines.length}`);
