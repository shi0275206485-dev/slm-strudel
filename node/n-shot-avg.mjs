import { transpiler } from '@strudel/transpiler';
import fs from 'fs';
import path from 'path';

// ================= config =================
const SHOT_DIR = 'finetuned-0-shot';
// ============================================

function calcPassRate(filePath) {
    const lines = fs.readFileSync(filePath, 'utf-8')
        .split('\n').filter(l => l.trim() !== '');
    let valid = 0;
    for (const line of lines) {
        try {
            const data = JSON.parse(line);
            const code = data.messages[2].content
                .replace(/```javascript|```js|```strudel|```/g, '')
                .trim();
            transpiler(code);
            valid++;
        } catch (_) {}
    }
    return lines.length > 0 ? valid / lines.length : 0;
}

// remove  _0shot_seed*.jsonl appendix
const allFiles = fs.readdirSync(SHOT_DIR).filter(f => f.endsWith('.jsonl'));

const modelMap = {};
for (const file of allFiles) {
    const model = file.replace(/_0shot_seed\d+\.jsonl$/, '');
    if (!modelMap[model]) modelMap[model] = [];
    modelMap[model].push(file);
}

console.log(` ${Object.keys(modelMap).length} \n`);
console.log('='.repeat(60));

for (const [model, files] of Object.entries(modelMap)) {
    const rates = [];
    for (const file of files.sort()) {
        const filePath = path.join(SHOT_DIR, file);
        const rate = calcPassRate(filePath);
        const seed = file.match(/seed(\d+)/)?.[1] ?? '?';
        rates.push(rate);
        console.log(`  seed${seed.padStart(2)}: ${(rate * 100).toFixed(2)}%  (${file})`);
    }
    const avg = rates.reduce((a, b) => a + b, 0) / rates.length;
    console.log(`\n  [${model}]  Avg Pass Rate: ${(avg * 100).toFixed(2)}%`);
    console.log('-'.repeat(60));
}

console.log('\nDone.\n');

