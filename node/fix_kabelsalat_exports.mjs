// fix_kabelsalat_exports.mjs
//
// @strudel/core imports @kabelsalat/web, whose package.json points Node's
// resolver ("main") at a browser IIFE build without ES exports; the ES build
// is only named in the bundler-only "module" field. Under plain Node this
// makes `import '@strudel/transpiler'` fail with
//   "does not provide an export named 'SalatRepl'".
// This script adds an "exports" map pointing to the ES build. It runs
// automatically after `npm ci` / `npm install` (see "postinstall").
import fs from 'fs';

const p = 'node_modules/@kabelsalat/web/package.json';
if (!fs.existsSync(p)) process.exit(0);
const pkg = JSON.parse(fs.readFileSync(p, 'utf8'));
if (!pkg.exports) {
  pkg.exports = { '.': { import: './dist/index.mjs', require: './dist/index.js' } };
  fs.writeFileSync(p, JSON.stringify(pkg, null, 2) + '\n');
  console.log('fix_kabelsalat_exports: added "exports" to', p);
}
