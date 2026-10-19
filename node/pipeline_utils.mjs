import fs from 'fs';
import path from 'path';
import process from 'process';

import { validate } from './strudel-validator.mjs';

export function ensureDir(dirPath) {
  fs.mkdirSync(dirPath, { recursive: true });
}

export function readJsonl(filePath) {
  return fs.readFileSync(filePath, 'utf8')
    .split(/\r?\n/)
    .filter((line) => line.trim().length > 0)
    .map((line, index) => {
      try {
        return JSON.parse(line);
      } catch (error) {
        const lineNo = index + 1;
        throw new Error(`Invalid JSON on line ${lineNo} in ${filePath}: ${error.message}`);
      }
    });
}

export function writeJsonl(filePath, rows) {
  ensureDir(path.dirname(path.resolve(filePath)));
  const text = rows.map((row) => JSON.stringify(row)).join('\n') + (rows.length ? '\n' : '');
  fs.writeFileSync(filePath, text, 'utf8');
}

function contentToString(content) {
  if (typeof content === 'string') return content;
  if (Array.isArray(content)) {
    return content
      .map((part) => {
        if (typeof part === 'string') return part;
        if (typeof part?.text === 'string') return part.text;
        return '';
      })
      .filter(Boolean)
      .join('\n');
  }
  if (typeof content?.text === 'string') return content.text;
  return '';
}

export function extractCode(record) {
  if (typeof record?.output === 'string' && record.output.trim()) {
    return record.output.trim();
  }
  if (Array.isArray(record?.messages)) {
    const assistant = [...record.messages].reverse().find((m) => m?.role === 'assistant');
    const content = contentToString(assistant?.content);
    if (content.trim()) return content.trim();
  }
  return null;
}

function firstMatchingLine(text, prefix) {
  const line = String(text || '')
    .split(/\r?\n/)
    .find((x) => x.toLowerCase().startsWith(prefix.toLowerCase()));
  return line ? line.slice(prefix.length).trim() : null;
}

export function extractDimension(record) {
  if (typeof record?.dimension === 'string' && record.dimension.trim()) {
    return record.dimension.trim();
  }

  if (typeof record?.input === 'string') {
    const dim = firstMatchingLine(record.input, 'Dimension:');
    if (dim) return dim;
  }

  if (Array.isArray(record?.messages)) {
    const user = record.messages.find((m) => m?.role === 'user');
    const content = contentToString(user?.content);
    const dim = firstMatchingLine(content, 'Dimension:');
    if (dim) return dim;
  }

  return 'Unknown';
}

export function extractId(record, lineNo = null) {
  if (typeof record?.id === 'string' && record.id.trim()) return record.id.trim();
  if (typeof lineNo === 'number') return `line-${lineNo}`;
  return `item-${Math.random().toString(36).slice(2, 10)}`;
}

function mulberry32(seed) {
  let t = seed >>> 0;
  return function rng() {
    t += 0x6D2B79F5;
    let r = Math.imul(t ^ (t >>> 15), 1 | t);
    r ^= r + Math.imul(r ^ (r >>> 7), 61 | r);
    return ((r ^ (r >>> 14)) >>> 0) / 4294967296;
  };
}

function hashStringToInt(text) {
  let h = 2166136261;
  for (const ch of String(text)) {
    h ^= ch.charCodeAt(0);
    h = Math.imul(h, 16777619);
  }
  return h >>> 0;
}

function shuffledCopy(items, seedText) {
  const arr = [...items];
  const rng = mulberry32(hashStringToInt(seedText));
  for (let i = arr.length - 1; i > 0; i -= 1) {
    const j = Math.floor(rng() * (i + 1));
    [arr[i], arr[j]] = [arr[j], arr[i]];
  }
  return arr;
}

export function stratifiedSplit(records, options = {}) {
  const trainRatio = Number.isFinite(options.trainRatio) ? options.trainRatio : 0.9;
  const seed = options.seed ?? 42;

  if (!(trainRatio > 0 && trainRatio < 1)) {
    throw new Error(`trainRatio must be between 0 and 1, got ${trainRatio}`);
  }

  const groups = new Map();
  records.forEach((record, idx) => {
    const dimension = extractDimension(record) || 'Unknown';
    if (!groups.has(dimension)) groups.set(dimension, []);
    groups.get(dimension).push({ record, idx });
  });

  const train = [];
  const valid = [];
  const stats = [];

  for (const [dimension, group] of groups.entries()) {
    const shuffled = shuffledCopy(group, `${seed}:${dimension}`);
    let trainCount = Math.floor(shuffled.length * trainRatio);

    if (shuffled.length > 1) {
      if (trainCount === 0) trainCount = 1;
      if (trainCount === shuffled.length) trainCount = shuffled.length - 1;
    }

    const trainPart = shuffled.slice(0, trainCount);
    const validPart = shuffled.slice(trainCount);

    train.push(...trainPart);
    valid.push(...validPart);
    stats.push({
      dimension,
      total: shuffled.length,
      train: trainPart.length,
      valid: validPart.length,
    });
  }

  train.sort((a, b) => a.idx - b.idx);
  valid.sort((a, b) => a.idx - b.idx);

  return {
    train: train.map((x) => x.record),
    valid: valid.map((x) => x.record),
    stats: stats.sort((a, b) => a.dimension.localeCompare(b.dimension)),
  };
}

function normalizeWarning(value) {
  if (value instanceof Error) return value.stack || value.message || String(value);
  if (typeof value === 'string') return value;
  try {
    return JSON.stringify(value);
  } catch {
    return String(value);
  }
}

export async function validateOneCode(code) {
  const validation = validate(code);
  return {
    ok: validation.valid,
    stage: 'syntax',
    error: validation.valid ? null : validation.errors.map((err) => `${err.type}:${err.line}:${err.column} ${err.message}`).join(' | '),
    errors: validation.errors,
    warnings: [],
  };
}

export async function validateRecords(records) {
  const results = [];

  for (let i = 0; i < records.length; i += 1) {
    const record = records[i];
    const id = extractId(record, i + 1);
    const code = extractCode(record);
    const dimension = extractDimension(record);

    if (!code) {
      results.push({
        line: i + 1,
        id,
        dimension,
        code: null,
        ok: false,
        stage: 'extract',
        error: 'No Strudel code found in obj.output or assistant message.',
        warnings: [],
      });
      continue;
    }

    const validation = await validateOneCode(code);
    results.push({
      line: i + 1,
      id,
      dimension,
      code,
      ...validation,
    });
  }

  return results;
}

export function escapeCsvCell(value) {
  const text = value == null ? '' : String(value);
  if (/[",\n]/.test(text)) {
    return `"${text.replaceAll('"', '""')}"`;
  }
  return text;
}

export function writeCsv(filePath, rows, columns) {
  ensureDir(path.dirname(path.resolve(filePath)));
  const header = columns.join(',');
  const body = rows.map((row) => columns.map((col) => escapeCsvCell(row?.[col])).join(','));
  const text = [header, ...body].join('\n') + '\n';
  fs.writeFileSync(filePath, text, 'utf8');
}

export function summarizeValidation(results) {
  const total = results.length;
  const pass = results.filter((r) => r.ok).length;
  const fail = total - pass;
  const withWarnings = results.filter((r) => Array.isArray(r.warnings) && r.warnings.length > 0).length;
  return { total, pass, fail, withWarnings };
}

export function splitValidatedRecords(records, results, options = {}) {
  const dropWarnings = Boolean(options.dropWarnings);
  const clean = [];
  const failed = [];
  const warned = [];

  for (let i = 0; i < records.length; i += 1) {
    const record = records[i];
    const result = results[i];
    const hasWarnings = Array.isArray(result?.warnings) && result.warnings.length > 0;

    if (!result?.ok) {
      failed.push({ record, result });
      continue;
    }
    if (dropWarnings && hasWarnings) {
      warned.push({ record, result });
      continue;
    }
    clean.push({ record, result });
    if (hasWarnings) warned.push({ record, result });
  }

  return { clean, failed, warned };
}

export function parseCliArgs(argv) {
  const args = { _: [] };
  for (let i = 2; i < argv.length; i += 1) {
    const token = argv[i];
    if (!token.startsWith('--')) {
      args._.push(token);
      continue;
    }
    const key = token.slice(2);
    const next = argv[i + 1];
    if (next == null || next.startsWith('--')) {
      args[key] = true;
    } else {
      args[key] = next;
      i += 1;
    }
  }
  return args;
}

export function writeJson(filePath, value) {
  ensureDir(path.dirname(path.resolve(filePath)));
  fs.writeFileSync(filePath, JSON.stringify(value, null, 2) + '\n', 'utf8');
}

export function rejectPayload(item, reason) {
  return {
    ...item.record,
    _validation: {
      reason,
      id: item.result?.id ?? null,
      line: item.result?.line ?? null,
      ok: item.result?.ok ?? false,
      stage: item.result?.stage ?? null,
      error: item.result?.error ?? null,
      warnings: item.result?.warnings ?? [],
    },
  };
}

export function keptPayload(item) {
  return item.record;
}

export function timestamp() {
  return new Date().toISOString();
}

export function die(message) {
  console.error(message);
  process.exit(1);
}
