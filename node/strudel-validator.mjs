// strudel-validator.mjs
//
// Validates Strudel code by running the same two parsers Strudel uses internally:
//   1. JavaScript layer:   acorn  (same parser @strudel/transpiler uses)
//   2. Mini-notation layer: krill-parser.js (peggy-generated parser from @strudel/mini)
//
// Returns: { valid: boolean, errors: [{ type, line, column, message }] }
// Line/column are 1-indexed (editor-style).

import * as acorn from 'acorn';
import { simple as walk } from 'acorn-walk';
import { parse as parseMini, SyntaxError as MiniSyntaxError } from '@strudel/mini/krill-parser.js';

/**
 * Validate Strudel source code.
 * @param {string} code
 * @returns {{ valid: boolean, errors: Array<{type: 'javascript'|'mini-notation', line: number, column: number, message: string}> }}
 */
export function validate(code) {
  const errors = [];

  // --- Layer 1: JavaScript syntax (acorn) ---
  let ast;
  try {
    ast = acorn.parse(code, {
      ecmaVersion: 'latest',
      sourceType: 'module',
      locations: true,
      allowAwaitOutsideFunction: true,
      allowReturnOutsideFunction: true,
    });
  } catch (err) {
    // acorn SyntaxError: err.loc = { line (1-indexed), column (0-indexed) }
    errors.push({
      type: 'javascript',
      line: err.loc?.line ?? 1,
      column: (err.loc?.column ?? 0) + 1,
      message: err.message.replace(/\s*\(\d+:\d+\)\s*$/, ''),
    });
    return { valid: false, errors };
  }

  // --- Layer 2: Mini-notation (krill parser) ---
  // Strudel's shapeshifter converts double-quoted strings (and plain backtick
  // strings) into mini-notation calls. We check the same set here.
  walk(ast, {
    Literal(node) {
      if (typeof node.value !== 'string') return;
      if (!node.raw?.startsWith('"')) return; // single-quoted strings are plain JS strings
      checkMini(node.value, node.loc, errors);
    },
    TemplateLiteral(node) {
      // Only validate template literals with no interpolation
      if (node.expressions.length > 0) return;
      if (node.quasis.length !== 1) return;
      checkMini(node.quasis[0].value.cooked, node.loc, errors);
    },
  });

  return { valid: errors.length === 0, errors };
}

function checkMini(str, loc, errors) {
  // Empty strings are valid in Strudel (they become silence)
  if (str === '') return;
  // The krill grammar's `mini` rule expects the surrounding quotes to be
  // part of the input, so we wrap the string the same way @strudel/mini's
  // `m()` function does internally.
  try {
    parseMini(`"${str}"`);
  } catch (err) {
    if (!(err instanceof MiniSyntaxError)) {
      errors.push({
        type: 'mini-notation',
        line: loc.start.line,
        column: loc.start.column + 1,
        message: String(err.message ?? err),
      });
      return;
    }
    // peggy SyntaxError: err.location.start = { line (1-indexed), column (1-indexed) }
    // Because we wrapped the string with a leading `"`, peggy's column 1 is
    // that virtual quote; column 2 is the first real char of the mini string.
    // In the source, `loc.start.column` (0-indexed) points at the real opening
    // quote. So source column of the first real char is `loc.start.column + 2`
    // (1-indexed), which generalises to `loc.start.column + inner.column`.
    const inner = err.location?.start ?? { line: 1, column: 1 };
    const absLine = loc.start.line + inner.line - 1;
    const absColumn = inner.line === 1
      ? loc.start.column + inner.column
      : inner.column;
    errors.push({
      type: 'mini-notation',
      line: absLine,
      column: absColumn,
      message: err.message,
    });
  }
}

/**
 * Convenience: format the result as a human-readable string.
 * @param {ReturnType<typeof validate>} result
 * @returns {string}
 */
export function formatResult(result) {
  if (result.valid) return '✓ valid';
  return result.errors
    .map((e) => `✗ [${e.type}] ${e.line}:${e.column} — ${e.message}`)
    .join('\n');
}
