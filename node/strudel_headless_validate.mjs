import {
  die,
  parseCliArgs,
  readJsonl,
  summarizeValidation,
  validateRecords,
  writeJson,
} from './pipeline_utils.mjs';

function usage() {
  console.error(`Usage:
  node strudel_headless_validate.mjs <input.jsonl> <output.json>

Options:
  --pretty-summary   Print a compact human-readable summary.
`);
}

async function main() {
  const args = parseCliArgs(process.argv);
  const [inputPath, outputPath] = args._;

  if (!inputPath || !outputPath) {
    usage();
    process.exit(1);
  }

  const records = readJsonl(inputPath);
  const results = await validateRecords(records);
  writeJson(outputPath, results);

  const summary = summarizeValidation(results);
  if (args['pretty-summary']) {
    console.log(`total=${summary.total} pass=${summary.pass} fail=${summary.fail} warnings=${summary.withWarnings}`);
  } else {
    console.log(JSON.stringify({ inputPath, outputPath, ...summary }, null, 2));
  }
}

main().catch((error) => die(error?.stack || String(error)));
