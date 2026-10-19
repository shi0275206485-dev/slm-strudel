import {
  die,
  parseCliArgs,
  readJsonl,
  stratifiedSplit,
  writeCsv,
  writeJson,
  writeJsonl,
} from './pipeline_utils.mjs';

function usage() {
  console.error(`Usage:
  node strudel_split_jsonl.mjs <input.jsonl> <out-dir>

Options:
  --train-ratio 0.9   Proportion assigned to train split.
  --seed 42           Deterministic split seed.
`);
}

async function main() {
  const args = parseCliArgs(process.argv);
  const [inputPath, outDir] = args._;
  if (!inputPath || !outDir) {
    usage();
    process.exit(1);
  }

  const trainRatio = args['train-ratio'] != null ? Number(args['train-ratio']) : 0.9;
  const seed = args.seed != null ? Number(args.seed) : 42;

  const records = readJsonl(inputPath);
  const split = stratifiedSplit(records, { trainRatio, seed });

  writeJsonl(`${outDir}/train_raw.jsonl`, split.train);
  writeJsonl(`${outDir}/valid_raw.jsonl`, split.valid);
  writeJson(`${outDir}/split_stats.json`, {
    inputPath,
    outDir,
    trainRatio,
    seed,
    total: records.length,
    train: split.train.length,
    valid: split.valid.length,
    dimensions: split.stats,
  });
  writeCsv(`${outDir}/split_stats.csv`, split.stats, ['dimension', 'total', 'train', 'valid']);

  console.log(JSON.stringify({
    inputPath,
    outDir,
    trainRatio,
    seed,
    total: records.length,
    train: split.train.length,
    valid: split.valid.length,
  }, null, 2));
}

main().catch((error) => die(error?.stack || String(error)));
