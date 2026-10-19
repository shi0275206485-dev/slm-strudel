import {
  die,
  keptPayload,
  parseCliArgs,
  readJsonl,
  rejectPayload,
  splitValidatedRecords,
  stratifiedSplit,
  summarizeValidation,
  timestamp,
  validateRecords,
  writeCsv,
  writeJson,
  writeJsonl,
} from './pipeline_utils.mjs';

function usage() {
  console.error(`Usage:
  node strudel_pipeline.mjs <input.jsonl> <out-dir>

What it does:
  1) Deterministically split the dataset into train/valid
  2) Run headless Strudel validation on both splits
  3) Export raw splits, validation reports, failure/warning CSVs
  4) Build cleaned train/valid JSONL files

Options:
  --train-ratio 0.9     Proportion assigned to train split.
  --seed 42             Deterministic split seed.
  --drop-warnings       Exclude warning-producing samples from clean splits.
`);
}

function reportRows(results) {
  return results.map((r) => ({
    line: r.line,
    id: r.id,
    dimension: r.dimension,
    ok: r.ok,
    stage: r.stage ?? '',
    haps_0_1: r.haps_0_1 ?? '',
    haps_0_2: r.haps_0_2 ?? '',
    warning_count: Array.isArray(r.warnings) ? r.warnings.length : 0,
    error: r.error ?? '',
    code: r.code ?? '',
    warnings: Array.isArray(r.warnings) ? r.warnings.join(' | ') : '',
  }));
}

async function validateAndWrite(splitName, records, outDir, dropWarnings) {
  const reportsDir = `${outDir}/reports`;
  const results = await validateRecords(records);
  const summary = summarizeValidation(results);
  const partition = splitValidatedRecords(records, results, { dropWarnings });

  writeJson(`${reportsDir}/${splitName}_validation.json`, results);
  writeCsv(`${reportsDir}/${splitName}_validation.csv`, reportRows(results), [
    'line',
    'id',
    'dimension',
    'ok',
    'stage',
    'haps_0_1',
    'haps_0_2',
    'warning_count',
    'error',
    'code',
    'warnings',
  ]);

  writeJsonl(
    `${outDir}/${splitName}_clean.jsonl`,
    partition.clean.map(keptPayload),
  );
  writeJsonl(
    `${outDir}/${splitName}_rejected_fail.jsonl`,
    partition.failed.map((item) => rejectPayload(item, 'failed')),
  );
  writeJsonl(
    `${outDir}/${splitName}_rejected_warn.jsonl`,
    partition.warned
      .filter((item) => item.result?.ok)
      .map((item) => rejectPayload(item, dropWarnings ? 'warning_dropped' : 'warning_only')),
  );

  writeCsv(
    `${reportsDir}/${splitName}_failures.csv`,
    partition.failed.map((item) => ({
      line: item.result?.line,
      id: item.result?.id,
      dimension: item.result?.dimension,
      stage: item.result?.stage,
      error: item.result?.error,
      code: item.result?.code,
    })),
    ['line', 'id', 'dimension', 'stage', 'error', 'code'],
  );

  writeCsv(
    `${reportsDir}/${splitName}_warnings.csv`,
    partition.warned.map((item) => ({
      line: item.result?.line,
      id: item.result?.id,
      dimension: item.result?.dimension,
      warning_count: Array.isArray(item.result?.warnings) ? item.result.warnings.length : 0,
      warnings: Array.isArray(item.result?.warnings) ? item.result.warnings.join(' | ') : '',
      code: item.result?.code,
    })),
    ['line', 'id', 'dimension', 'warning_count', 'warnings', 'code'],
  );

  return {
    summary,
    cleanCount: partition.clean.length,
    failedCount: partition.failed.length,
    warnedCount: partition.warned.length,
  };
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
  const dropWarnings = Boolean(args['drop-warnings']);

  const records = readJsonl(inputPath);
  const split = stratifiedSplit(records, { trainRatio, seed });

  writeJsonl(`${outDir}/train_raw.jsonl`, split.train);
  writeJsonl(`${outDir}/valid_raw.jsonl`, split.valid);
  writeCsv(`${outDir}/reports/split_stats.csv`, split.stats, ['dimension', 'total', 'train', 'valid']);
  writeJson(`${outDir}/reports/split_stats.json`, split.stats);

  const trainInfo = await validateAndWrite('train', split.train, outDir, dropWarnings);
  const validInfo = await validateAndWrite('valid', split.valid, outDir, dropWarnings);

  const summary = {
    createdAt: timestamp(),
    inputPath,
    outDir,
    trainRatio,
    seed,
    dropWarnings,
    inputTotal: records.length,
    trainRaw: split.train.length,
    validRaw: split.valid.length,
    trainValidation: trainInfo,
    validValidation: validInfo,
    cleanTotals: {
      train: trainInfo.cleanCount,
      valid: validInfo.cleanCount,
      total: trainInfo.cleanCount + validInfo.cleanCount,
    },
  };

  writeJson(`${outDir}/reports/summary.json`, summary);
  console.log(JSON.stringify(summary, null, 2));
}

main().catch((error) => die(error?.stack || String(error)));
