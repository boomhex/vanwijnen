# Extraction Eval

This golden eval compares offer extraction output against golden snapshots.

It does not make LLM calls or re-extract PDFs. Fuzzy matching is only used to
pair expected posts with actual posts for scoring; the expected extraction data
remains explicit and reviewable in `testcases/*.json`.

Run:

```bash
env PYTHONPATH=development/app venv/bin/python development/evals/eval_posten/main.py
```

Each testcase in `testcases/*.json` contains:

- `ExtractPath`: the current actual `extract.json` to evaluate
- `Expected`: the golden extraction result to compare against

The scorer uses fuzzy matching to pair expected posts with actual posts, but the
expected data remains explicit and reviewable in the golden testcase files.

Generate a golden testcase from an existing extract:

```bash
env PYTHONPATH=development/app venv/bin/python development/evals/eval_posten/main.py \
  --generate-from-extract app/storage/test/22.31_baksteen/postma/extract.json \
  --overwrite
```

Primary metrics:

- `post_recall`: expected posts found in the actual extraction
- `post_precision`: actual posts that match expected posts
- `field_accuracy`: checked post fields that match after normalization
- `rich_schema.coverage`: coverage of the newer matching-oriented post fields
