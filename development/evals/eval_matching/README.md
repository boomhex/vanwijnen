# Matching Eval

This golden eval checks whether the deterministic candidate selector keeps the
correct offer post inside the top candidates for each comparison row. It can
also ask an LLM to choose the final match from those candidates and score that
choice against the same golden dataset.

By default it does not make LLM calls. Fuzzy matching is used for candidate
retrieval; the expected matches remain explicit and reviewable in
`testcases/*.out`.

Run:

```bash
env PYTHONPATH=development/app venv/bin/python development/evals/eval_matching/main.py
```

By default it reads:

- `testcases/*.in` for comparison rows
- `testcases/*.out` for expected matches
- the `OfferRoot` in each testcase for offer posts

Outputs are written to `runs/<timestamp>/`.
If a testcase points to missing offer extracts, the case is reported as an error
instead of failing the whole eval run.

Testcase input files look like:

```json
{
  "OfferRoot": "app/storage/test/22.31_baksteen",
  "Posten": [
    {
      "Omschrijving": "Vermetselen gevelsteen halfsteens verband",
      "Aantal": "86,95",
      "Eenheid": "dzd"
    }
  ]
}
```

Use a different offer folder:

```bash
env PYTHONPATH=development/app venv/bin/python development/evals/eval_matching/main.py \
  --offers development/app/storage/test/44.31_gipsplafonds
```

The main metric is `candidate_recall_at_8`: how often the expected match is
present in the top 8 candidates. Optimize this for recall before asking an LLM
to choose the final match.

Run the LLM final-match step:

```bash
env PYTHONPATH=development/app venv/bin/python development/evals/eval_matching/main.py \
  --with-llm \
  --run-name prompt_v1
```

This writes extra files per case:

- `*.llm_response.txt`: raw LLM response
- `*.llm_matches.json`: parsed final matches
- `*.llm_score.json`: final-match score against golden expected matches

Use the evals iteratively:

1. Run without LLM and inspect `candidate_recall_at_8`.
2. If recall is low, improve deterministic retrieval: synonyms, normalization,
   field weights, schema fields, or extraction richness.
3. Run with `--with-llm` only when the right match is usually in the candidates.
4. If candidate recall is high but LLM accuracy is low, improve the LLM prompt.
5. If both are low, improve extraction schema/output first or add better
   `MatchHints`, `Werksoort`, `PostType`, `Prijsbasis`, and `Brontekst`.
6. Keep every prompt/schema experiment in a named run, then compare reports.

Generate fixtures from an existing `comparison.json`:

```bash
env PYTHONPATH=development/app venv/bin/python development/evals/eval_matching/main.py \
  --generate-from-comparison app/storage/test/46.00_schilderwerk/comparison.json \
  --overwrite
```
