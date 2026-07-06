# Matching Eval

This eval checks whether the deterministic candidate selector keeps the correct
offer post inside the top candidates for each comparison row.

Run:

```bash
env PYTHONPATH=development/app venv/bin/python development/evals/eval_matching/main.py
```

By default it reads:

- `testcases/*.in` for comparison rows
- `testcases/*.out` for expected matches
- the `OfferRoot` in each testcase for offer posts

Outputs are written to `runs/<timestamp>/`.

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

Generate fixtures from an existing `comparison.json`:

```bash
env PYTHONPATH=development/app venv/bin/python development/evals/eval_matching/main.py \
  --generate-from-comparison app/storage/test/46.00_schilderwerk/comparison.json \
  --overwrite
```
