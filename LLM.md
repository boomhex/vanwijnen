# LLM Guide

This repository is a Python application for extracting structured data from
Dutch construction offer PDFs and comparing those extracted offer posts against
a project comparison sheet. Treat the JSON files in `development/app/storage/`
as the main application data contract.

## Quick Start

Run tests:

```bash
env PYTHONPATH=development/app pytest development/app/tests
```

Run the NiceGUI app:

```bash
cd development/app
env PYTHONPATH=. python main.py
```

The app listens on `PORT` or `8080` by default. It requires a user in
`development/app/users.json`; manage users from `development/app` with:

```bash
env PYTHONPATH=. python -m services.auth add <username>
env PYTHONPATH=. python -m services.auth grant <username> <workspace>
```

Run extraction evals:

```bash
env PYTHONPATH=development/app python development/evals/eval_posten/main.py
```

Run matching evals:

```bash
env PYTHONPATH=development/app python development/evals/eval_matching/main.py
```

Use the repository's existing virtual environment path if present, for example
`venv/bin/python`, as shown in the eval READMEs.

## Environment

Dependencies are listed in `requirements.txt`: `pdfplumber`, `pdfminer.six`,
`openai`, and `nicegui`. The current extraction and matching code imports the
Google GenAI SDK too, so a working runtime may also need `google-genai` and
`python-dotenv` installed even though they are not currently listed.

Important environment variables:

- `GEMINI_API_KEY`: required for LLM calls.
- `GEMINI_MODEL`: fallback model for generic LLM calls, default
  `gemini-2.5-flash-lite`.
- `EXTRACT_OFFER_MODEL`: extraction model, default `gemini-3.5-flash`.
- `EXTRACT_MODE`: `auto`, `chunked`, or `one_shot`; default `auto`.
- `EXTRACT_MAX_OUTPUT_TOKENS`, `EXTRACT_POST_CHUNK_MAX_OUTPUT_TOKENS`,
  `EXTRACT_SUMMARY_MAX_OUTPUT_TOKENS`: token limits for extraction calls.
- `EXTRACT_CHUNKED_THRESHOLD_CHARS`, `EXTRACT_POST_CHUNK_CHARS`,
  `EXTRACT_CHUNK_OVERLAP_LINES`: chunking controls.
- `STORAGE_SECRET`: NiceGUI session signing secret.
- `VANWIJNEN_USERS_FILE`: optional override for the user/password store.

`services.extract_offer` reads prompt files with relative paths such as
`./prompts/extract_prompt.txt`, so run direct extraction commands from
`development/app` unless you also adjust the working directory.

## Data Model

Storage is rooted at `development/app/storage/`. In the UI each logged-in user
selects a workspace, and projects live under:

```text
storage/<workspace>/<project>/<offer>/
```

Core files:

- `document.pdf`: original offer PDF.
- `raw.txt`: text extracted from the PDF by `pdfplumber`.
- `extract.json`: structured extraction result for one offer.
- `llm_response.txt`: raw extraction LLM response for debugging.
- `llm_summary_response.txt` and `llm_posts_chunk_<n>_response.txt`: chunked
  extraction debug artifacts.
- `status.json`: extraction job status for an offer.
- `comparison.json`: project-level comparison rows and matched results.
- `comparison_status.json`: project-level matching/recalculation status.
- `comparison_llm_response.txt`: raw comparison matching LLM response.

The JSON schema intentionally uses Dutch field names. Important top-level offer
fields are:

- `Naam aannemer`
- `Totaalprijs inc. BTW`
- `Totaalprijs exc. BTW`
- `Posten`

Each extracted post should preserve at least these stable fields:

- `Omschrijving`
- `Beschrijving`
- `Categorie`
- `Aantal`
- `Eenheid`
- `Eenheidsprijs`
- `Totaalbedrag`

The richer extraction schema may also include matching-oriented fields such as
`PostType`, `Status`, `Code`, `Regelnummer`, `Subcategorie`, `Werksoort`,
`Prijsbasis`, `Inclusief`, `Exclusief`, `Voorwaarden`, `DoorOpdrachtgever`,
`MatchHints`, and `Brontekst`. `domain.offer.Posten` preserves unknown fields in
`extra`, so do not drop them when editing offer rows.

Use `ONBEKEND` for unknown values. Money and amount normalization lives in
`domain.money` and `services.extract_offer.parse_money_value`.

## Architecture

`development/app/main.py` is the NiceGUI entry point. It registers pages,
adds `AuthMiddleware`, configures static `/storage` file serving, and starts
the app.

`development/app/domain/` contains small domain objects and deterministic
formatting/parsing helpers:

- `offer.py`: `Offer` and `Posten`.
- `project.py`: a pure project folder representation.
- `comparison.py`: lightweight comparison dataclasses.
- `money.py`, `units.py`, `comparison_checks.py`: deterministic business rules.
- `fields.py`: canonical editable field lists and field-to-attribute mapping.

`development/app/services/` is infrastructure and orchestration:

- `folder_handler.py`: all storage path mapping and JSON/PDF artifact
  persistence.
- `project.py`: service-backed `Project` subclass with `offers()`,
  `load_comparison()`, `save_comparison()`, rename, and delete.
- `extract_offer.py`: PDF text extraction, prompt execution, JSON parsing,
  chunked extraction, validation, normalization, and artifact writes.
- `comparison_matcher.py`: LLM comparison matching facade.
- `auth.py`: password hashing and workspace authorization CLI.

`development/app/application/` holds UI-facing application services:

- `OfferService`: loads/saves extraction results and mutates offer rows.
- `ProjectService`: lists, creates, renames, deletes projects.
- `ComparisonService`: mutates comparison rows, matched cells, warnings,
  matching status, recalculation, and saved comparison output.
- `ExtractionJobService`: runs extraction in background asyncio tasks.

`development/app/matching/` contains the comparison matching pipeline:

- `comparison_prompt.py`: builds the LLM prompt and response schema.
- `match_normalizer.py`: converts LLM output into the stable `MatchedPosten`
  shape and fills missing offer entries.
- `match_fields.py`, `match_lookup.py`: map LLM matches back to extracted posts.
- `match_calculation.py`: recalculates totals from comparison quantities.
- `match_deduplicator.py`: removes duplicate offer matches across rows.

`development/app/interface/` contains the NiceGUI UI. The left drawer handles
workspace/project/offer navigation and actions. The right side renders offer
details and project comparisons, including editable tables.

`development/evals/` contains offline eval harnesses. They do not modify the
main app flow, but they are the best safety net when changing extraction or
matching behavior.

## Main Flows

Offer extraction:

1. A PDF is uploaded through `FolderHandler.add_uploaded_file`.
2. The PDF becomes `storage/<workspace>/<project>/<offer>/document.pdf`.
3. `ExtractionJobService` starts `services.extract_offer.extract_offer`.
4. `pdfplumber` extracts raw text and saves `raw.txt`.
5. The app calls Gemini with prompts from `development/app/prompts/`.
6. Long PDFs use chunked summary plus posts extraction in `auto` mode.
7. Responses are parsed, truncated JSON can be partially recovered, non-price
   context posts are filtered, money fields are normalized, and `extract.json`
   is saved.
8. `status.json` records progress or failure.

Comparison matching:

1. A project `comparison.json` contains manual comparison `Posten`.
2. `ComparisonService.match_project_posts` sets status to running.
3. `ComparisonMatcher.project_offer_results` loads every offer `extract.json`
   in the project.
4. `matching.comparison_prompt.build_comparison_match_prompt` sends comparison
   posts plus reduced offer post context to the LLM.
5. `match_normalizer.normalize_matched_posts` produces `MatchedPosten`, one row
   per comparison post and one offer entry per offer.
6. Totals are recalculated with `matching.match_calculation.calculate_offer_total`.
7. `comparison.json` and `comparison_status.json` are saved.

Manual editing:

- Editing comparison `Posten` clears stale `MatchedPosten` and `Matches`.
- Editing matched offer descriptions looks up selected extracted posts and
  refreshes unit, category, unit price, and total fields.
- Group matches use `Gematchte posten` and usually keep summed/fallback totals.

## Prompt Files

Prompt files live in `development/app/prompts/`:

- `extract_prompt.txt`: one-shot full offer extraction.
- `extract_summary_prompt.txt`: summary fields for chunked extraction.
- `extract_posts_chunk_prompt.txt`: post extraction for one chunk.
- `comparison_match_prompt.txt`: match project comparison rows to offer posts.

When changing prompts, run the appropriate eval. For extraction prompts, run
`eval_posten`; for comparison prompts, first check deterministic candidate
recall with `eval_matching` without LLM, then use `--with-llm` when needed.

## Tests And Evals

Unit tests are in `development/app/tests/` and cover:

- money parsing and formatting;
- extraction JSON recovery, chunk splitting, and duplicate merging;
- auth and middleware behavior;
- comparison matching normalization and recalculation.

Eval docs:

- `development/evals/eval_posten/README.md`
- `development/evals/eval_matching/README.md`

The extraction eval compares existing `extract.json` files against golden
snapshots and does not make LLM calls. The matching eval checks whether the
expected offer post stays in the top candidates; it can optionally call an LLM
for final matching with `--with-llm`.

## Change Guidance For Future LLMs

- Preserve the Dutch JSON keys unless changing every reader/writer/eval.
- Keep filesystem writes behind `FolderHandler` where possible.
- Keep domain modules deterministic and side-effect free.
- Add or update tests for changes in money parsing, unit logic, matching
  normalization, extraction recovery, or auth.
- Do not delete generated storage artifacts unless the user asks. They are
  useful fixtures and debugging traces.
- Treat prompts, schemas, normalizers, and eval expectations as one system:
  update them together when extraction or matching output changes.
- Prefer improving deterministic normalization/candidate logic before relying
  on a prompt-only fix.
- Be careful with imports. Most commands expect `PYTHONPATH=development/app`
  from repo root, while direct app execution expects `PYTHONPATH=.` from
  `development/app`.
