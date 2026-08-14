from __future__ import annotations

import re
from datetime import datetime, timedelta, timezone
from typing import Any


STALE_RUNNING_AFTER = timedelta(minutes=5)

# Fixed sequence of steps a one-shot or chunked extraction run moves through
# (see services/extract_offer.py). 'calling_llm' and 'extracting_summary' are
# alternative names for the same phase as 'extracting_chunked', depending on
# which extraction mode ran.
EXTRACTION_STEP_ORDER = [
    'reading_pdf',
    'saving_raw_text',
    'extracting_chunked',
    'saving_llm_response',
    'validating_json',
    'normalizing_amounts',
    'saving_extract',
]
_EXTRACTION_STEP_ALIASES = {
    'calling_llm': 'extracting_chunked',
    'extracting_summary': 'extracting_chunked',
}
_CHUNK_STEP_PATTERN = re.compile(r'^extracting_posts_chunk_(\d+)_of_(\d+)$')


def extraction_progress_fraction(step: str | None) -> float | None:
    """Rough progress estimate (0..1) for an extraction job, based on its current step.

    Returns None when the step is unknown, so callers can fall back to an
    indeterminate progress indicator instead of a misleading percentage.
    """
    if not step:
        return None

    total = len(EXTRACTION_STEP_ORDER)
    chunk_match = _CHUNK_STEP_PATTERN.match(step)
    if chunk_match:
        chunk_index, chunk_total = int(chunk_match.group(1)), int(chunk_match.group(2))
        base_index = EXTRACTION_STEP_ORDER.index('extracting_chunked')
        within_phase = chunk_index / chunk_total if chunk_total else 1.0
        return min((base_index + within_phase) / total, 1.0)

    canonical_step = _EXTRACTION_STEP_ALIASES.get(step, step)
    if canonical_step not in EXTRACTION_STEP_ORDER:
        return None

    return (EXTRACTION_STEP_ORDER.index(canonical_step) + 1) / total


def is_running(status: dict[str, Any] | None) -> bool:
    return bool(status and status.get('status') == 'running')


def is_stale_running(status: dict[str, Any] | None, *, now: datetime | None = None) -> bool:
    if not is_running(status):
        return False

    updated_at = parse_status_time(status.get('updated_at'))
    if updated_at is None:
        return False

    current_time = now or datetime.now(timezone.utc)
    return current_time - updated_at > STALE_RUNNING_AFTER


def is_active_running(status: dict[str, Any] | None) -> bool:
    return is_running(status) and not is_stale_running(status)


def parse_status_time(value: Any) -> datetime | None:
    if not isinstance(value, str) or not value:
        return None

    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        return None

    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)

    return parsed.astimezone(timezone.utc)
