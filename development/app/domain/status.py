from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any


STALE_RUNNING_AFTER = timedelta(minutes=5)


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
