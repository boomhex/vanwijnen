"""Central logging for the app.

`setup_logging()` sends everything (app logs, user actions, tester reports,
unhandled exceptions) to stdout and to a rotating file at ``logs/app.log``.

User actions are recorded through :func:`log_action` or the
:func:`logged_action` decorator, so crashes reported by testers can be
correlated with what happened right before them.
"""
from __future__ import annotations

import functools
import logging
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any, Callable

APP_DIR = Path(__file__).resolve().parents[1]
LOG_DIR = APP_DIR / 'logs'
LOG_FILE = LOG_DIR / 'app.log'

action_logger = logging.getLogger('actions')
feedback_logger = logging.getLogger('feedback')


def setup_logging(level: int = logging.INFO) -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    formatter = logging.Formatter('%(asctime)s %(levelname)s [%(name)s] %(message)s')

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)

    file_handler = RotatingFileHandler(LOG_FILE, maxBytes=5_000_000, backupCount=5, encoding='utf-8')
    file_handler.setFormatter(formatter)

    root = logging.getLogger()
    root.setLevel(level)
    root.handlers.clear()
    root.addHandler(stream_handler)
    root.addHandler(file_handler)


def current_username() -> str:
    """'user@workspace' of the logged-in user, or 'system' outside a UI context."""
    try:
        from nicegui import app

        username = app.storage.user.get('username')
        if not username:
            return 'anonymous'
        workspace = app.storage.user.get('workspace')
        return f'{username}@{workspace}' if workspace else username
    except Exception:
        return 'system'


def log_action(action: str, **details: Any) -> None:
    action_logger.info('%s | %s%s', current_username(), action, _format_details(details))


def log_feedback(message: str) -> None:
    single_line = ' / '.join(part.strip() for part in message.splitlines() if part.strip())
    feedback_logger.warning('USER REPORT | %s | %s', current_username(), single_line)


def logged_action(func: Callable) -> Callable:
    """Log every call to the decorated method to the action log.

    Positional/keyword arguments are summarized (offers and projects by name),
    and a ``DrawerActionResult``-like return value adds its outcome.
    """

    @functools.wraps(func)
    def wrapper(self, *args, **kwargs):
        described = _describe_args(args, kwargs)
        try:
            result = func(self, *args, **kwargs)
        except Exception as error:
            action_logger.info(
                '%s | %s%s | raised %s: %s',
                current_username(), func.__name__, described, type(error).__name__, error,
            )
            raise
        action_logger.info('%s | %s%s%s', current_username(), func.__name__, described, _describe_result(result))
        return result

    return wrapper


def _format_details(details: dict[str, Any]) -> str:
    parts = [f'{key}={_describe(value)}' for key, value in details.items() if _describe(value)]
    return f' | {", ".join(parts)}' if parts else ''


def _describe_args(args: tuple, kwargs: dict[str, Any]) -> str:
    parts = [described for described in (_describe(value) for value in args) if described]
    parts += [f'{key}={described}' for key, value in kwargs.items() if (described := _describe(value))]
    return f' | {", ".join(parts)}' if parts else ''


def _describe(value: Any) -> str:
    if value is None or isinstance(value, dict):
        return ''
    if hasattr(value, 'project_name') and hasattr(value, 'name'):  # Offer
        return f'{value.project_name}/{value.name}'
    if hasattr(value, 'name') and hasattr(value, 'path'):  # Project
        return str(value.name)
    if isinstance(value, (str, int, float, bool)):
        text = str(value).replace('\n', ' ')
        return text if len(text) <= 120 else f'{text[:117]}...'
    if isinstance(value, (list, set, tuple)):
        return f'<{len(value)} items>'
    return type(value).__name__


def _describe_result(result: Any) -> str:
    success = getattr(result, 'success', None)
    message = getattr(result, 'message', None)
    if success is None and message is None:
        return ''
    parts = ['ok' if success else 'failed']
    if message:
        parts.append(str(message))
    return f' | {" ".join(parts)}'
