import os
import secrets
from pathlib import Path

from nicegui import app, ui

from utils.app_logging import setup_logging

setup_logging()

import interface.login_page  # noqa: F401  registers /login
import interface.workspace_page  # noqa: F401  registers /workspaces
from interface.auth_middleware import AuthMiddleware
from interface.main_page import main_page  # noqa: F401  registers /

APP_DIR = Path(__file__).resolve().parent

app.add_middleware(AuthMiddleware)


def storage_secret() -> str:
    """Secret signing the session cookies; a new secret logs everyone out."""
    from_env = os.environ.get('STORAGE_SECRET')
    if from_env:
        return from_env

    secret_file = APP_DIR / '.storage_secret'
    if not secret_file.exists():
        secret_file.write_text(secrets.token_hex(32), encoding='utf-8')
        secret_file.chmod(0o600)
    return secret_file.read_text(encoding='utf-8').strip()


ui.run(
    title='AI Offerte Vergelijking',
    favicon=APP_DIR / 'images' / 'vw.png',
    storage_secret=storage_secret(),
    port=int(os.environ.get('PORT', '8080')),
    show=False,  # served over the network; don't pop a browser on the host
)
