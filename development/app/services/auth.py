"""User store and workspace authorization for the login page.

Passwords are stored as scrypt hashes in ``users.json`` (never plain text).
Each user also carries the list of workspaces (top-level folders under
``storage/``) they may open. Manage both on the host from the app directory:

    python -m services.auth add <username>
    python -m services.auth remove <username>
    python -m services.auth grant <username> <workspace>
    python -m services.auth revoke <username> <workspace>
    python -m services.auth list
"""
from __future__ import annotations

import getpass
import hashlib
import hmac
import json
import logging
import os
import re
import secrets
import sys
from pathlib import Path

logger = logging.getLogger(__name__)

APP_DIR = Path(__file__).resolve().parents[1]
USERS_FILE = Path(os.environ.get('VANWIJNEN_USERS_FILE', APP_DIR / 'users.json'))

_SCRYPT_PARAMS = {'n': 2 ** 14, 'r': 8, 'p': 1}

# Usernames appear in log lines and session data: 1-32 safe chars,
# alphanumeric at both ends. Workspace names become directory names under
# storage/ and URL path segments: same characters, up to 64 chars.
_USERNAME_PATTERN = re.compile(r'[a-z0-9](?:[a-z0-9._-]{0,30}[a-z0-9])?')
_WORKSPACE_PATTERN = re.compile(r'[a-z0-9](?:[a-z0-9._-]{0,62}[a-z0-9])?')


def normalize_username(username: str | None) -> str:
    return (username or '').strip().lower()


def is_valid_username(username: str | None) -> bool:
    return bool(username) and _USERNAME_PATTERN.fullmatch(username) is not None


def normalize_workspace(workspace: str | None) -> str:
    return (workspace or '').strip().lower()


def is_valid_workspace_name(workspace: str | None) -> bool:
    return bool(workspace) and _WORKSPACE_PATTERN.fullmatch(workspace) is not None


def load_users() -> dict[str, dict[str, str]]:
    if not USERS_FILE.exists():
        return {}
    try:
        return json.loads(USERS_FILE.read_text(encoding='utf-8'))
    except (OSError, ValueError):
        logger.exception('Could not read users file %s', USERS_FILE)
        return {}


def save_users(users: dict[str, dict[str, str]]) -> None:
    USERS_FILE.write_text(json.dumps(users, indent=2), encoding='utf-8')
    USERS_FILE.chmod(0o600)


def has_users() -> bool:
    return bool(load_users())


def add_user(username: str | None, password: str) -> str:
    username = normalize_username(username)
    if not is_valid_username(username):
        raise ValueError(
            'Invalid username: use 1-32 letters, digits, ".", "_" or "-", starting and ending with a letter or digit'
        )
    if not password:
        raise ValueError('Password cannot be empty')

    salt = secrets.token_bytes(16)
    digest = hashlib.scrypt(password.encode('utf-8'), salt=salt, **_SCRYPT_PARAMS)
    users = load_users()
    existing_workspaces = users.get(username, {}).get('workspaces', [])
    users[username] = {'salt': salt.hex(), 'hash': digest.hex(), 'workspaces': existing_workspaces}
    save_users(users)
    return username


def remove_user(username: str | None) -> bool:
    username = normalize_username(username)
    users = load_users()
    if username not in users:
        return False
    del users[username]
    save_users(users)
    return True


def user_workspaces(username: str | None) -> list[str]:
    """Workspaces the user is authorized to open, in granted order."""
    entry = load_users().get(normalize_username(username), {})
    return [workspace for workspace in entry.get('workspaces', []) if is_valid_workspace_name(workspace)]


def workspace_authorized(username: str | None, workspace: str | None) -> bool:
    return is_valid_workspace_name(workspace) and workspace in user_workspaces(username)


def workspace_claimed(workspace: str | None) -> bool:
    """Whether any user already has this workspace granted."""
    workspace = normalize_workspace(workspace)
    return any(workspace in entry.get('workspaces', []) for entry in load_users().values())


def grant_workspace(username: str | None, workspace: str | None) -> str:
    username = normalize_username(username)
    workspace = normalize_workspace(workspace)
    if not is_valid_workspace_name(workspace):
        raise ValueError(
            'Invalid workspace name: use 1-64 letters, digits, ".", "_" or "-", '
            'starting and ending with a letter or digit'
        )

    users = load_users()
    if username not in users:
        raise ValueError(f'User "{username}" does not exist')

    workspaces = users[username].setdefault('workspaces', [])
    if workspace not in workspaces:
        workspaces.append(workspace)
        save_users(users)
    return workspace


def revoke_workspace(username: str | None, workspace: str | None) -> bool:
    username = normalize_username(username)
    workspace = normalize_workspace(workspace)
    users = load_users()
    workspaces = users.get(username, {}).get('workspaces', [])
    if workspace not in workspaces:
        return False
    workspaces.remove(workspace)
    save_users(users)
    return True


def verify_user(username: str | None, password: str | None) -> bool:
    entry = load_users().get(normalize_username(username))
    if not entry or not password:
        return False
    try:
        salt = bytes.fromhex(entry['salt'])
        expected = bytes.fromhex(entry['hash'])
    except (KeyError, ValueError):
        return False
    digest = hashlib.scrypt(password.encode('utf-8'), salt=salt, **_SCRYPT_PARAMS)
    return hmac.compare_digest(digest, expected)


def _main(argv: list[str]) -> int:
    if len(argv) >= 2 and argv[0] == 'add':
        password = argv[2] if len(argv) >= 3 else getpass.getpass(f'Password for {argv[1]}: ')
        if len(argv) < 3 and password != getpass.getpass('Repeat password: '):
            print('Passwords do not match')
            return 1
        try:
            username = add_user(argv[1], password)
        except ValueError as error:
            print(error)
            return 1
        print(f'Saved user "{username}" to {USERS_FILE}')
        return 0

    if len(argv) >= 2 and argv[0] == 'remove':
        if remove_user(argv[1]):
            print(f'Removed user "{normalize_username(argv[1])}"')
            return 0
        print(f'User "{normalize_username(argv[1])}" not found')
        return 1

    if len(argv) >= 3 and argv[0] in ('grant', 'revoke'):
        try:
            if argv[0] == 'grant':
                workspace = grant_workspace(argv[1], argv[2])
                print(f'Granted "{normalize_username(argv[1])}" access to workspace "{workspace}"')
                return 0
            if revoke_workspace(argv[1], argv[2]):
                print(f'Revoked workspace "{normalize_workspace(argv[2])}" from "{normalize_username(argv[1])}"')
                return 0
            print(f'"{normalize_username(argv[1])}" has no access to "{normalize_workspace(argv[2])}"')
            return 1
        except ValueError as error:
            print(error)
            return 1

    if argv and argv[0] == 'list':
        users = load_users()
        for username in sorted(users):
            workspaces = ', '.join(users[username].get('workspaces', [])) or '(no workspaces)'
            print(f'{username}: {workspaces}')
        if not users:
            print(f'No users in {USERS_FILE}')
        return 0

    print(
        'Usage: python -m services.auth '
        'add <username> [password] | remove <username> | grant <username> <workspace> | '
        'revoke <username> <workspace> | list'
    )
    return 1


if __name__ == '__main__':
    sys.exit(_main(sys.argv[1:]))
