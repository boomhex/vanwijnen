import pytest

from services import auth


@pytest.fixture()
def users_file(tmp_path, monkeypatch):
    file = tmp_path / 'users.json'
    monkeypatch.setattr(auth, 'USERS_FILE', file)
    return file


def test_add_and_verify_user(users_file):
    auth.add_user('Timo', 'geheim123')

    assert auth.verify_user('timo', 'geheim123')
    assert auth.verify_user(' Timo ', 'geheim123')
    assert not auth.verify_user('timo', 'wrong')
    assert not auth.verify_user('timo', '')
    assert not auth.verify_user('unknown', 'geheim123')


def test_password_is_not_stored_in_plain_text(users_file):
    auth.add_user('timo', 'geheim123')

    assert 'geheim123' not in users_file.read_text(encoding='utf-8')


def test_remove_user(users_file):
    auth.add_user('timo', 'geheim123')

    assert auth.remove_user('timo')
    assert not auth.verify_user('timo', 'geheim123')
    assert not auth.remove_user('timo')


def test_empty_username_or_password_rejected(users_file):
    with pytest.raises(ValueError):
        auth.add_user('   ', 'secret')
    with pytest.raises(ValueError):
        auth.add_user('timo', '')


def test_unsafe_usernames_rejected(users_file):
    for username in ['timo jolman', 'timo/j', '..', '.', '-timo', 'timo-', 'a' * 33]:
        with pytest.raises(ValueError):
            auth.add_user(username, 'secret')


def test_safe_usernames_accepted(users_file):
    assert auth.add_user('Timo.Jolman-2 ', 'secret') == 'timo.jolman-2'
    assert auth.add_user('t', 'secret') == 't'
    assert auth.is_valid_username('timo.jolman-2')
    assert not auth.is_valid_username('timo/j')


def test_grant_and_revoke_workspace(users_file):
    auth.add_user('timo', 'secret')

    assert auth.grant_workspace('timo', 'Team1 ') == 'team1'
    auth.grant_workspace('timo', 'team2')
    auth.grant_workspace('timo', 'team1')  # granting twice is fine

    assert auth.user_workspaces('timo') == ['team1', 'team2']
    assert auth.workspace_authorized('timo', 'team1')
    assert not auth.workspace_authorized('timo', 'team3')
    assert not auth.workspace_authorized('unknown', 'team1')

    assert auth.revoke_workspace('timo', 'team1')
    assert auth.user_workspaces('timo') == ['team2']
    assert not auth.revoke_workspace('timo', 'team1')


def test_grant_requires_existing_user_and_safe_name(users_file):
    with pytest.raises(ValueError):
        auth.grant_workspace('nobody', 'team1')

    auth.add_user('timo', 'secret')
    for workspace in ['team 1', '../evil', '..', '']:
        with pytest.raises(ValueError):
            auth.grant_workspace('timo', workspace)


def test_password_reset_keeps_workspaces(users_file):
    auth.add_user('timo', 'secret')
    auth.grant_workspace('timo', 'team1')

    auth.add_user('timo', 'new-password')

    assert auth.verify_user('timo', 'new-password')
    assert auth.user_workspaces('timo') == ['team1']


def test_corrupt_users_file_locks_nobody_in(users_file):
    users_file.write_text('not json', encoding='utf-8')

    assert not auth.verify_user('timo', 'geheim123')
    assert not auth.has_users()
