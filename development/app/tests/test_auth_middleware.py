from interface.auth_middleware import storage_request_allowed

GRANTED = ['team1', 'team2']


def test_non_storage_paths_are_not_restricted():
    assert storage_request_allowed('/', GRANTED)
    assert storage_request_allowed('/login', [])
    assert storage_request_allowed('/_nicegui/3.11.1/static/nicegui.js', GRANTED)


def test_granted_workspace_subtrees_allowed():
    assert storage_request_allowed('/storage/team1/project/offer/document.pdf', GRANTED)
    assert storage_request_allowed('/storage/team2/x.pdf', GRANTED)
    assert storage_request_allowed('/storage/team1', GRANTED)


def test_other_workspaces_denied():
    assert not storage_request_allowed('/storage/team3/project/offer/document.pdf', GRANTED)
    assert not storage_request_allowed('/storage/team1/x.pdf', ['team2'])
    assert not storage_request_allowed('/storage/team1/x.pdf', [])


def test_storage_root_denied():
    assert not storage_request_allowed('/storage', GRANTED)
    assert not storage_request_allowed('/storage/', GRANTED)


def test_traversal_is_always_denied():
    assert not storage_request_allowed('/storage/team1/../team2/document.pdf', GRANTED)
    assert not storage_request_allowed('/storage/team1/../team3/document.pdf', GRANTED)
    assert not storage_request_allowed('/storage/../users.json', GRANTED)
    assert not storage_request_allowed('/storage/team1/../../main.py', GRANTED)
