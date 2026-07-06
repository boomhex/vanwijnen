from pathlib import Path

from fastapi.responses import RedirectResponse
from nicegui import app, ui

from services import auth
from interface.session_panel import logout
from interface.theme import PRIMARY_RED
from utils.app_logging import log_action

LOGO = Path(__file__).resolve().parents[1] / 'images' / 'vw.png'


@ui.page('/workspaces')
def workspace_page() -> RedirectResponse | None:
    username = auth.normalize_username(app.storage.user.get('username'))
    if not auth.is_valid_username(username):
        app.storage.user.clear()
        return RedirectResponse('/login')

    ui.colors(primary=PRIMARY_RED)
    workspaces = auth.user_workspaces(username)
    current = app.storage.user.get('workspace')

    def open_workspace(workspace: str) -> None:
        if not auth.workspace_authorized(username, workspace):
            ui.notify('You no longer have access to this workspace', color='negative')
            return
        app.storage.user['workspace'] = workspace
        log_action('workspace_selected', workspace=workspace)
        ui.navigate.to('/')

    with ui.column().classes('absolute-center items-center gap-4'):
        if LOGO.exists():
            ui.image(LOGO).classes('w-24')
        with ui.card().classes('items-stretch gap-3 p-6 w-80'):
            ui.label('Choose a workspace').classes('text-xl font-bold self-center')
            ui.label(f'Logged in as {username}').classes('text-sm text-gray-500 self-center')

            if not workspaces:
                ui.label('No workspaces assigned to you yet.').classes('text-sm')
                ui.label('Ask the admin to run on the host:').classes('text-xs text-gray-500')
                ui.label(f'python -m services.auth grant {username} <workspace>') \
                    .classes('text-xs font-mono text-gray-500')

            for workspace in workspaces:
                button = ui.button(
                    workspace,
                    icon='folder_shared',
                    on_click=lambda workspace=workspace: open_workspace(workspace),
                ).props('no-caps')
                if workspace == current:
                    button.props('outline').tooltip('Current workspace')

            ui.button('Log out', icon='logout', on_click=logout).props('flat no-caps size=sm')
    return None
