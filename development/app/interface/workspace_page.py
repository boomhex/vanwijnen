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
            ui.notify('U heeft geen toegang meer tot deze werkruimte', color='negative')
            return
        app.storage.user['workspace'] = workspace
        log_action('workspace_selected', workspace=workspace)
        ui.navigate.to('/')

    with ui.column().classes('absolute-center items-center gap-4'):
        if LOGO.exists():
            ui.image(LOGO).classes('w-24')
        with ui.card().classes('items-stretch gap-3 p-6 w-80'):
            ui.label('Kies een werkruimte').classes('text-xl font-bold self-center')
            ui.label(f'Ingelogd als {username}').classes('text-sm text-gray-500 self-center')

            if not workspaces:
                ui.label('Er zijn nog geen werkruimtes aan u toegewezen.').classes('text-sm')
                ui.label('Neem contact op met de beheerder om toegang aan te vragen.') \
                    .classes('text-xs text-gray-500')

            for workspace in workspaces:
                button = ui.button(
                    workspace,
                    icon='folder_shared',
                    on_click=lambda workspace=workspace: open_workspace(workspace),
                ).props('no-caps')
                if workspace == current:
                    button.props('outline').tooltip('Huidige werkruimte')

            ui.button('Uitloggen', icon='logout', on_click=logout).props('flat no-caps size=sm')
    return None
