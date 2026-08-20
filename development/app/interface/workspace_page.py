from pathlib import Path

from fastapi.responses import RedirectResponse
from nicegui import app, ui

from services import auth
from interface.main_page import PROJECTS_DIR
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

    def create_workspace() -> None:
        workspace = auth.normalize_workspace(new_workspace_input.value)
        if not auth.is_valid_workspace_name(workspace):
            ui.notify(
                'Ongeldige naam: gebruik 1-64 letters, cijfers, ".", "_" of "-", '
                'beginnend en eindigend met een letter of cijfer',
                color='negative',
            )
            return

        # Only claim genuinely new names here - joining a workspace someone else
        # already uses still goes through the beheerder's grant step, so access to
        # existing shared data isn't handed out just by guessing/typing its name.
        if auth.workspace_claimed(workspace) or (PROJECTS_DIR / workspace).exists():
            ui.notify('Werkruimte bestaat al, vraag toegang aan de beheerder.', color='negative')
            return

        auth.grant_workspace(username, workspace)
        log_action('workspace_created', workspace=workspace)
        open_workspace(workspace)

    with ui.column().classes('absolute-center items-center gap-4'):
        if LOGO.exists():
            ui.image(LOGO).classes('w-24')
        with ui.card().classes('items-stretch gap-3 p-6 w-80'):
            ui.label('Kies een werkruimte').classes('text-xl font-bold self-center')
            ui.label(f'Ingelogd als {username}').classes('text-sm text-gray-500 self-center')

            if not workspaces:
                ui.label('Er zijn nog geen werkruimtes aan u toegewezen.').classes('text-sm')
                ui.label('Maak hieronder een nieuwe werkruimte aan, of vraag toegang aan de beheerder.') \
                    .classes('text-xs text-gray-500')

            for workspace in workspaces:
                button = ui.button(
                    workspace,
                    icon='folder_shared',
                    on_click=lambda workspace=workspace: open_workspace(workspace),
                ).props('no-caps')
                if workspace == current:
                    button.props('outline').tooltip('Huidige werkruimte')

            ui.separator()
            ui.label('Nieuwe werkruimte aanmaken').classes('text-xs font-medium text-gray-500')
            with ui.row().classes('items-center gap-2 w-full no-wrap'):
                new_workspace_input = ui.input(placeholder='naam-werkruimte').classes('flex-grow').props('dense outlined')
                new_workspace_input.on('keydown.enter', create_workspace)
                ui.button(icon='add', on_click=create_workspace).props('dense round flat')

            ui.button('Uitloggen', icon='logout', on_click=logout).props('flat no-caps size=sm')
    return None
