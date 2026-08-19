import asyncio
from pathlib import Path

from fastapi.responses import RedirectResponse
from nicegui import app, ui

from services import auth
from interface.theme import PRIMARY_RED
from utils.app_logging import log_action

LOGO = Path(__file__).resolve().parents[1] / 'images' / 'vw.png'


@ui.page('/login')
def login_page() -> RedirectResponse | None:
    if app.storage.user.get('authenticated', False):
        return RedirectResponse('/')

    ui.colors(primary=PRIMARY_RED)

    async def try_login() -> None:
        username = auth.normalize_username(username_input.value)
        if auth.verify_user(username, password_input.value):
            app.storage.user.update({'authenticated': True, 'username': username})
            log_action('login')
            workspaces = auth.user_workspaces(username)
            if len(workspaces) == 1:
                app.storage.user['workspace'] = workspaces[0]
                log_action('workspace_selected', workspace=workspaces[0])
                ui.navigate.to(app.storage.user.pop('referrer_path', '/'))
            else:
                ui.navigate.to('/workspaces')
            return

        log_action('login_failed', username=username)
        await asyncio.sleep(1)  # slow down password guessing
        ui.notify('Onjuiste gebruikersnaam of wachtwoord', color='negative')

    with ui.column().classes('absolute-center items-center gap-4'):
        if LOGO.exists():
            ui.image(LOGO).classes('w-24')
        with ui.card().classes('items-stretch gap-3 p-6 w-80'):
            ui.label('AI Offerte Vergelijking').classes('text-xl font-bold self-center')
            username_input = ui.input('Gebruikersnaam').props('outlined dense autofocus')
            username_input.on('keydown.enter', try_login)
            password_input = ui.input('Wachtwoord', password=True, password_toggle_button=True).props('outlined dense')
            password_input.on('keydown.enter', try_login)
            ui.button('Inloggen', on_click=try_login).props('no-caps')
            if not auth.has_users():
                ui.label('Er zijn nog geen accounts aangemaakt. Neem contact op met de beheerder.') \
                    .classes('text-xs text-gray-500')
    return None
